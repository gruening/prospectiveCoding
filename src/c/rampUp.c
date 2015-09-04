#include <stdio.h>
#include <time.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_randist.h>

gsl_rng *r; //! to hold global random generator

#include "helper.h"
#include "paramsRampUp.h"

//! how to open the file for xxx, default is for writing of binary data.
#ifndef FILEPOST_FLAG
	#define FILEPOST_FLAG "wb"
#endif

/**
 main simulation loop
*/
int main() {

  // init own parameters.
  initDerivedParams(); 

  // init random generator
  gsl_rng_env_setup();
  r = gsl_rng_alloc(gsl_rng_default);
  gsl_rng_set(r, SEED_MAIN);

  // file handle for xxx file
  FILE *postF = fopen(FILENAME_POST, FILEPOST_FLAG);

  // file handle for xxx file
  FILE *preF = fopen(FILENAME_PRE, "wb");
	
  // set up vectors:

  // to hold post synaptic potentials [unused??]
  gsl_vector *psp = gsl_vector_alloc(NPRE);
  // to hold post synaptic potentials 1st filtered
  gsl_vector *pspS = gsl_vector_alloc(NPRE);
  // to hold "excitatory" part of psp for Euler integration
  gsl_vector *sue = gsl_vector_alloc(NPRE);
  // to hold "inhibitory" part of psp for Euler integration
  gsl_vector *sui = gsl_vector_alloc(NPRE);
  // to hold psp 2nd filter
  gsl_vector *pspTilde = gsl_vector_alloc(NPRE);
  // to hold weights
  gsl_vector *w  = gsl_vector_alloc(NPRE);
  // to hold xxx
  gsl_vector *pres  = gsl_vector_alloc(NPRE);

  // ?? ou XXX \todo
#ifdef PREDICT_OU
  gsl_vector *ou = gsl_vector_alloc(N_OU);
  gsl_vector *preU = gsl_vector_calloc(NPRE);
  gsl_vector *wInput = gsl_vector_alloc(N_OU);
  gsl_matrix *wPre  = gsl_matrix_calloc(NPRE, N_OU);
  double *preUP = gsl_vector_ptr(preU,0);
  double *ouP = gsl_vector_ptr(ou,0);
  double *wInputP = gsl_vector_ptr(wInput,0);
  double *wPreP = gsl_matrix_ptr(wPre,0,0);
#endif

  // get pointers to array within the gsl_vector data structures above.
  double *pspP = gsl_vector_ptr(psp,0);
  double *pspSP = gsl_vector_ptr(pspS,0);
  double *sueP = gsl_vector_ptr(sue,0);
  double *suiP = gsl_vector_ptr(sui,0);
  double *pspTildeP = gsl_vector_ptr(pspTilde,0);
  double *wP = gsl_vector_ptr(w,0);
  double *presP = gsl_vector_ptr(pres,0);

  for(int i=0; i<NPRE; i++) {

    // init pspP etc to zero
    *(pspP+i) = 0;
    *(sueP+i) = 0;
    *(suiP+i) = 0;
#ifdef RANDI_WEIGHTS
    // Gaussian weights
    *(wP+i) = gsl_ran_gaussian(r, .1);
#else
    *(wP+i) = 0;
#endif
  }


  //! OU \todo what for?	
#ifdef PREDICT_OU
  for(int j=0; j < N_OU; j++) {
    *(ouP + j) = gsl_ran_gaussian(r, 1) + M_OU;
    *(wInputP + j) = gsl_ran_lognormal(r, 0., 2.)/N_OU/exp(2.)/2.;
    for(int i=0; i < NPRE; i++) *(wPreP + j*NPRE + i) = gsl_ran_lognormal(r, 0., 2.)/N_OU/exp(2.)/2.;
  }
#endif

  // temp variables for the simulation yyyy
  double 
    u = 0, // soma potential.
    uV = 0, // some potential from dendrite only (ie discounted
	    // dendrite potential
    rU = 0, // instantneou rate 
    rV = 0, // rate on dendritic potential only
    uI = 0, // soma potential only from somatic inputs
    rI = 0, // rate on somatic potential only
    uInput = 0; // for OU?

  // run simulatio TRAININGCYCLES number of times
  for( int s = 0; s < TRAININGCYCLES; s++) {

    // for all TIMEBINS
    for( int t = 0; t < TIMEBINS; t++) {

#ifdef PREDICT_OU
      for(int i = 0; i < N_OU; i++) {
	*(ouP+i) = runOU(*(ouP+i), M_OU, GAMMA_OU, S_OU);
      }
      gsl_blas_dgemv(CblasNoTrans, 1., wPre, ou, 0., preU); 
#endif

      // update PSP of our neurons for inputs from all presynaptic neurons
      for( int i = 0; i < NPRE; i++) {

#ifdef RAMPUPRATE
	/** just read in the PRE_ACT and generate a spike and store it in presP -- so PRE_ACT has inpretation of potential */
	updatePre(sueP+i, suiP+i, pspP + i, pspSP + i, pspTildeP + i, *(presP + i) = spiking(PRE_ACT[t*NPRE + i], gsl_rng_uniform(r)));

#elif defined PREDICT_OU
	//*(ouP+i) = runOU(*(ouP+i), M_OU, GAMMA_OU, S_OU); // why commented out?
	updatePre(sueP+i, suiP+i, pspP + i, pspSP + i, pspTildeP + i, *(presP + i) = DT * phi(*(preUP+i)));//spiking(DT * phi(*(preUP+i)), gsl_rng_uniform(r))); // why commented out?

#else
	// PRE_ACT intepreated as spikes
	updatePre(sueP+i, suiP+i, pspP + i, pspSP + i, pspTildeP + i, *(presP + i) = PRE_ACT[t*NPRE + i]);
#endif
      } // endfor NPRE

#ifdef PREDICT_OU
      gsl_blas_ddot(wInput, ou, &uInput);
      GE[t] = DT * phi(uInput);

#endif
      // now update the membrane potential.
      updateMembrane(&u, &uV, &uI, w, psp, GE[t], GI[t]);


      // now calculate rates from from potentials.
#ifdef POSTSPIKING // usually switch off as learning is faster when
		   // learning from U
      // with low-pass filtering of soma potential from actual
      // generation of spikes (back propgating dentric spikes?
      rU = GAMMA_POSTS*rU + (1-GAMMA_POSTS)*spiking(DT * phi(u),  gsl_rng_uniform(r))/DT;
#else
      // simpler -- direct.
      rU = phi(u); 
#endif
      rV = phi(uV); rI = phi(uI);

      // now update weights based on rU, RV, the 2nd filtered PSP and
      // the pspSP
      for(int i = 0; i < NPRE; i++) {
	updateWeight(wP + i, rU, *(pspTildeP+i), rV, *(pspSP+i));
      }
#ifdef TAUEFF
      /**
	 write rU to postF, but only for the last run of the
	 simulation and then only before the STIM_ONSET time --
	 ie it is the trained output without somatic drive.
       */
      if(s == TRAININGCYCLES - 1 && t < STIM_ONSET/DT) {
	fwrite(&rU, sizeof(double), 1, postF); 
      }
#else
      /**
	 for every 10th training cycle write all variables below to
	 postF in order:
       */
      if(s%(TRAININGCYCLES/10)==0) {
	fwrite(&rU, sizeof(double), 1, postF);
	fwrite(GE+t, sizeof(double), 1, postF);
	fwrite(&rV, sizeof(double), 1, postF);
	fwrite(&rI, sizeof(double), 1, postF);
	fwrite(&u, sizeof(double), 1, postF);
      }
      if(s == TRAININGCYCLES - 1) {
#ifdef RECORD_PREACT
	// for the last cycle also record the activity of the
	// presynaptic neurons
	fwrite(PRE_ACT + t * NPRE, sizeof(double), 20, preF);
	//fwrite(ouP, sizeof(double), 20, preF);
	fwrite(presP, sizeof(double), 20, preF);
#else
	// and the 1st and 2nd filtered PSP
	fwrite(pspSP, sizeof(double), 1, preF);
	fwrite(pspTildeP, sizeof(double), 1, preF);
#endif
      }
#endif
    }
  }
  
  fclose(preF);
  fclose(postF);
  
  return 0;
}
