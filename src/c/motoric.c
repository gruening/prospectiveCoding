#define BASENAME "motoric"

#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

gsl_rng *r; //! random number generator.
#include "helper.h"
#include "paramsMotoric.h"

int main() {
    // set up random number generator:
    gsl_rng_env_setup();
    r = gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(r, RANDOM_SEED);

    // call service routine from paramsMotoric.h to set up frozen inputs
    initDerivedParams();
	
    // open files for writing output:

    //! file to store the network state
    FILE *outF = popen("gzip > data/" BASENAME ".gz", "w"); // state file.
    // header for state file:
    for(int i=0; i<N; i++) {
      fprintf(outF, "%s%u\t%s%u\t", "PSP", i, "Rate",i);
    }
    fputc('\n', outF);

    //FILE *wDF = popen("gzip > data/" BASENAME "WD.gz", "w");
    FILE *wF = popen("gzip > data/" BASENAME "W.gz", "w"); //! weight file
    // header for weight file:
    for(int i = 0; i < N; i++) {
      fprintf(wF, "Col%i\t", i);
    }
    fputc('\n', wF);


    // so for the simulation we use a different random generator.
    gsl_rng_set(r, 0); // \todo attention here -- the seed it
    // reset, so the above random seed is only 
    // used for initialisation? \todo seed must be diffewrent from
    // above and was there.
	
    gsl_vector *psp = gsl_vector_calloc(N); //! postsynaptic
    //! potentials (or post?)
    gsl_vector *pspS = gsl_vector_calloc(N);
    //! positive part for Euler integration presynaptic potential.
    gsl_vector *sue = gsl_vector_calloc(N); 
    //! dito, negative part
    gsl_vector *sui = gsl_vector_calloc(N);
    gsl_vector *pspTilde = gsl_vector_calloc(N);
    //! weight matrix N by N
    gsl_matrix *w  = gsl_matrix_calloc(N, N);
    //! vector of membran potentials.
    gsl_vector *u  = gsl_vector_calloc(N);
    //! vector of membran potentials as if calcululated from
    //! dendritic potential only
    gsl_vector *uV  = gsl_vector_calloc(N);
    //! what is this for?
    gsl_vector *pre  = gsl_vector_calloc(N); 
    //! instantaneous rate from u
    gsl_vector *rU  = gsl_vector_calloc(N);
    //! instantaneous rate as though from uV only?
    gsl_vector *rV  = gsl_vector_calloc(N);

    // pointers to the data structures of the above.
    double *pspP = gsl_vector_ptr(psp,0);
    double *pspSP = gsl_vector_ptr(pspS,0);
    double *sueP = gsl_vector_ptr(sue,0);
    double *suiP = gsl_vector_ptr(sui,0);
    double *pspTildeP = gsl_vector_ptr(pspTilde,0);
    double *wP = gsl_matrix_ptr(w,0,0);
    double *uP = gsl_vector_ptr(u,0);
    double *uVP = gsl_vector_ptr(uV,0);
    double *preP = gsl_vector_ptr(pre,0);
    double *rUP = gsl_vector_ptr(rU,0);
    double *rVP = gsl_vector_ptr(rV,0);


    gsl_vector_view tmpv;  // to keep a row of weight from w;
    double gE;
    double wij; 
    // double wd; // helper war to accumulate all weight changes.
    double 	uI;
	
    // start der simulation, last additional training cycles is
    // for what?
	
    for( uint s = 0; s < TRAININGCYCLES; s++) {
      fprintf(outF, "# Trainingcycle %u\n", s);
	
//	wd = 0;
	for( int t = 0; t < TIMEBINS; t++) {
	  fprintf(outF, "# Timebin %u\n", t);
	    for( int i = 0; i < N; i++) {
		// update psps and determin whether a we have an incomping spike from rUP
		// (owr own??) 
		updatePre(sueP+i, suiP+i, pspP + i, pspSP + i, pspTildeP + i, *(preP + i) = spiking(*(rUP+i), gsl_rng_uniform_pos(r))); 
		tmpv = gsl_matrix_row(w, i); // extract weight vector
					     // of incoming synapeses for this neuron.
	    
		// set gE to some of external drive
		if(i >= MOTORIC_START && i < MOTORIC_END) {
		    gE = *(motoric_GE + t*N + i);
		}
		else {// auditory neurons 
		    gE = *(auditory_GE + t*N + i);
		}

		updateMembrane(uP+i, uVP+i, &uI, &tmpv.vector, psp, gE, 0);
		*(rUP+i) = phi(*(uP+i)); *(rVP+i) = phi(*(uVP+i));
	    
		// now go through all neurons a for the second index j
		for(int j = 0; j < N; j++) {
		    wij = *(wP + i*N + j); // weight before updating it this round.
		    if(i != j) { // no self-connections
			updateWeight(wP + i*N + j, *(rUP+i), *(pspTildeP+j), *(rVP+i), *(pspSP+j));
			// wd += (wij -  *(wP + i*N + j)) * (wij -  *(wP + i*N + j)); // square
			// of
			// total
			// weight change
		    }
		    // in the last timebin of the last training cycle, printout trained weights
		    // i is row and j is columns
		    if( s == TRAININGCYCLES - 1 && t == TIMEBINS - 1) {
			fprintf(wF, "%g\t", wij);
		    }
		}
		if( s == TRAININGCYCLES - 1 && t == TIMEBINS - 1) {
		fputc('\n', wF);
		}
	    }
	    // write out the state (preP, rUP), for each timebin and each cycle in a row

	    for(int i=0; i<N; i++) {
		fprintf(outF, "%g\t%g\t", preP[i], rUP[i]);
	    }
	    fputc('\n', outF);
	}
	// for each simulation run write 
	//fwrite(&wd, sizeof(double), 1, wDF); // print total weight
	// change travelled in
	// one run.
    }

    pclose(outF); 
    // pclose(wDF); 
    pclose(wF); 
	
    return errno;
}
