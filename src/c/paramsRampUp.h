#include <math.h>
#include <gsl/gsl_rng.h>


#ifndef DURATION
	#define DURATION 2000	//! duration [ms] of one simulation run
#endif			

#ifndef NPRE	
	#define NPRE 2000  //! number of presynaptic neurons
#endif

//! [ms] onset of teaching clamping in ms (ie injection of somatic current) for experiments in Fig xxx
#define STIM_ONSET 1800

#ifndef INPUT_AT_STIM				
	#define INPUT_AT_STIM .015	//! input current in [ms^-1] applied from time STIM_ONSET
#endif

#ifndef TRAININGCYCLES
   #define TRAININGCYCLES 100     //! number of repetitions of simulation run
#endif

//! time constant for low-pass filtering soma potential [ms]
#define TAU_POSTS 10			//! ms xxx, sth with
					//! postsynaptic? (again in
					//! the order of membrane time constant


#define GAMMA_POSTS exp(- DT/TAU_POSTS) 
//! decay factor of low-pass filter for soma potential when generating spikes
#ifndef GAMMA_POSTS  
#endif


#define SEED_MAIN 230100138 //! seed for random generator for reproducibility for main simulation
#define SEED_PARAM 2389120  //! seed for random generator used to generate the input patterns

#define M_OU 1
#define TAU_OU 400
#define N_OU 1
#define GAMMA_OU exp(- DT/TAU_OU)
#define S_OU sqrt(2 * DT / TAU_OU)

#define TIMEBINS DURATION/DT //! number of timemins to sort what into??

//! the exhitatory somatic conductance in "units of DT"
double *GE;
//! the inhibitory somatic conductance in "units of DT"
double *GI; 
/*! to hold precalculated input activation patterns (depending on
  precise simulation setup either to be interpretated as potential of
  presynaptic neurons or an actual spike */ 
double *PRE_ACT;

/**
   Initialise parameters only for this simulation setting
*/
void initDerivedParams() {

  //! separate random number generator for setting up the input spike patterns.
	gsl_rng *r;
	gsl_rng_env_setup();
	r = gsl_rng_alloc(gsl_rng_default);
	gsl_rng_set(r, SEED_PARAM);
	 
	//! allocate a GE for each timebin
	GE = malloc(TIMEBINS * sizeof(double)); 

	//! allocate a GI for each timebin
	GI = malloc(TIMEBINS * sizeof(double)); 

	//! precalculate the input activations.
	PRE_ACT = malloc(TIMEBINS * NPRE * sizeof(double)); 
	for(int t = 0; t < TIMEBINS; t++) {
#if defined ROTATIONS
	  // GE is varied wave-like in units of [DT] right from the start
	  double omega = 6.283185307179586/DURATION;
	  GE[t] = .006 * (1 - sin(omega * t * DT) * sin(2 * omega * t * DT) * cos(4 * omega * t * DT)) * DT;
#elif defined RAMPUPRATE
	  // GE is only swithed in after 2/3 of total simulation time -- to simulate the ramp up.
	  if(t < 2*DURATION/DT/3) GE[t] = 0;
	  else GE[t] = INPUT_AT_STIM * DT; // again in units of DT
#else 
	  // stimulus is switched on at STIM_ONSET -- to simulate xxx
	  if(t > STIM_ONSET/DT) GE[t] = INPUT_AT_STIM * DT;
	  else GE[t] = 0;
#endif
	  GI[t] = 0; // no inhibitory in put
#ifdef CURRENTPRED
	  GI[t] = 4 * GE[t]; // strong inhibitory input only if we do CURRENT-lgsl -lgslcblas -lmPRED xxx
#endif
		 
#ifdef FROZEN_POISSON
	  // precalc poission inputs with rate 0.20 kHz, PRE_ACT as intepretation as spike
	  for(int i = 0; i < NPRE; i++) {
	    if(gsl_rng_uniform(r) < .02 * DT) {
	      PRE_ACT[NPRE * t + i] = 1;
	    } else {
	      PRE_ACT[NPRE * t + i] = 0;
	    }
	  }
#elif defined RAMPUPRATE
	  // repeat activations of first third of duration in the second and third third. PRE_ACT has interpretation as potential.
	  for(int i = 0; i < NPRE; i++) {
	    if(t%((int)(DURATION/DT)/3) == 0) 
	      PRE_ACT[NPRE * t + i] = gsl_rng_uniform(r) * PHI_MAX/2 * DT; // this time with rate PHI_MAX / 2
	    else 
	      PRE_ACT[NPRE * t + i] = PRE_ACT[NPRE * (t-1) + i];
	  }
#else 
	  // one input at a time -- PREACT as integpreation as spike.
	  for(int i = 0; i < NPRE; i++) {
	    if(i == t * DT) {
	      PRE_ACT[NPRE * t + i] = 1;
	    } else {
	      PRE_ACT[NPRE * t + i] = 0;
	    }
	  }
#endif
	} // endfor t (TIMEBINS	
}
