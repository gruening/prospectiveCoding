#include <math.h>
#include <float.h>

// experiment parameters
#define RANDOM_SEED 421251 //! seed for global random number generator
#ifndef TRAININGCYCLES
	#define TRAININGCYCLES 100	//! number of training cycles
#endif
#define DURATION 6000			//! duration of simulation in timesteps in ms
#define NPRE 2000				//! number of presynaptic neurons
#define NUDGE_AT_US .005		//! excitatory nudging conductance in ms^-1
#define TAU_OU 1000				//! time constant of template OU in ms
#define TAU_OU2 100				//! time constant of individual run OU in ms
#define CS_START 1000			//! start time of conditioned stimulus
#define CS_END 2000				//! end of CS 
#define US_START 3000			//! start of CS
#define US_END 4000			//! end of CS
#define MIX_LOW1 0.1			//! xxxx mixing parameter end of CS (unitless)
#define MIX_LOW2 0.2			//! xxxx mixing parameter beginning of US (unitless)

//derived -- xxx -- what are these?
double GAMMA_OU;
double GAMMA_OU2; 
double SIGMA_OU; //! holds variance of OU process used for template rate trajectories (xxx)
double SIGMA_OU2; //! holds variance of OU process used for actual sampling trajectories (xxx)
int TIMEBINS;
double *OU1;
double *OU2;
double *MIX;
double *I1, *I2; //! to hold the two different time courses of input activations.

/**
   @param newOU
   @return 
 */
void mixOUs(double * restrict newOU, double oldOU, double mix, double * restrict mixedOU) {
	*mixedOU = runOU(*mixedOU, (1. - mix) * oldOU-.5 * mix, GAMMA_OU2, mix * SIGMA_OU2) ;
} 

/** init simulation specific data structures etc */
void initDerivedParams() {
  GAMMA_OU = exp(- DT/TAU_OU); // decay rate of OU process??
  GAMMA_OU2 = exp(- DT/TAU_OU2);
  SIGMA_OU = sqrt(2 * DT / TAU_OU);
  SIGMA_OU2 = sqrt(2 * DT / TAU_OU2); 
  TIMEBINS = DURATION/DT; //! number of time bins with resolution DT
	 
  OU1 = malloc(TIMEBINS * NPRE * sizeof(double)); // to hold pregenerated noise inputs
  OU2 = malloc(TIMEBINS * NPRE * sizeof(double)); // to hole pregenerate noisy inputs
  for( int i = 0; i < NPRE; i++) {
    OU1[i] = gsl_ran_gaussian_ziggurat(r,1);
    OU2[i] = gsl_ran_gaussian_ziggurat(r,1);
    for( int t = 1; t < TIMEBINS; t++) {
      // file OU1 and OU2 with pregenerated noise
      OU1[t * NPRE + i] = runOU(OU1[(t-1) * NPRE + i], 0, GAMMA_OU, SIGMA_OU);
      OU2[t * NPRE + i] = runOU(OU2[(t-1) * NPRE + i], 0, GAMMA_OU, SIGMA_OU);
    }
	 }
	 
	 MIX = malloc(TIMEBINS * sizeof(double));
	 I1 = malloc(TIMEBINS * sizeof(double)); //! input pattern standing for US
	 I2 = malloc(TIMEBINS * sizeof(double));  //! input pattern standing for no US
	 for(int t = 0; t < TIMEBINS; t++) {
	   I2[t] = 0; // no inputs for no US

	   // switch on current I1 for US stimulus
	   if(t < US_START / DT || t > US_END / DT) I1[t] = 0; else I1[t] = NUDGE_AT_US * DT;

	   // before CS stimulus
	   if(t < CS_START / DT) {
	     MIX[t] = 1.;
		 }
		 // during CS stimulus
		 else if(t < CS_END / DT) {
			 MIX[t] = MIX[t-1] - DT * (1. - MIX_LOW1) / (CS_END - CS_START); 
		 }
		 // between CS and US
		 else if(t < US_START / DT) {
			 MIX[t] = MIX[t-1] + DT * (MIX_LOW2 - MIX_LOW1) / (US_START- CS_END);
		 } 
		 // during and after the US
		 else MIX[t] = MIX[t-1] + DT * (1. - MIX_LOW2) / (US_END - US_START);
	 }	 
}
