#include <math.h>
#include <float.h>

// global experimental parameters

/*! random seed this time, but this is overriden by a later call with 0, 
  seems to be used only for the initialisation. */    
#define RANDOM_SEED 44635842 
#define DURATION 100				//! duration of one run in [ms]
#define TRAININGCYCLES 10 //! number of repetitions of experiment.
#define TIMEBINS DURATION*BINS_PER_MS //! number of timebins needed for resolution DT

// definition of the network size and structure

#define MOTORIC_START 0
#define N_MOTORIC 1000 // number of motoric neurons
#define MOTORIC_END MOTORIC_START + N_MOTORIC

#define AUDITORY_START MOTORIC_END
#define N_AUDITORY 1000 // number of auditory neurons
#define AUDITORY_END AUDITORY_START + N_AUDITORY

//! total number of neurons in the network
#define N AUDITORY_END

// *****
#define NUDGE_AT_US .02				//! ms^-1, somatic current for activation.


// global vars to contain the "data" on which the simulation acts

//! to contain the random patterns that drive the motoric neurons during babbling
double motoric_GE[TIMEBINS * N_MOTORIC]; 
//! to contain the auditory equivalents of the motoric patterns.
double auditory_GE[TIMEBINS * N_AUDITORY]; // __attribute__(noinit);

// 2 pies!			  
const double omega = 6.283185307179586/DURATION;

/** initialises parameters specific for this simulation **/
void initDerivedParams() {

  //! initialise "random drivers" for the motoric patterns, currenly
  //! all are doing the same. What the current or potential difference
  //! actually transmitted via GE?
  for(int t = 0; t < TIMEBINS; t++) {
    for(int i = 0; i < N_MOTORIC; i++ ) {
      motoric_GE[t*N_MOTORIC + i] = .006 * 
	(1 - sin(omega * t * DT) * sin(2 * omega * t * DT) * cos(4 * omega * t * DT)) * DT; 
      // what does a pattern for
      // a conductance based
      // neuron look like -- it
      // is just the pattern
      // transfered via the soma synapse
    }
  }	

  //! potential that the auditory neurns accually gets:
  for(int t = 0; t < TIMEBINS; t++) {
    for(int i = 0; i < N_AUDITORY; i++ ) {
#if N_MOTORIC != N_AUDITORY 
#error "Must be same currently" 
#endif


#define DELAY_EXTERNAL 50 //! [ms] delay of closing external the
			  //! moto-auditory loop
      // rewrite this as an in-patterns generator!
      if(t < DELAY_EXTERNAL*BINS_PER_MS) { // do to: loop through it, so that
				  // we hear some thing
	auditory_GE[t*N_AUDITORY +i] = 0; // we don´t hear anything yet
      }
      else {
	// we hear the delayed motoric pattern driving us. // todo: is
	// the below correct?
	auditory_GE[t*N_AUDITORY + i] = motoric_GE[(t-DELAY_EXTERNAL*BINS_PER_MS)*N_MOTORIC + i]; 
      }
    }
  }											 
}
