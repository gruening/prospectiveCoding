#include <math.h>
#include <float.h>

// experiment parameters
#define RANDOM_SEED 44635842 //! random seed this time, but this is overriden by a later call with 0.
#define DURATION 400				//! in the manuscropt it is 800ms? xxx
#define N 200 //! number of neurons in the network
#define NGROUPS 4 //! neurons in 4 Groups 
#define NUDGE_AT_US .02				//! ms^-1, somatic current for activation.
#define TRAININGCYCLES 300 //! number of repetitions of experiment.


//derived
#define TIMEBINS DURATION/DT //! number of timebins needed for resolution DT

double *GE;							//! unitless,
								//! vector
								//! of
								//! driving somativ input (already multiplied by DT) 

void initDerivedParams() {
  GE = malloc(TIMEBINS * N * sizeof(double)); // GE individual for
					      // each neuron and timebin.
	for(int t = 0; t < TIMEBINS; t++) {
	  // prepare input pattern for nudging, sequentially activate
	  // each group.
		for(int i = 0; i < N; i++ ) {
			if(i/(N/NGROUPS) <= t*DT/(DURATION/NGROUPS) && i/(N/NGROUPS) + 1 >= t*DT/(DURATION/NGROUPS)) *(GE + t*N + i) = NUDGE_AT_US * DT;
			else *(GE + t*N + i) = 0;
		}
	}	
}
