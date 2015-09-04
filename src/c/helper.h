#include <math.h>

#define DT .1	//! time step [ms] for forward Euler integration

// parameters of neural dynamics
#define GD 1.8	//! [ms^-1 = mS/muF] two-way conductance between dendrite and soma
#define GL .1   //! [ms^-1 = mS/muF] leak conductance of some.
#define GS .3	//! [ms^-1 = mS/muF] "conductance" aka inverse timescale of synaptic response decay
#define EE 4.666666666666666666666667		//! [unitless] excitatory reversal potential
#define EI -.333333333333333333333333		//! [unitless] inhibitory reversal potential

#ifndef TAU
#define TAU 9	//! time constant for 2nd low-pass filtering of
		//! PSP. This is for the elgibility trace, but why is
		//! TAU then 9ms < TAU_MEMBRANE of 10ms? \todo
#endif

// parameters of synaptic dynamics
#ifndef ETA
	#define ETA .5	  //! [unitless] learning rate
#endif
#ifndef TAU_ALPHA
	#define TAU_ALPHA 600					//! effective ramp-up time constant in ms xxx
#endif

// spiking parameters
#define PHI_MAX .06  // [kHz] asymptotic / maximal spike rate

//derived -- in units commensurate with DT
#define	GD_DT GD * DT						//! unitless
#define	GL_DT GL * DT						//! unitless
#define	GAMMA_GL exp(- GL * DT)	   //! decay of soma (and also post-synaptic) potential in one DT through GL 
#define	GAMMA_GL_GD exp(- (GL+GD) * DT)		//! decay of soma potential in one DT through GL and GD
#define	GAMMA_GS exp(- GS * DT)				//! decay of synaptic potential through GS
#ifdef CURRENTPRED //! flag xxx set in makefile
#define GAMMA 0	//! 2nd lowpass filter for PSP swithed off
#define ALPHA 1	//! unitless xxx
#else
#define	GAMMA exp(- DT/TAU) //! decay factor of 2nd lowpass filter for
			    //! PSP
//! "factor of potentiation" xxx \todo look up what is really meant by
//! it and what significance it has.
#define	ALPHA (1. - exp(- DT * (1./TAU - 1./TAU_ALPHA))) 
#endif
//! prefactor $c= tau_m - tau_s$ that normalises the spike response kernel \kappa
#define	C_PSP GL * GS/(GS-GL)
//! learning rate commensurate with time step DT [ms]
#define ETA_DT ETA * DT						

/** step-wise solve soma potential of an Urbanczik-Senn neuron
 * according to eq (1) by Euler integration with stepsize DT.
 @param u soma potential
 @param uV soma potential as would arise only from dendritic inputs
 @param uI some potential as would arise only from somatic inputs
 @param w weight vector of incoming synapses
 @param psp vector of PSPs of incoming synapses
 @param gE conductance of excitory somatic synpase [in units of DT?]
 @param gI conductance of inhibitory somatic synpaes [in units of DT xxx]
 */
void updateMembrane(double * u, double * uV, double * uI, gsl_vector *w, gsl_vector *psp, double gE, double gI) {
	double v; // to hold gross dendrite potential

// dendritic potential is weighted some of all its PSPs, see eq (7)
	gsl_blas_ddot(w, psp, &v); 

	/* Euler integration of eq (1). 
	   @todo Should gE and gI also be prefixed with DT? 
	   @todo Should GAMMA_GL_GD be replace by approximation exp(x)
	   = 1+x just as done for the second term?
	*/
	*u = GAMMA_GL_GD * *u + GD_DT * v + gE * (EE - *u) + gI * (EI - *u);
	// potential if we just had the somatic inputs
	*uI = GAMMA_GL_GD * *uI +  gE * (EE - *u);
	// potential as though we just had the dendritic inputs
	*uV  = GAMMA_GL_GD * *uV + GD_DT * v;
}

/**
   step-wise calculate PSPs according to spike responsee kernel \kappa
   (between eqs (7) and (8) by Euler integration. Kernel \kappa
   consists of two terms -- both are integrated and being kept track
   of separatly (sue and sui)

   @param sue "excitatory" part of PSP for Euler integration, coming from first term of \kappa
   @param sui "inhibitory" part of PSP for Euler integration, coming from first term of \kappa
   @param psp postsynaptic potential
   @param pspS 1st filtered psp
   @param pspTilde 2nd filtered psp
   @param pre value 1.0 if there an incoming spike at this instance in
   time, 0.0 otherwise
				   @todo rename update PSP?
 */
void updatePre(double * sue, double * sui, double * psp, double * pspS, double * pspTilde, double pre) {

	// Euler integral of first term of \kappa; decays with GAMMA_GL (1/tau_m)
	*sue = GAMMA_GL * *sue + pre; 

	// Euler integral of second term of \kappa; decays with GAMMA_GL (1/tau_m)
	*sui = GAMMA_GS * *sui + pre;

	// integrated PSP is sum of sue and sui (times kernel normalisation)
	*psp = C_PSP * (*sue - *sui);

	// 1st low-pass filtered intergrated PSP with time scale in the
	// order of the total soma leak, xxx? switched off here.
	// ie with the soma membrane time constant?
	*pspS = *psp; //GAMMA_GL_GD * *pspS + GD_DT * *psp;

	// 2nd low-pass filtered PSP with time scale set by
	// GAMMA. xxx, not normalised
	// this is the eligibility trace that decay with 
	// \todo normalisation missing?
	*pspTilde = GAMMA * *pspTilde + *pspS;
}

/** intergrate synaptic weights according to eq (3) via Euler Integration.
     @param w vector of synaptic weights
     @param rU intantaneous firing rate based on soma potential
     @param pspTilde low-pass filtered PSPs
     @param rV would-be instantaneous firing rate based on
     (attenuated) dendritic potential allone.
     @param psp current PSP
     @todo look up in Brea's paper or the Uranczik and Senn paper,
     what ALPHA really means.
 */
void updateWeight(double * w, double rU, double pspTilde, double rV, double psp) {
	*w += ETA_DT * (ALPHA * rU * pspTilde - rV * psp);
}

/** calculate instantaneous spike rate from soma potential.
    Here a piece-wise linear function is used as this calculates
    faster than an exponential sigmoid.
    @todo why is DT part of the full formula? Should it not e part of
    the approximated formula as well?
    @param u [units?] soma potential.
    @return instansaneous spike rate in units of [?] / DT?
 */
double phi(double u) {
	if(u < 0) return 0;
	if(u > 1) return PHI_MAX;
	return PHI_MAX * u;
	//return PHI_MAX * DT / (1 + K * exp(BETA * (THETA - u)));
}


/**
   emulate point process to generate a spike.
   @param rate instaneous firing rate [units xxx], range
   @param rn a random number in the range of [xxx]
   @return if rn <= rate 1.0 (ie a spike is generated), 0.0 otherwise
*/
double spiking(double rate, double rn) {
	if( rn <=  rate) return 1.;
	return 0.;
}

/**
   some form of noise: OU process
   @param ou  current value of OU process?
   @param gamma scaling of process?
   @param s stddev of change
   @param m mean of raw change
   @return

   uses global random r.
*/
double runOU(double ou, double m, double gamma, double s) {
	return gamma * ou + (1.-gamma) * m + s * gsl_ran_gaussian_ziggurat(r,1);
}
