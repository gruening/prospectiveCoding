#!/usr/bin/python

# \todo: - how is rand initialised -- it seems always the same sequence? Seed it?
# \todo: - sth is rotton with the plotting. Errors stay the same
#          between HVC-driven and Tutor song since the introduction of
#           copy learning

import numpy as np
from scitools.std import *

def delayperm(x,n):  # Circles the columns of matrix x to the right by n columns. 
    return np.roll(x, n, axis=1);

# Learning of the inverse and then the predictive inverse model for song generation. 
# Based on discussions with Richard and Surya, Jan 14, 2011. Bugs etc
# attributed to Walter (and now also Andre) 
# 
# Shared with Andre Gruning, 2015/10/19. Changes:
# - Oct 2015: updating comments
# - Oct 2015:  translate to pyhton.
# - 2015/11/02: replace random drive with fixed drive.
# - 2015/11/02: rename variable to better fit our current terminology
# - 2015/11/02: change from random drive of RA during inversion
#               learning to drive with one fixed pattern.
# - 2015/11/02: add error between actual sound and predicted sound via
#               LMAN -- this is lower than the error between the
#               motoric representations. -- Why? -- perhaps because
#               matrix S is contraxcting?
# - 2015/11/02  switch driving from direct injection into RA to
#               injection via HVC -- leads to same error of
#               causal-inverse learning
# - 2015/11/02  start weight copying in phase 2 -- problem is still that the overall error does not go down.
# - 2015/11/02  HVC branch: implement HVC as a clock.
# - 2015/11/10  rectify plots and legends
#                

# number of motor neurons (RA)
n_ra = 50; # 3 for testing, or 7, n_ra=200; , currently 50 for testing
# on train

# dimension of song, ie number of "acoustic" degrees of freedom"
n_sound = n_ra; 

# number of auditory neurons which receive the sound and convert into neural activity.
n_aud = n_ra; 

# number of auditory memory neurons which memorize the tutor song (for one time step)
n_lman = n_ra; 

# duration of subsongs and of tutor song  [ms?]
T = 100; 

# Number of HVC clock neurons -- one time step is 10ms

n_hvc = T;

# Delay of auditory activity with respect to the generating motor activity (syrinx + auditory pathway) [ms?] -- isnt that rather in the area of 50ms-70ms?
tau = 7; 

# learning steps for model 
n_learn = 300; 

eta_lman = 0.002; # learning rate for inverse model via lman -- that
# seems to be sufficient
eta_hvc = 0.01; # learning rate for weight copying orig = 0.001

epsi = 1/1000; # Parameter to regularize the matrices to avoid near
# zero EV -- bad for inversion.

# syrinx; converts the RA motor activity signal m into a sound (here just matrix)
S = (np.random.randn(n_sound,n_ra) + epsi*np.eye(n_ra)) / sqrt(n_ra);

# auditory pathway; converts the song into an auditory signal for aud
# neurons (here just a matrix)
A = (np.random.randn(n_aud,n_sound) + epsi*np.eye(n_ra)) / sqrt(n_aud); 

# Total motor to auditory transformation. 
Q=A.dot(S); 

# acoustic representation of tutor song -- this what we start from.
song_sound_tut = np.random.randn(n_ra,T); 

# auditory representation of tutor song auditory -- generated by the 
# tutor song, or by rehearsing of the memory this is our imprint.
song_aud_tut = A.dot(song_sound_tut);

# initial weight matrix converting the auditory representation from
# LMAN into RA motor activity (inverse model) 
w_lman = np.random.randn(n_ra,n_aud) / sqrt(n_aud); 

# initial weight matrix from HVC to RA:
#w_hvc = np.random.randn(n_ra,n_lman) / sqrt(n_lman); 

# some size as length of song.
w_hvc = np.random.randn(n_ra, n_hvc);
hvc_soma = np.eye(n_hvc, T);

# Initialize error values for each learning step
# e_lman_ra = np.zeros(n_learn); 
e_lman_sound = np.zeros(n_learn);
e_lman_sound2 = np.zeros(n_learn);
#e_weight = np.zeros(n_learn);
e_potential = np.zeros(n_learn);
e_hvc_sound = np.zeros(n_learn);

#def causal_inverse():

 #   return;


for i in xrange(0,n_learn):

    # Phase B -- Causal Inverse Learning.
    # "learning the inverse model from  babbling a subsong from HVC"
    # trying to reproduce the motor pattern it generated  
    # ie this is prospective learning.
    # - HVC is driving, 
    # - LMAN reflects auditory input,
    # - LMAN synapse is learning the inverse model.
    # - LMAN dendrite subject to shunting inhibtion, ie RA soma is
    #   equal to potential of HVC dendrite.
    # - HVC not learning

    #  causal_inverse();

    # HVC drives RA during imitation learning
    ra_soma = w_hvc.dot(hvc_soma) 

    # auditory activity produced by the sub song 
    aud_soma = delayperm(Q.dot(ra_soma),tau); 

    # motor activity predicted from the auditory activity (ie
    # potential on RA dendrite from LMAN:
    ra_pred = w_lman.dot(aud_soma); 

    # sound predicted from LMAN activity:
    sound_pred = S.dot(ra_pred); 

    # Difference between actual RA activity and predicted (via lman)
    # for learning rule:
    diff_ra_lman=(delayperm(ra_soma, tau) - ra_pred); 

    # Difference between LMAN-predicted song activty and actual HVC song:
    diff_sound_lman = (delayperm(S.dot(ra_soma),tau) - sound_pred);

    # Difference between LMAN-predicted song activity and actual tutor song:
    diff_sound_tut = (delayperm(song_sound_tut,tau) - sound_pred);

    # weight change: dw = (m_t-Delta - ra_pred_t) * a (postdictive
    # learning, need trace of prior motor activity) for weights from
    # auditory to motoric representation.
    dw_lman=diff_ra_lman.dot(aud_soma.T); 

    # apply weight change
    w_lman=w_lman+eta_lman*dw_lman; 

    # Mean squared error in motor estimation per time step and motor
    # neuron 
    # e_lman_ra[i]=(sum(diff_ra_lman*diff_ra_lman))/(T*n_ra);  #
    # motoric

    # MSE between LMAN prediction of sound and HVC song:
    e_lman_sound[i]=(sum(diff_sound_lman*diff_sound_lman))/(T*n_sound);  # in sound domain

    # MSE between LMAN prediction of sound and tutor song:
    e_lman_sound2[i]=(sum(diff_sound_tut*diff_sound_tut))/(T*n_sound);  
    
    # MSE between HVC produced sound and tutor song:
    diff_sound_hvc = S.dot(ra_soma) - song_sound_tut;
    e_hvc_sound[i] = sum(diff_sound_hvc*diff_sound_hvc)/ (T*n_sound);

     # This was Phase B -- inverse learning
  
    # we should perhaps clear here the local variables (or use a
    # function anyway)

    # Now Phase C -- weight copying.
    # - driving from LMAN from sound memory
    # - no actual acoustic feedback
    # - connections from HVC to RA are learning
    # - as a consequence e_lman_sound2 should go down above.
    # - LMAN dentrite potentail drives RA potential
    # - HVC dendrite shunting inihbition
    
    # song from memory:
    lman_soma = song_aud_tut;

    # mapping from LMAN to RA (
    ra_soma = w_lman.dot(song_aud_tut) # +
    # 1/10000*np.random.randn(n_ra, n_hvc); makes no differences

    # Presynaptic activitta at HVC dendrite
    pre_hvc = hvc_soma;

    # Potential in HVC dendrite:
    ra_dend_hvc =  w_hvc.dot(pre_hvc);  

    # potential different between soma and hvc dendrite:
    diff_hvc = ra_soma - ra_dend_hvc

    # normal learning:
    dw_hvc = (diff_hvc).dot(pre_hvc.T)
    
    w_hvc = w_hvc + eta_hvc * dw_hvc; # for some reason += does not
    # deliver the result I want (probaly a problem with reference vs
    # value)  

    #    e_weight[i] = sum((w_hvc - w_lman)*(w_hvc - w_lman)) /
    #    (T*n_lman*n_ra);
  
    # MSE between RA potentials at the soma (from LMAN) and at the
    # dendrite from HVC -- a measure that copy learning works:
    e_potential[i] = sum(diff_hvc*diff_hvc) / (T*n_ra);

    
#    sound_lman = S.dot(ra_soma); ## NoNo, this is the lman sound that
    ## would have been produced by lman if we allow the bird to sing
#    diff_sound_lman = song_sound_tut - sound_lman;
#    e_hvc_sound[i]=(sum(diff_sound_lman*diff_sound_lman))/(T*n_sound);  # in sound domain

    
#    e_weights[i]=(sum(d_weights*d_weights))/(T*n_hvc*n_ra); 

# print figures;

figure(1);
clf();
#title('Birdsong learning');
xlabel('Steps'); 
ylabel('SME');

#plot(e_lman_ra); 
#legend('SME between tutor motoric and "inverse" activity via LMAN');
#figure(2);

plot(e_lman_sound); 
legend('Phase B: Causal Inverse Learning: HVC-song vs predicated song from LMAN)');

hold("on");

plot(e_lman_sound2); 
legend('Phase B+C: Causal Inverse + Weight Copying: Tutor sound vs predicted (by LMAN)');

#hold("off");


#figure(2);
#clf();
#title('Birdsong learning');
#xlabel('Steps'); 
#ylabel('SME');

plot(e_hvc_sound); 
legend('Tutor song vs actual performed song (by HVC)');

#hold("on")

plot(e_potential); 
legend('Phase C: Activity copying from LMAN to HVC');

hold("off")

# This figure does not make sense anymore as there is not explicit
# identifcal representation of memory in HVC and LMAN   
# figure(2);
# plot(e_weight); 
# xlabel('Learning steps'); 
# ylabel('Error');
# title('Difference between HVC and LMAN weights');
# legend('Weights');
