#!/usr/bin/python

# \todo: - how is rand initialised -- it seems always the same sequence? Seed it?
# \todo: - Tutor-HVC Errors stay the same between HVC-driven and Tutor song since the introduction of
#           copy learning
# \todo: 4. im octave code von walter, das zufälige stammeln durch HVC input ersetzen.
# \todo: - try with a regularisation
# \todo: - look at the system as a dynamical system
# \todo: - why does it not work if we have a local dimenospnal bottleneck in n_sound?
# \todo: - send the code to Walter (eg the g
# \todo: - check with identiy for S and A
# \todo: - could we have a noise source that gets weaker and weaker as we learn? (but wasn't this source LMAN)
# \todo: - could audio input via elman be the noise source -- there
# was this experiment that songbirds geht confused when the song is
# played back with a delay? 
# \todo: phaseC0 without phase C doubles the final error roughly,
# while C without C0 multipley the error by an order of magnitude.
# - it appears we need matrices that are easy to invert for Q.
# - how ever pretraining is crucial.
# \todo: normqlsirung von allen vektoren und matrizen durch gehen, da
# erscheint wichtig für die numerische stabilität
# \todo: test other mapping than identity between aud and lman
# - lift the clipping where not necessary


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
# - 2015/11/16  ready to demonstrate
# - 2015/11/16  implementing Phase A


# dimension of song, ie number of "acoustic" degrees of freedom"
n_sound = 50; 

# number of motor neurons (RA)
n_ra = n_sound # / 2 #/ 2 # 3 for testing, or 7, n_ra=200; , currently 50 for testing,
# fine for 1,2 neurons, but from 3 onwords, it separates
# on train


# number of auditory neurons which receive the sound and convert into neural activity.
n_aud = n_sound;

# number of neurons in LMAN 
n_lman = n_aud;

# number of auditory memory neurons which memorize the tutor song (for one time step)
n_lman = n_sound; 

# duration of subsongs and of tutor song  [ms?]
T = 100; 

# Number of HVC clock neurons -- one time step is 10ms

n_hvc = T;

# Delay of auditory activity with respect to the generating motor activity (syrinx + auditory pathway) [ms?] -- isnt that rather in the area of 50ms-70ms?
tau = 7; # 7 orig, 0 for testing purposes, but this does not change
# anything, so timings are alright 

# learning steps for model 
n_learn = 600; 

eta_lman = 0.002 # 0.002; # learning rate for inverse model via lman -- that
# seems to be sufficient
eta_hvc = 1 # 0.01; # 0.02 # learning rate for weight copying orig = 0.001 = 1 ist one shot learning

epsi = 1; # 1/10; # Parameter to regularize the matrices to avoid near
# zero EV -- bad for inversion.

# syrinx; converts the RA motor activity signal m into a sound (here just matrix)
#S = (np.random.randn(n_sound,n_ra).clip(0.1,10) + epsi*np.eye(n_sound,n_ra)) / sqrt(n_ra);
#S = (np.random.randn(n_sound,n_ra).clip(0,10) + epsi*np.ones((n_sound,n_ra))) #/ sqrt(n_ra);
S = np.eye(n_sound, n_ra);

# auditory pathway; converts the song into an auditory signal for aud
# neurons (here just a matrix)
A = (np.random.randn(n_aud, n_sound).clip(0.01,1000))/ sqrt(n_sound); 
#A = (np.random.randn(n_aud, n_sound).clip(0,10) + epsi*np.ones((n_aud, n_sound))) / sqrt(n_sound); 
#A = np.eye(n_aud, n_sound);

# Total motor to auditory transformation. 
Q=A.dot(S); 

# acoustic representation of tutor song -- this what we start from.
song_sound_tut = np.random.randn(n_sound,T); 

# auditory representation of tutor song auditory -- generated by the 
# tutor song, or by rehearsing of the memory this is our imprint.
# PhaseA:
song_aud_tut = A.dot(song_sound_tut);

# weights between HVC and LMAN that hold the imprinted tutor song:
w_mem = np.zeros((n_lman, n_hvc));


# imprint tutor memory, song is the (n_sound,T) acoustic representation of the tutor song:
# assume that HVC has a one-hot encoding for each time step:
# It is actually futile to implement that, as the former direct once-forever calculation of song_aud_tut does mathematically the same (as long HVC is a one hot encoding)
# I guess one could also formulate this more generally as a product between (transposed) matrices:
def phaseA(song):
    
    global w_mem; # is this necessary?

    # auditory representation of the song, ie thus is it heard by the bird:
    song_aud_tut = A.dot(song); # this can probably go
    
    # imprint song into memory connections, however delay by tau ms as only the tauth HVC neuron shall evoke
    # the first part of the song:
    w_mem = delayperm(song_aud_tut, tau);

    #soma_lman = w_men.dot(hvc_soma);
    #soma_lman[t] = w_mem.dot(hvc_soma[t])
    #soma_lman[t0] = w_mem.dot(delta[t0, t])

    
phaseA(song_sound_tut);
    

# initial weight matrix converting the auditory representation from
# LMAN into RA motor activity (inverse model) 
# R = np.linalg.inv(Q)
w_lman =np.random.randn(n_ra,n_lman) / sqrt(n_lman); 
# w_lman = R

# same size as length of song: hvc just gives the rhythm, by having
# specific neurons bursting for a specific period of 10ms
w_hvc = np.random.randn(n_ra, n_hvc) # / sqrt(n_hvc);
hvc_soma = np.eye(n_hvc, T);

# Initialize error values for each learning step
e_lman_sound = np.zeros(n_learn);
e_potential = np.zeros(n_learn);
e_hvc_sound = np.zeros(n_learn);

n_pretraining =  1400;  # 500; #1000;

def phaseC0(): 

    global w_lman;

    # Phase C0 -- Causal Inverse Learning.
    # - random driving, 
    # - LMAN reflects auditory input,
    # - LMAN synapse is learning the inverse model.
    # - LMAN dendrite subject to shunting inhibtion, ie RA soma is
    #   equal to potential of HVC dendrite.
    # - HVC not learning.

    # Random drive on  RA during imitation learning
    ra_soma = w_hvc.dot(hvc_soma); # + 1/100 * np.random.randn(n_ra, T); 
    #    ra_soma = np.random.randn(n_ra,T);
    
    # auditory activity produced by the random activity:
    aud_soma = delayperm(Q.dot(ra_soma),tau); 

    lman_soma = aud_soma

    # motor activity predicted from the auditory activity (ie
    # potential on RA dendrite from LMAN:
    ra_pred = w_lman.dot(lman_soma); 

    # sound predicted from LMAN activity:
    sound_pred = S.dot(ra_pred); 

    # Difference between actual RA activity and predicted (via lman)
    # for learning rule:
    diff_ra_lman=(delayperm(ra_soma, tau) - ra_pred); 

    # weight change: dw = (m_t-Delta - ra_pred_t) * a (postdictive
    # learning, need trace of prior motor activity) for weights from
    # auditory to motoric representation.
    dw_lman=diff_ra_lman.dot(aud_soma.T).clip(-1,1); 

    # apply weight change
    w_lman=w_lman+eta_lman*dw_lman; 

for _ in xrange(0,n_pretraining):
    phaseC0();

def phaseC(): # causal inverse learning

    global w_lman

    # Phase C -- Causal Inverse Learning.
    # "learning the inverse model from  babbling a subsong from HVC"
    # trying to reproduce the motor pattern it generated  
    # ie this is prospective learning.
    # - HVC is driving, 
    # - LMAN reflects auditory input,
    # - LMAN synapse is learning the inverse model.
    # - LMAN dendrite subject to shunting inhibtion, ie RA soma is
    #   equal to potential of HVC dendrite.
    # - HVC not learning

    # HVC drives RA during imitation learning
    y = 0.0 # setting this to values different from zero makes it wrse
    ra_soma = (1-y) * w_hvc.dot(hvc_soma) # + y*lman_soma;

    # auditory activity produced by the sub song 
    aud_soma = delayperm(Q.dot(ra_soma),tau); 

    # lman soma is a convex mix of auditory input and imprinted tutor memory:
    x = 0.0
#    lman_soma = (1-x) * aud_soma  + x* delayperm(song_aud_tut, tau);
    lman_soma = (1-x) * aud_soma  + x * delayperm(w_mem.dot(hvc_soma), 0);

    # motor activity predicted from the auditory activity (ie
    # potential on RA dendrite from LMAN:
    ra_pred = w_lman.dot(lman_soma); 

    # sound predicted from LMAN activity:
    sound_pred = S.dot(ra_pred);  # \todo: take this away

    # Difference between actual RA activity and predicted (via lman)
    # for learning rule:
    diff_ra_lman=(delayperm(ra_soma, tau) - ra_pred); 

    # Difference between LMAN-predicted song activty and actual HVC song:
    diff_sound_lman = (delayperm(S.dot(ra_soma),tau) - sound_pred); # \todo: simplify this -- all is linear

    # weight change: dw = (m_t-Delta - ra_pred_t) * a (postdictive
    # learning, need trace of prior motor activity) for weights from
    # auditory to motoric representation.
    dw_lman=diff_ra_lman.dot(aud_soma.T).clip(-1,1); 

    # apply weight change
    w_lman=w_lman+eta_lman*dw_lman; 

    # MSE between LMAN prediction of sound and HVC song:
    e_lman_sound[i]=(sum(diff_sound_lman*diff_sound_lman))/(T*n_sound);  # in sound domain

    # print i


def phaseB():

    global w_hvc

    # Now Phase B -- weight copying.
    # - driving from LMAN from sound memory
    # - no actual acoustic feedback
    # - connections from HVC to RA are learning
    # - as a consequence e_lman_sound2 should go down above.
    # - LMAN dentrite potentail drives RA potential
    # - HVC dendrite shunting inihbition
    
    # song from memory -- here i need to delay:
#    lman_soma = song_aud_tut; 
    lman_soma = w_mem.dot(hvc_soma);


    # mapping from LMAN to RA (
    ra_soma = w_lman.dot(lman_soma) # +
    # 1/10000*np.random.randn(n_ra, n_hvc); makes no differences

    # Presynaptic activity at HVC dendrite
    pre_hvc = hvc_soma;

    # Potential in HVC dendrite:
    ra_dend_hvc =  w_hvc.dot(pre_hvc);  

    # potential difference between soma and hvc dendrite (predictive learning):
#    diff_hvc = ra_soma - delayperm(ra_dend_hvc, 0)
    diff_hvc = ra_soma - delayperm(ra_dend_hvc, tau)

    # (predictive) learning:
    #    dw_hvc = (diff_hvc.dot(pre_hvc.T)).clip(-1,1);
    dw_hvc = (diff_hvc.dot(delayperm(pre_hvc, tau).T)).clip(-1,1);

    w_hvc = w_hvc + eta_hvc * dw_hvc; # for some reason += does not
    # deliver the result I want (probaly a problem with reference vs
    # value)  

    #    e_weight[i] = sum((w_hvc - w_lman)*(w_hvc - w_lman)) /
    #    (T*n_lman*n_ra);
  
    # MSE between RA potentials at the soma (from LMAN) and at the
    # dendrite from HVC -- a measure that copy learning works:
    e_potential[i] = sum(diff_hvc*diff_hvc) / (T*n_ra);

def sing_HVC():

    # MSE between HVC produced sound and tutor song:

    ra_soma = w_hvc.dot(hvc_soma);
    song_sound_hvc = S.dot(ra_soma)
    diff_sound_hvc =  song_sound_hvc - song_sound_tut;
    e_hvc_sound[i] = sum(diff_sound_hvc*diff_sound_hvc)/ (T*n_sound);

    return song_sound_hvc;


for i in xrange(0,n_learn):

    phaseC(); # causal inverse on HVC-song

# for i in xrange(0,n_learn):

    phaseB(); # acitivity copying from LMAN to HVC

    sing_HVC();

    
# print figures:

figure(1);
clf();
title('Birdsong learning');
xlabel('Steps'); 
ylabel('SME');

plot(e_lman_sound); 
legend('Phase C: Causal Inverse Learning: HVC-song vs predicated song from LMAN)');

hold("on");

plot(e_hvc_sound); 
legend('Tutor song vs actual performed song (by HVC)');

plot(e_potential); 
legend('Phase B: Activity copying from LMAN to HVC');

hold("off")

hardcopy("imitation_learning.png");


# pick a random component of the sound:
i=floor(np.random.rand()*n_sound);

figure(2);
clf;
title("Sample Song Dimension");
xlabel('Time of Song'); 
ylabel('Activity');


plot(song_sound_tut[i])
legend("Tutor Song");

hold("on");

plot(sing_HVC()[i]);
legend("HVC Song");

hold("off");

