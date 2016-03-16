#!/usr/bin/python

# \todo experiment that songbirds get confused when the song is
#    played back with a delay? 
# \todo: normalsirung von allen vektoren und matrizen durch gehen
# \todo: PCA (or Seungs nonnegative decomposition) of sound samples to get the image from RA to the sound via a low dim bottle neck
# \todo wavelet encodeing: RA burst is on-set of wavelet.16 wavelets scheinen zu reichen
# \todo print nature 2006 and nat neursci Lewicksi "Efficient auditory coding"
# \todo teile neuron wavelet 6 832Hz) auf in 6.1 and 6.2 etc to ensure sparse coding
# \todo look up positive matrix decomposition (for sparse coding in RA)
# \todo STSRT HERE IN THE NEW YEAR
#   - try to work out what is going wrong with audio.py as it only can transform sinus wave 
#       - it seems to lower the frequencies
#       - check the transposes and dimension of the matrices -- are they alligned?
#       - check also the norm of all.
#   - try both wavelets and stft
#   - try to work with formants
# \todo "physical model" of song/speech syntheseby wave-let pca to
#    generate a physical model and subspace of speech 

import numpy as np
from scitools.std import *
import scipy.fftpack as fftp #.fftpack as fft # this is not the numpy implementation! Sth better available?
import ewave as wav
import ou
import audio

show_on = False

rate = 10000

# acoustic representation of tutor song -- this is what we start from.
#song_sound_tut = audio.load_wave("caesar_3s.wav");
song_sound_tut = audio.load_wave("song_f3.III.wav");

# duration of subsongs and of tutor song  [steps of 10ms] -- so we are
# modelling 1 sec here -- realistic length of song?
T = len(song_sound_tut[0]) 

# from audio import *
def delayperm(x,n):  # Circles the columns of matrix x to the right by n columns. 
    return np.roll(x, n, axis=1);

# temporary storage for 10ms snipped of song
snippet = np.copy([0.0 for _ in xrange(0,100)])

import birdsong

def syrinx(beta, alpha):
    birdsong.finch(beta, alpha, snippet)
    return fftp.dct(snippet, type=2, norm='ortho')




# lowest and highest value for activitiy clipping:

least = 0; # -1000
threshold = 0.0;
ra_thresh = 0.0;
sat = 8000

# take the "membrane potential" x and applies a threadhold linear function to it:
def activation(x, thresh = threshold): 
#    return maximum(x - thresh, 0); // I cant activate this as long as I ask RA to produce the sound direct => 4fold decomposition into positive values only?
    return x
#    return x.clip(least, sat)


# Learning of the inverse and then the predictive inverse model for song generation. 
# Based on discussions with Richard and Surya, Jan 14, 2011. Bugs etc
# attributed to Walter (and now also Andre) 
# 
# Shared with Andre Gruning, 2015/10/19. Changes:
# - Oct 2015: updating comments
# - Oct 2015:  translate to python
# - 2015/11/02: replace random drive with fixed drive.
# - 2015/11/02: rename variable to better fit our current terminology
# - 2015/11/02: change from random drive of RA during inversion
#               learning to drive with one fixed pattern.
# - 2015/11/02: add error between actual sound and predicted sound via
#               LMAN
# - 2015/11/02  switch driving from direct injection into RA to
#               injection via HVC -- leads to same error of
#               causal-inverse learning.
# - 2015/11/02  start weight copying in phase 2
# - 2015/11/02  HVC branch: implement HVC as a clock.
# - 2015/11/10  rectify plots and legends
# - 2015/11/16  ready to demonstrateo
# - 2015/11/16  implement Phase A
# - 2015/12/07  implement threshold
# - 2015/12/08  implement large and sparse RA
# - 2016/03/15  in the process of replacing S with a modelled syrinx


# dimension of song, ie number of "acoustic" degrees of freedom
n_sound = 100 # 50 # 100 # 50 # 200; # was 20, or 50

# number of motor neurons (RA)
n_ra = n_sound #* 5;

# degrees of freedom of the vocal tract:
n_syrinx = 2

# number of auditory neurons which receive the sound and convert into neural activity.
n_aud = n_sound;

# number of neurons in LMAN 
n_lman = n_aud;

# Number of HVC clock neurons -- one time step is 10ms
n_hvc = T;

# Delay of auditory activity with respect to the generating motor
# activity (syrinx + auditory pathway) [70ms]  
tau = 7; 

# learning steps for model 
n_learn = 1000 #2 # 4000 #10 # 8000 # 150 # 1000 # 8000 # 4000 # 1200 # 2000 # 600 

eta_lman = 0.002 # 0.05 # learning rate for inverse model via lman
# -- that # seems to be sufficient. higfher learning rates in the
# order of 0.005,
# lead to oscilations, 0.003 to fluctations , 0.0001 is too small.

# learning rate for weight copying 
eta_hvc = 0.001 # 0.01 # one shot learning does not explore the inverse map sufficiently.

# Parameter to regularize the matrices 
# epsi = 1/10; 


# weights between RA and the syrinx degrees of freedom (the syrinx is the bottle neck re degrees of freedom) 

#w_mot = np.eye(n_syrinx, n_ra) #/ sqrt(sqrt(n_ra));
w_mot = np.random.randn(n_syrinx, n_ra) / sqrt(n_ra) / 100;

# syrinx; converts the motor activity signal to the syrinx m into a sound (in frequency domain)
# S = (np.random.randn(n_sound,n_mot)) / sqrt(n_mot); #+ epsi*np.eye(n_sound,n_mot)) 
# S = np.eye(n_sound, n_syrinx);

# auditory pathway; converts the song into an auditory (frequence domain) signal for aud
# neurons (here just a matrix)
A = np.random.randn(n_aud, n_sound) / sqrt(n_sound); 
# + epsi*np.ones((n_aud, n_sound))) / sqrt(n_sound); 

# Total RA to auditory transformation. 
#M=A.dot(S).dot(w_mot); 

# mot activation of the tutor (necessary to generate a singable song
# song_mot_tut = activation(np.random.randn(n_mot,T));

# song_mot_tut = np.reshape(ou.ou(n_mot*T), (n_mot,T));
# song_mot_tut = np.random.randn(n_mot,T);
# song_sound_tut = np.random.randn(n_sound,T); 


# weights between HVC and LMAN that hold the imprinted tutor song:
w_mem = np.zeros((n_lman, n_hvc));

# imprint tutor memory, song is the (n_sound,T) acoustic representation of the tutor song:
# assume that HVC has a one-hot encoding for each time step:
# It is actually futile to implement that, as the former direct
# once-forever calculation of song_aud_tut does mathematically the
# same (as long HVC is a one hot encoding) 
# I guess one could also formulate this more generally as a product between (transposed) matrices:
def phaseA(song):
    
    global w_mem;

    # auditory representation of the song, ie as perceived by the bird:
    song_aud_tut = A.dot(song); 
    
    # imprint song into memory connections, however delayed by tau steps as
    # only the tauth HVC neuron shall evoke the first part of the song:
    w_mem = delayperm(song_aud_tut, tau);

    
phaseA(song_sound_tut);
    

# initial weight matrix converting the auditory representation from
# LMAN into RA motor activity (inverse model) 
# R = np.linalg.inv(M)
# w_lman =np.random.randn(n_ra,n_lman) / sqrt(n_lman); 
w_lman =np.random.randn(n_ra,n_lman) / sqrt(n_lman); # higher weights to work with sparse representations?

# w_lman = R  # injection of inverse matrix

# hvc just gives the rhythm, by having a neurons burst for 10ms at
# each time step:
hvc_soma = np.eye(n_hvc, T);

# w_hvc = np.random.uniform(-1,1,(n_ra, n_hvc)) # / sqrt(n_hvc); # no
w_hvc = np.random.randn(n_ra, n_hvc) # / sqrt(n_hvc); # random start of hvc weights.

# Initialize error values for each learning step
e_lman_sound = np.zeros(n_learn);
e_potential = np.zeros(n_learn);
e_hvc_sound = np.zeros(n_learn);

n_pretraining =  0 # 2000;  # 0 # 1400;  # 500; #1000;

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
    ra_soma = activation(w_hvc.dot(hvc_soma), ra_thresh);

    syrinx_soma = w_mot.dot(ra_soma)
    # auditory activity produced by the random activity:
    syrinx_out = map(syrinx, syrinx_soma[0], syrinx_soma[1])
    aud_soma = activation(delayperm(A.dot(syrinx_out),tau)) 
    lman_soma = aud_soma

    # motor activity predicted from the auditory activity (ie
    # potential on RA dendrite from LMAN:
    ra_pred = activation(w_lman.dot(lman_soma))

    # syrinx predicted from LMAN activity:
    syrinx_pred = w_mot.dot(ra_pred); 
    # sound fft from syring
    sound_pred = map(syrinx, syrinx_pred[0], syrinx_pred[1])

    # Difference between actual RA activity and predicted (via lman)
    # for learning rule:
    diff_ra_lman=(delayperm(ra_soma, tau) - ra_pred); 

    # weight change, postdictive learning, need trace of prior motor activity:
    dw_lman=diff_ra_lman.dot(aud_soma.T).clip(-10,10); # clip required for higher learning rates 

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
    lamb = 1.0 # setting this to values different from zero makes it
    # worse (due to the time lag between HVC and LMAN?) -- test again
    #    ra_soma = lamb * w_hvc.dot(hvc_soma).clip(least,sat) # + (1-lamb)*w_lman.doc(lman_soma);
    ra_soma = activation(lamb * w_hvc.dot(hvc_soma), ra_thresh) 

    # auditory activity produced by the sub song 



    syrinx_soma = w_mot.dot(ra_soma)
    syrinx_out = np.array(map(syrinx, syrinx_soma[0], syrinx_soma[1]))

    aud_soma = activation(delayperm(A.dot(syrinx_out.T),tau)) #.clip(least,sat); 

    # lman soma is a convex mix of auditory input and imprinted tutor memory:
    # mu = fraction of memory in the mixture.
    #    mu = 0.95  # 0.5 # 0.99 --
    mu = 0.8
    lman_soma = activation((1-mu) * aud_soma  + mu * w_mem.dot(hvc_soma))

    # motor activity predicted from the auditory activity (ie
    # potential on RA dendrite from LMAN:
    ra_pred = activation(w_lman.dot(lman_soma)) #.clip(least,sat); 

    # sound predicted from LMAN activity:
    syrinx_pred = w_mot.dot(ra_pred)
    sound_pred = map(syrinx, syrinx_pred[0], syrinx_pred[1]) 

    # Difference between actual RA activity and predicted (via lman)
    # for learning rule:
    diff_ra_lman=(delayperm(ra_soma, tau) - ra_pred); 

    # Difference between LMAN-predicted song activty and actual HVC
    # song:

    syrinx_pred = w_mot.dot(ra_soma)
    syrinx_out = map(syrinx, syrinx_pred[0], syrinx_pred[1])
    diff_sound_lman = (delayperm(syrinx_out,tau) - sound_pred); 
    # \todo: simplify the above -- all is linear  

    # weight change: dw = (m_t-Delta - ra_pred_t) * a (postdictive
    # learning, need trace of prior motor activity) for weights from
    # auditory to motoric representation.
    dw_lman=diff_ra_lman.dot(aud_soma.T).clip(-10,10); # clipping
    # seems necessay here for numeric stability

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
    
    # (delayed) song from memory:
    lman_soma = activation(w_mem.dot(hvc_soma)) 

    # mapping from LMAN to RA (
    ra_soma = activation(w_lman.dot(lman_soma), ra_thresh) 
    # 1/10000*np.random.randn(n_ra, n_hvc); makes no difference

    # Presynaptic activity at HVC dendrite
    pre_hvc = hvc_soma;

    # Potential in HVC dendrite:
    ra_dend_hvc =  activation(w_hvc.dot(pre_hvc)) 

    # potential difference between soma and hvc dendrite (predictive learning):
#    diff_hvc = ra_soma - delayperm(ra_dend_hvc, 0)
    diff_hvc = ra_soma - delayperm(ra_dend_hvc, tau)

    # predictive learning:
    dw_hvc = (diff_hvc.dot(delayperm(pre_hvc, tau).T)) # .clip(-1,1);

    w_hvc = w_hvc + eta_hvc * dw_hvc; # for some reason += does not
    # deliver the result I want (probaly a problem with reference vs
    # value)  

    #    e_weight[i] = sum((w_hvc - w_lman)*(w_hvc - w_lman)) /
    #    (T*n_lman*n_ra);
  
    # MSE between RA potentials at the soma (from LMAN) and at the
    # dendrite from HVC -- a measure that copy learning works:
    e_potential[i] = sum(diff_hvc*diff_hvc) / (T*n_ra);


ra_soma = np.zeros(n_ra);

def sing_HVC():

    global ra_soma;

    # MSE between HVC produced sound and tutor song:

    ra_soma = activation(w_hvc.dot(hvc_soma), ra_thresh) 

    syrinx_pred = w_mot.dot(ra_soma)
    song_sound_hvc = np.array(map(syrinx, syrinx_pred[0], syrinx_pred[1])).T
    # diff_sound_hvc =  song_sound_hvc - song_sound_tut
    diff_sound_hvc =  song_sound_tut
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

plot(e_lman_sound, show=show_on); 
legend('Phase C: Causal Inverse Learning: HVC-song vs predicated song from LMAN)');

hold("on");

plot(e_hvc_sound, show=show_on); 
legend('Tutor song vs actual performed song (by HVC)');

plot(e_potential, show=show_on); 
legend('Phase B: Activity copying from LMAN to HVC');

hold("off")

hardcopy("imitation_learning.png");


# pick a random fft component of the sound:
song = sing_HVC()




i=floor(np.random.rand()*n_sound);

figure(2);
clf;
title("Sample Song Dimension");
xlabel('Time of Song'); 
ylabel('Activity');


plot(song_sound_tut[i], show=show_on)
legend("Tutor Song");

hold("on");
plot(song[i], show=show_on);
legend("HVC Song");
hold("off");
hardcopy("song.png")


final_song = audio.save_wave("song.wav", song);





i=floor(np.random.rand()*n_ra);


figure(3);
clf;
title("Sample RA Dimension");
xlabel('Time'); 
ylabel('Activity');


plot(ra_soma[i], show = show_on)
legend("RA activity");

hardcopy("RA_activity.png")
