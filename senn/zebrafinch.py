#!/usr/bin/python

# The copying part works, but the inverse part does not work, 
# - however that worked in inverse thingy -- so I should have a look there.
# - is Doupe ein homeostatischer Effect?
# - where does the time shift come from? It has no functional explanation.
# - we can simplify code by taking out tau shifting of matrices.
# - test the imitation learning separately and/or cc the code from predictiveLearning.py
# - stutze diesen Code so dass er zum inverse learning identisch ist.
# - und umgekehrt: machen predicticInverse lernen nur von einem determinstischen Signal.
# - later ausgabe als summe von frquenzen mit amplitude, fourier zerlegung 
# - gedaechtnis der Tutor song ist in den Gewichten von HVC to LMAN imprinted, dh wahl von eimen gewichtsstaz# - gedaechnsuts des Mature song ist in den Gewichten von HVC to RA
# - 30ms (dh dauer des bursts) ontime der RA neurone für phi(t). (10ms von Leonard)
# - wie lang is die Zeit zwischen 2 bursts.
# - print file von Senn Hanloser und das mit Drosopila
# - within-burst-spike count gives amplitude of each frequency band.
# - zerlegung in 4 positive componenten.
# - LMAN lauft \delta\tau hinterher.
# - HVC treibt (je 10ms ein burst)
# - frage leslie ob erst fourier oder erst log?
# - positive matrix decomposition of Qmot = aud, to ensure all the phis are positive.
# - cast code into functions
# - how can I include 


# for predictivelearning with mtut as the driver of HVC in the first phase, inverse error goes to 

import numpy as np
from scitools.std import *


def delayperm(x,n):  # Circles the columns of matrix x to the right by n columns. 
    return np.roll(x, n, axis=1);


# Learning of the inverse and then the predictive inverse model for song generation. 
# Based on discussions with Richard and Surya, Jan 14, 2011. Bugs etc attributed to Walter.
# 
# Shared with Andre Gruning, 2015/10/19.
# - update comments
# - rewrite in pyhton.
# - extended to different forms of the zebra finch model

n_ra   = 100 #7 ; # 3 for testing, 200 for production run
n_song = n_ra; # dimension of song, ie number of "aciustic"/artilatory degrees of freedom"
n_hvc  = n_ra; # number of neurons per time step in hvc
n_lman = n_hvc; # number of lman neurons memoriting per time step.

mem_strength = 0.1 # when mixing lman memory and auditory input in a convex combination, this give the mix-in weight of lman memory


T   = 100; # duration of subsongs and of tutor song  [ms?] or [5ms?]? 100 for real runnig, 5 for testing
tau = 10;  # 10 Delay of auditory activity with respect to the
# generating motor activity (syrinx + auditory pathway) [ms?] -- isnt
# that rather in the area of 50ms? 

#oneStep = 1; # Propagation delay of one neuronal connection [ms?]

n_learn = 3000; # learning steps for inverse model (consisting of one
# alternatinve hvc and lman drive -- could change this later to random
# stuff. 


eta_hvc = 0.001 # 0.001; # learning rate for inverse model, started with 0.001
eta_lman = 0.002; # learning rate for predictive inverse model with 0.002

# syrinx; converts the motor activity signal m into a sound (here just matrix)
S = np.random.randn(n_song,n_ra) / sqrt(n_song);

# auditory pathway; converts the song into an auditory signal (here
# just a matrix) -- lman here correct.
A = np.random.randn(n_lman,n_song) / sqrt(n_song); 

# total operator for mapping from RA to Aud:
Q = A.dot(S); # Total motor to auditory transformation, 

# Tutor song in "audio" representation:
# This is what we want to reproduce
tutor_song = np.random.randn(n_song,T); 

# Phase A -- not modelled: sensory representation of tutor song is imprinted into both HVC and LMAN
tutor_hvc = A.dot(tutor_song);
tutor_lman = tutor_hvc;

# initial weight matrix from hvc to ra: -- how to be normalised? Divide by n_ra.
w_hvc = np.random.randn(n_ra,n_hvc)/sqrt(n_hvc); 

# initial weights froms lman to ra
w_lman = np.random.randn(n_ra,n_lman)/sqrt(n_lman); 

# Initialize error value for each learning step:
e_hvc_song = np.zeros(n_learn); 
e_lman_song = np.zeros(n_learn); 

e_weights = np.zeros(n_learn);

for i in xrange(0,n_learn):
    
    # Phase B (Weight copying from lman to hvc)
    # - lman->ra driving, hvc->ra learning.
    # - auditory feedback blocked. 
    # - no plasticity on lman->ra synapse
    # - input from hvc to ra shunted off
    # - that is potential ra = w.lman
    # - that is 
    # - Variant 1: lman and hvc are in sync:

    # potential at soma:
    ra_soma = w_lman.dot(tutor_lman)
    
    # song that would have been produced if we were not silent:
    # lman_song = S.dot(ra_soma);  

    # potential in dendrite from hvc:
    ra_dend_hvc = w_hvc.dot(tutor_hvc)

    # presynaptic potential from hvc onto ra:
    pre_hvc = tutor_hvc;

    dw_hvc = (ra_soma - ra_dend_hvc).dot(pre_hvc.T)

    w_hvc = w_hvc + eta_hvc * dw_hvc; # for some reason += does not
    # deliver the result I want (probaly a problem with reference vs
    # value)  
    
    # Phase C: Learning the causal inverse model
    # - hvc driving ra
    # - lman input to ra, shunted off, but synapse is learning postdictively.
    
    # potential at ra soma from hvc:
    ra_soma = w_hvc.dot(tutor_hvc)# + 1/100*np.rand.randn(na_ra, T));

    # Song produced:
    hvc_song = S.dot(ra_soma);  

    # auditory input at auditory neurons, but delayed by tau:
    aud = delayperm(Q.dot(ra_soma),tau); 

    # potential on lman dendrite from aud neurons
    # assume mapping from aud to lman is identity.
    lman_dend_aud = aud #+ 1/100*np.random.randn(n_song,T) / sqrt(n_song); 

    # potential of lman dentrite from lman delayed memory (triggered by hvc): 
    lman_dend_mem = w_lman.dot(delayperm(tutor_lman, tau)); 

    # potential at lman is (proportional to) convex mixture of own song heard and delayed lman memory 
    lman_soma = (1 - mem_strength) * lman_dend_aud + mem_strength * lman_dend_mem

    ra_dend_lman = w_lman.dot(lman_soma);

    lman_song = S.dot(ra_dend_lman) # song as if re-produced from
    # acoustic input

    # postdictive learning of lman to ra connections (ie sensory-motor association)
    # - requires synapse to have a trace of RA activity tau steps back:
    
    dw_lman = ( delayperm( ra_soma, tau) - ra_dend_lman ).dot(lman_soma.T)
    w_lman  = w_lman + eta_lman * dw_lman.clip(-1,1); # regularisation, Q may have EVs near 0.

    # calculate deviation between produced song and tutor song:
    d_hvc_song = delayperm(hvc_song,0) - lman_song; # tutor_song
    e_hvc_song[i]=(sum(d_hvc_song*d_hvc_song))/(T*n_song); 

    d_lman_song = delayperm(lman_song,-tau) - tutor_song;
    e_lman_song[i]=(sum(d_lman_song*d_lman_song))/(T*n_song); 


    # calculate difference between weight to RA from LMAN and from HVC
    d_weights = w_hvc - w_lman;
    e_weights[i]=(sum(d_weights*d_weights))/(T*n_hvc*n_ra); 

  
# print figure;
figure(1);
clf;
plot(e_hvc_song); 
xlabel('Learning steps'); 
ylabel('MSE across all dimensions of song and the duration of the song');
title('Error between Tutor Song from HVC and Produced Song');
legend('$e_{hvc-song}$');


figure(4);
clf;
plot(e_lman_song); 
xlabel('Learning steps'); 
ylabel('MSE across all dimensions of song and the duration of the song');
title('Error between Tutor Song from LMAN and Produced Song');
legend('$e_{lman-song}$');



figure(2);
clf;
plot(e_weights); 
xlabel('Learning steps'); 
ylabel('MSE across different of weighrts');
title('Weight');
legend('$weights$');


# 3. Phase
# %%%% to test Adult song production

# Select a song domesion neuron
i=floor(np.random.rand(1)*n_song)[0]; 

figure(3);
clf;
plot(song[i])
legend("Song")
hold("on");
plot(tutor_song[i])
legend("Tutor Song");
hold("off");


xlabel("Time steps"); 
title("Activity of in one song dimension");

r=corrcoef(song[:],tutor_song[:]);
print('Corr. coeff. predictive inv. model: %.4f\n',r);


