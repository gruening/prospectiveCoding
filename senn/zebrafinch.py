#!/usr/bin/python

# - is Doupe ein homeostatischer Effect?
# - where does the time shift come from? It has no functional explanation.

import numpy as np
from scitools.std import *


def delayperm(x,n):  # Circles the columns of matrix x to the right by n columns. 
    return np.roll(x, n, axis=1);


# Learning of the inverse and then the predictive inverse model for song generation. 
# Based on discussions with Richard and Surya, Jan 14, 2011. Bugs etc attributed to Walter.
# 
# Shared with Andre Gruning, 2015/10/19.
# - updating comments
# - rewrite in pyhton.

n_ra   = 200; # 3 for testing
n_song = n_ra; # dimension of song, ie number of "aciustic"/artilatory degrees of freedom"
n_hvc  = n_ra; # number of neurons per time step in hvc
n_lman = n_hvc; # number of lman neurons memoriting per time step.

T   = 100; # duration of subsongs and of tutor song  [ms?] or [5ms?]?
tau = 10;  # 10 Delay of auditory activity with respect to the
# generating motor activity (syrinx + auditory pathway) [ms?] -- isnt
# that rather in the area of 50ms? 

oneStep=1; # Propagation delay of one neuronal connection [ms?]

nlearn = 300; # learning steps for inverse model (consisting of one
# alternatinve hvc and lman drive -- could change this later to random
# stuff. 


eta_hvc=0.001; # learning rate for inverse model
eta_lman=0.001; # learning rate for predictive inverse model

# syrinx; converts the motor activity signal m into a sound (here just matrix)
S = np.random.randn(n_song,n_ra) / sqrt(nm);

# auditory pathway; converts the song into an auditory signal (here
# just a matrix) -- lman here correct.
A = np.random.randn(n_lman,n_song) / sqrt(na); 

# total operator for mapping from RA to Aud:
Q = A.dot(S); # Total motor to auditory transformation, eg atut=delayperm(Q*mtut,tau)

# Tutor song in "audio" representation:
# This is what we want to reproduce
tutor_song = np.random.randn(n_song,T); 

# sensory representation of tutor song is imprinted into both HVC and LMAN
tutor_hvc = S.dot(tutor_song);
tutor_lman = S.dot(tutor_song);

# initial weight matrix from hvc to ra:
w_hva = np.random.randn(n_ra,n_hvc)/sqrt(na); 

# initial weights froms lman to ra
w_lman = np.random.randn(n_ra,n_lman)/sqrt(na); 

# Initialize error value for each learning step (do we need to do this
# for pyhton?

e_inv = np.zeros(nlearn_inv); 

for i in xrange(0,n_learn):
    
    # 1.phase #########################
    # - hvc driving, 
    # - motoric output (song) is actively produced.
    # - lman listening and updating, 
    # - RA dendritic compartment driving by lman shuting inhibition

    # Ra motor activity:
    ra = delayperm(w_hvc.dot(tutor_hvc), oneStep);

    # resulting auditory activity at lman:
    lman = delayperm(Q.dot(m),tau); 

    # motor activity predicted from the auditory activity (ie
    # potential in the dendritic compartment with shunting inhibition:
    ra_predicted_lman = delayperm(w_lman.dot(lman),oneStep); 
  
    # postdictive learning (with conductive synapses??)

    dm=(delayperm(m,tau+oneStep)-minv); 
  
  # weight change: dw = (m_t-Delta - minv_t) * a (postdictive
  # learning, need trace of prior motor activity) for weights from
  # auditory to motoric representation.
  dwinv=dm.dot(delayperm(a,oneStep).T); 

  # apply weight change
  winv=winv+eta_inv*dwinv; 

  # Mean squared error in motor estimation per time step and motor
  # neuron 
  # do not see what the second sum is for? And why divide by T?
  e_inv[i]=(sum(dm*dm))/(T*nm); # change from dot product? Probably not.
  
# print figure;
figure(1);
plot(e_inv); 
xlabel('Learning steps'); 
ylabel('Inverse error');
title('Inverse error');
legend('w_{inv}');

# 2. Phase
# learning the predictive inverse model during sleep from internally
# rehearsing the tutor song (phase 2) -- ie trying weights from memory
# to RA.

# initial weight matrix converting the auditory activity a into a motor activity mS (inverse model)
wpred=np.random.randn(nm,nam)/sqrt(nam); 

# Initialize error value for each learning step
e_pred=np.zeros(nlearn_pred); 

# motor activity produced by the inverse connections from the
# rehearsed memory through  the auditory representation (thats where
# the delay comes from?)
minv=delayperm(winv.dot(atut),oneStep); # dot product?

# eligibility trace of length tau+oneStep for synapses from the memory
# to the motor area 
amtut_del=delayperm(amtut,tau+2*oneStep); 


for i in xrange(0, nlearn_pred):

  # delayed silent input from the auditory memory 
  mpred=wpred.dot(amtut_del); 

  # estimation error; 
  dm=(minv-mpred); 

  # learning rule for inverse weights
  # presynaptic eligibility trace of length tau (why do we have a
  # reference here to an elegibility trace?
  # THis is for the connections from the memory direct to RA
  # m_inv - m_pred)*am -- normal instantanous learning.
  dwpred=dm.dot(amtut_del.T); # dot product? Transponed?

  # update inverse weights
  wpred=wpred+eta_pred*dwpred; # XXX can replace with +=?
  
  # Mean squared error in motor estimation per time step and motor
  # neuron 
  e_pred[i]=sum(dm*dm)/(T*nm); 

# make the figure:
figure(2);
plot(e_pred); 
xlabel('Learning steps'); 
ylabel('Predictive error');
legend('$w_{pred}$');


# 3. Phase
# %%%% to test Adult song production

# Auditory activity generated by the inverse model, but not delayed by
# the song production + audit. pathway:
atinv=delayperm(Q.dot(winv).dot(atut),0); # does this keep the correct
# order of things? - ja, denn matrix multiplications is associative.

# Auditory activity in the memory area generated by the predictive inverse model
amtpred=delayperm(Q.dot(wpred).dot(amtut),tau); # why do I need tau here, but
# not above?

# Comparison between memorized and generated auditory activity

# Select an auditory neuron
i=floor(np.random.rand(1)*na)[0]; 

figure(3);
clf;
#plot([atut(i,:);atinv(i,:)]) # this could need a change as well from

plot(atut[i])
hold("on");
plot(atinv[i,:])
hold("off");
legend("atut", "atinv");
 # this could need a change as well from
# semicolon to coma.
xlabel('time steps'); 
title("Activity of one auditory neuron recieved if song generarted via \
the inverse model from auditory respresentation"); # ie a test how good an inverse of Q
# w_inv has become.
#r = corr(atut(:),atinv(:));
#r = np.corrcoef(atut[:],atinv[:])[0,1];
#fprintf('Corr. coeff. inv. model: %.4f\n',r);

figure(4);
clf;
plot(amtut[i])
hold("on");
plot(amtpred[i]);
xlabel('Time Steps'); 
title('Activity of one memory neuron generated by the predictive inverse model');
legend("amtut", "amtpred");
hold("off");


#r=corrcoef(amtut[:],amtpred[:]);
#fprintf('Corr. coeff. predictive inv. model: %.4f\n',r);
# ie a good an inverse of Q the preditictve pathway has become.

# Playback test
#%mplayback=wpred*delayperm(A*song_tut,oneStep); % Takes auditory response in memory area during playback, an#d produces the motor sequence out of that
#%msong=wpred*delayperm(amtut,oneStep); % The song generated motor sequence is trivially the same here becaus#e we assumed perfect memory, i.e. amtut=A*song
#%figure();clf;plot([mplayback(i,:); mplayback(i,:)]')
#%xlabel('time steps'); title('Mirroring is trivial because M has same drive during singing and playback');


