#!/usr/bin/python

import numpy as np
from scitools.std import *

def delayperm(x,n):  # Circles the columns of matrix x to the right by n columns. 
    return np.roll(x, n, axis=1);


# Learning of the inverse and then the predictive inverse model for song generation. 
# Based on discussions with Richard and Surya, Jan 14, 2011. Bugs etc attributed to Walter.
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

# number of motor neurons (RA)
n_ra = 50; # 3 for testing, or 7, n_ra=200; 

# dimension of song, ie number of "acoustic" degrees of freedom"
n_sound = n_ra; 

# number of auditory neurons which receive the sound and convert into neural activity.
n_aud = n_ra; 

# number of auditory memory neurons which memorize the tutor song (for one time step)
n_mem = n_ra; 

# duration of subsongs and of tutor song  [ms?]
T = 100; 

# Delay of auditory activity with respect to the generating motor activity (syrinx + auditory pathway) [ms?] -- isnt that rather in the area of 50ms-70ms?
tau = 10; 

# learning steps for model 
n_learn=300; 

# 100 learning steps for predicitve inverse model (ie to learn direct
# connections from memory to motor neurons TO BE DELETED
nlearn_pred=1; 

eta_lman=0.002; # learning rate for inverse model via lman.

eta_pred=0.001; # learning rate for predictive inverse model TO BE DELETED.

epsi = 0; # Parameter to regularize the matrices -- not needed?

# syrinx; converts the RA motor activity signal m into a sound (here just matrix)
S = (np.random.randn(n_sound,n_ra) + epsi*np.eye(n_ra)) / sqrt(n_ra);

# auditory pathway; converts the song into an auditory signal for aud
# neurons (here just a matrix)
A = (np.random.randn(n_aud,n_sound) + epsi*np.eye(n_ra)) / sqrt(n_aud); 

# Total motor to auditory transformation, eg song_aud_tut=delayperm(Q*song_ra_tut,tau)
Q=A.dot(S); 

# motor activity to generate the tutor song, ie each time step is vector of n_ra-activities
# This is what we want to learn!
song_ra_tut=np.random.randn(n_ra,T); 

# acoustic representation of tutor song
song_sound_tut = S.dot(song_ra_tut);


# auditory representation of tutor song auditory -- generated by the 
# tutor song, or by rehearsing of the memory this is our imprint.
song_aud_tut = A.dot(song_sound_tut);



# Phase B:
# learning the inverse model from  babbling a subsong and
# trying to reproduce the motor pattern it generated (phase 1)  
# ie this is prospective learning.

# initial weight matrix converting the auditory representation from
# LMAN into RA motor activity (inverse model) 
w_lman = np.random.randn(n_ra,n_aud)/sqrt(n_aud); 

# Initialize error values for each learning step
e_lman_ra = np.zeros(n_learn); 
e_lman_sound = np.zeros(n_learn);
e_lman_sound2 = np.zeros(n_learn);


# fixed activity drives RA during imitation learning: 
ra_soma=song_ra_tut;

for i in xrange(0,n_learn):

  # auditory activity produced by the subsong 
  aud_soma=delayperm(Q.dot(ra_soma),tau); 

  # motor activity predicted from the auditory activity
  ra_pred=w_lman.dot(aud_soma); 
  
  # Difference between actual RA activity and predicted (via lman):
  diff_ra_lman=(delayperm(ra_soma,tau)-ra_pred); 

  sound_pred = S.dot(ra_pred); # from lman activity.

  diff_sound_lman = (delayperm(S.dot(ra_soma),tau) - sound_pred);
  diff_sound_tut = (delayperm(song_sound_tut,tau) - sound_pred);



  # weight change: dw = (m_t-Delta - ra_pred_t) * a (postdictive
  # learning, need trace of prior motor activity) for weights from
  # auditory to motoric representation.
  dw_lman=diff_ra_lman.dot(aud_soma.T); 

  # apply weight change
  w_lman=w_lman+eta_lman*dw_lman; 

  # Mean squared error in motor estimation per time step and motor
  # neuron 
  e_lman_ra[i]=(sum(diff_ra_lman*diff_ra_lman))/(T*n_ra);  # motoric
  e_lman_sound[i]=(sum(diff_sound_lman*diff_sound_lman))/(T*n_sound);  # in sound domain
  # in sound domain # in sound domain between lman predicted and tutor sound
  e_lman_sound2[i]=(sum(diff_sound_tut*diff_sound_tut))/(T*n_sound);  

  # This was Phase B -- inverse learning.
  


  
# print figure;

figure(1);
plot(e_lman_ra); 
xlabel('Learning steps'); 
ylabel('Error');
title('Inverse error of forced motoric activity at RA vs predicted motoric activity via LMAN$');
legend('SME between tutor motoric and "inverse" activitey via LMAN');

figure(6);
plot(e_lman_sound); 
xlabel('Learning steps'); 
ylabel('Error');
title('Inverse error between sound produced via actual RA vs predicitdc motoric activity via LMAN$');
legend('SME between actual sound and sound predicted by via LMAN');


figure(7);
plot(e_lman_sound2); 
xlabel('Learning steps'); 
ylabel('Error');
title('Inverse error between sound predicited motoric activity via LMAN to tutor song');
legend('SME between actual sound and sound predicted by via LMAN');






#exit();

# 2. Phase
# learning the predictive inverse model during sleep from internally
# rehearsing the tutor song (phase 2) -- ie trying weights from memory
# to RA.

# initial weight matrix converting the auditory activity a into a motor activity mS (inverse model)
wpred=np.random.randn(n_ra,n_mem)/sqrt(n_mem); 

# Initialize error value for each learning step
e_pred=np.zeros(nlearn_pred); 

# motor activity produced by the inverse connections from the
# rehearsed memory through the auditory representation 
ra_pred=delayperm(w_lman.dot(song_aud_tut),0); 

# eligibility trace of length tau+0 for synapses from the memory
# to the motor area 
song_aud_tut_del=delayperm(song_aud_tut,tau+2*0); 


for i in xrange(0, nlearn_pred):

  # delayed silent input from the auditory memory 
  mpred=wpred.dot(song_aud_tut_del); 

  # estimation error; 
  diff_ra_lman=(ra_pred-mpred); 

  # learning rule for inverse weights
  # THis is for the connections from the memory direct to RA
  # m_inv - m_pred)*am -- normal instantanous learning.
  dwpred=diff_ra_lman.dot(song_aud_tut_del.T); # dot product? Transponed?

  # update inverse weights
  wpred=wpred+eta_pred*dwpred; # XXX can replace with +=? Probably not!
  
  # Mean squared error in motor estimation per time step and motor
  # neuron 
  e_pred[i]=sum(diff_ra_lman*diff_ra_lman)/(T*n_ra); 

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
atinv=delayperm(Q.dot(w_lman).dot(song_aud_tut),0); # does this keep the correct
# order of things? - ja, denn matrix multiplications is associative.

# Auditory activity in the memory area generated by the predictive inverse model
amtpred=delayperm(Q.dot(wpred).dot(song_aud_tut),tau); # why do I need tau here, but
# not above?

# Comparison between memorized and generated auditory activity

# Select a random auditory neuron
i=floor(np.random.rand()*n_aud);

figure(3);
clf;
#plot([song_aud_tut(i,:);atinv(i,:)]) # this could need a change as well from

plot(song_aud_tut[i])
hold("on");
plot(atinv[i,:])
hold("off");
legend("song_aud_tut", "atinv");
 # this could need a change as well from
# semicolon to coma.
xlabel('time steps'); 
title("Activity of one auditory neuron recieved if song generarted via \
the inverse model from auditory respresentation"); # ie a test how good an inverse of Q
# w_inv has become.
#r = corr(song_aud_tut(:),atinv(:));
#r = np.corrcoef(song_aud_tut[:],atinv[:])[0,1];
#fprintf('Corr. coeff. inv. model: %.4f\n',r);

figure(4);
clf;
plot(song_aud_tut[i])
hold("on");
plot(amtpred[i]);
xlabel('Time Steps'); 
title('Activity of one memory neuron generated by the predictive inverse model');
legend("song_aud_tut", "amtpred");
hold("off");


#r=corrcoef(song_aud_tut[:],amtpred[:]);
#fprintf('Corr. coeff. predictive inv. model: %.4f\n',r);
# ie a good an inverse of Q the preditictve pathway has become.

# Playback test
#%mplayback=wpred*delayperm(A*song_tut,0); % Takes auditory response in memory area during playback, an#d produces the motor sequence out of that
#%msong=wpred*delayperm(song_aud_tut,0); % The song generated motor sequence is trivially the same here becaus#e we assumed perfect memory, i.e. song_aud_tut=A*song
#%figure();clf;plot([mplayback(i,:); mplayback(i,:)]')
#%xlabel('time steps'); title('Mirroring is trivial because M has same drive during singing and playback');


