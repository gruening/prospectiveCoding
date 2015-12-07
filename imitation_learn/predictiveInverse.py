#!/usr/bin/python

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

#nm=200; # number of motor neurons (RA)
nm=7; # 3 for testing, or 7
nsong=nm; # dimension of song, ie number of "aciustic"/artilatory degrees of freedom"
na=nm; # number of auditory neurons
nam=nm; # number of auditory memory neurons which memorize the tutor song (for one time step)
T=100; # duration of subsongs and of tutor song  [ms?]
tau=10; # 10 Delay of auditory activity with respect to the generating motor activity (syrinx + auditory pathway) [ms?] -- isnt that rather in the area of 50ms?
oneStep=1; # Propagation delay of one neuronal connection [ms?]
nlearn_inv=300; # learning steps for inverse model (ie to learn reverse of "acoustic" mapping)
nlearn_pred=1; # 100 learning steps for predicitve inverse model (ie to learn direct connections from memory to motor neurons
#nlearn_pred=1; # for testing inv learning only
eta_inv=0.002; # learning rate for inverse model
eta_pred=0.001; # learning rate for predictive inverse model
epsi=0; # Parameter to regularize the matrices -- not needed?

# syrinx; converts the motor activity signal m into a sound (here just matrix)
S = (np.random.randn(nsong,nm) + epsi*np.eye(nm)) / sqrt(nm);

# auditory pathway; converts the song into an auditory signal (here just a matrix)
A = (np.random.randn(na,nsong) + epsi*np.eye(nm)) / sqrt(na); 

# motor activity to generate the tutor song, ie each time step is vector of nm-activities
# This is what we want to learn!
mtut=np.random.randn(nm,T); 

# acoustic representation of tutor song (delayed)
# This is what we should hear if we reproduced the tutor song.
song_tut=delayperm(S.dot(mtut),tau-oneStep); 

# auditory representation of tutor song auditory -- generated by the
# tutor song, or by rehearsing of the memory    
# Is this our imprint? 
atut=delayperm(A.dot(song_tut),oneStep);  

# memorized tutor song, one time step ahead of the rehearsed auditory
# activity; if memory perfect, also generated by playback  
# Is this our imprint?
amtut=A.dot(song_tut);  

Q=A.dot(S); # Total motor to auditory transformation, eg atut=delayperm(Q*mtut,tau)

# 1. Phase:
# learning the inverse model from  babbling a subsong and
# trying to reproduce the motor pattern it generated (phase 1)  
# ie this is prospective learning.

# initial weight matrix converting the auditory representation a into a motor
# activity (inverse model) (ie looking for sth like an inverse of Q)
winv=np.random.randn(nm,na)/sqrt(na); 

# Initialize error values for each learning step
e_inv = np.zeros(nlearn_inv); 

#m=mtut;

for i in xrange(0,nlearn_inv):

# random motor activity to produce the subsong different each epoch.
  m = mtut + 1/100 * np.random.randn(nm,T); 
#  m= np.random.randn(nm,T); 

  # auditory activity produced by the subsong 
  a=delayperm(Q.dot(m),tau); 

  # motor activity predicted from the auditory activity
  minv=delayperm(winv.dot(a),oneStep); 
  
  # estimation error; postsynaptic eligibility trace of length
  # tau+oneStep
  # The question is how this is implmeneted: we need an elegibility
  # trace.
  # Sparse bursty firing might simplify the biological implementation
  # of the trace of m.
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
title('Inverse error of motoric activity at RA vs tutor motoric activity via $w_{inv}$');
legend('SME between tutor motoric and "inverse" activitey');

# 2. Phase
# learning the predictive inverse model during sleep from internally
# rehearsing the tutor song (phase 2) -- ie trying weights from memory
# to RA.

# initial weight matrix converting the auditory activity a into a motor activity mS (inverse model)
wpred=np.random.randn(nm,nam)/sqrt(nam); 

# Initialize error value for each learning step
e_pred=np.zeros(nlearn_pred); 

# motor activity produced by the inverse connections from the
# rehearsed memory through the auditory representation 
minv=delayperm(winv.dot(atut),oneStep); 

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
  wpred=wpred+eta_pred*dwpred; # XXX can replace with +=? Probably not!
  
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

# Select a random auditory neuron
i=floor(np.random.rand()*na);

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

