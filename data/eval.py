#!/usr/bin/python

import numpy as np

# weights of the last (ie most trained) simulation
weight = np.genfromtxt("motoricW.gz", skip_header = 1)

# state, ie whether a neuron spiked or not etc for all simulations
state = np.genfromtxt("motoric.gz", skip_header = 1)

# input spike trains?
inputs = np.genfromtxt("motoricI.gz", skip_header = 1);

spikes = state[:,0::2] # extract spikes
potentials = state[:,1::2] # extract potentials

# get a list of spike times.
events = np.nonzero(spikes)

from scitools.std import *

plot(potentials[:,0]) # potential of neuron 0
title("Potential of one neuron, here 0") #ok

figure();
plot(events[0],events[1], "x") # spike raster
title("Spike Raster") # ok

figure();
plot([sum(spikes[:,x]) for x in np.arange(2,len(spikes[0]))], "x") # distribution of spikes across neurons
title("Distribution of Spikes across neurons"); # ok

#figure()
#plot([sum(spikes[:,x]) for x in np.arange(len(spikes[0]))], "o") 

figure()
plot([sum(spikes[x]) for x in np.arange(len(spikes))], "x") # distribution of spikes across time
title("Distribution of Spikes across time") #ok


# only print spikes in the last epoch (of how many)
last_spikes = spikes[range(9000,10000),]
figure()
hold("on")
plot(last_spikes[0], last_spikes[1], "x")
plot(100000*inputs[:,0]) # plot input curve
hold("off")
title("Spikes events from last epoch together with the input curve")
