import numpy as np

#weight = np.genfromtxt("motoricW.gz", names=True) # this force to a structured dtype
#state = np.genfromtxt("motoric.gz", names=True)
weight = np.genfromtxt("motoricW.gz", skip_header = 1)
state = np.genfromtxt("motoric.gz", skip_header = 1)

spikes = state[:,0::2]
potentials = state[:,1::2]

from scitools.std import *
plot(potentials[:,0])
