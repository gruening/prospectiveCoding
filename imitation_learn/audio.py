#!/usr/bin/python

import ewave as wav
import numpy as np
import scipy.fftpack as fftp 
from scitools.std import *

rate = 10000 # sampling rate in Hz
bins = 2

def create_wav() :
    sample = wav.wavfile("sinus.wav", mode = 'w', sampling_rate = rate);
#    rate = sample.sampling_rate;

    T = 1         # sample duration (seconds)
    f = 2200     # sound frequency (Hz)
    t = np.linspace(0, T, T*rate, endpoint=False)
    x = np.sin(2*np.pi * f * t)
#    x += np.sin(8*np.pi * f * t)

    sample.write(x)
    sample.flush()


def load_wave(fname): 
    tutor_wav = wav.wavfile(fname);
    aud_raw = np.array(tutor_wav.read());
 #   aud_raw -= aud_raw.mean();
#    aud_raw /= aud_raw.std();
    aud_sample = np.reshape(aud_raw, (bins, -1)) # samples into 100 time bins.
    aud_dct = fftp.dct(aud_sample.real.astype(float), type=2, norm='ortho').T # matrix right way round?

    # Normalise:
    #aud_dct -= aud_dct.mean(); 
    #aud_dct /= aud_dct.std();

    return aud_dct;

def save_wave(fname, song):
    output = fftp.idct(x=song.T, type=2, norm='ortho');
#    output -= output.mean();
#    output /= output.std();
    output2 = (output.T).ravel();
    out_wav = wav.wavfile(fname, mode="w", sampling_rate = rate);
    out_wav.write(data=output2); # do I need to reshape, or is the automatic flattening the right thing?
    out_wav.flush()

    return output


#create_wav();

transf = load_wave("sinus.wav")
#sound = load_wave("startreck.wav")
signal = save_wave("output.wav", transf)


