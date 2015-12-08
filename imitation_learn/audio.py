#!/usr/bin/python

import ewave as wav
import numpy as np
import scipy #.fftpack as fft # this is not the numpy implementation! Sth better available?
from scitools.std import *


rate = 10000

def create_wav() :
    sample = wav.wavfile("test.wav", mode = 'w', sampling_rate = rate);
#    rate = sample.sampling_rate;

    T = 1         # sample duration (seconds)
    f = 2200     # sound frequency (Hz)
    t = np.linspace(0, T, T*rate, endpoint=False)
    x = np.sin(2*np.pi * f * t)
#    x += np.sin(8*np.pi * f * t)

    sample.write(x)
    sample.flush()
    # sample.fp.close()

create_wav();


def load_wave(fnmae): 
    tutor_wav = wav.wavfile("test.wav");
    aud_raw = np.array(tutor_wav.read());
    aud_sample = np.reshape(aud_raw, (100, -1)) # sample into 100 time bins.
    aud_dct = scipy.fftpack.dct(aud_sample.real.astype(float)) # matrix right way round?

    # Normalise:
    aud_dct -= aud_dct.mean(); 
    aud_dct /= aud_dct.std();

    return aud_dct;

sound = load_wave("test.wav")


output = scipy.fftpack.idct(sound);
output /= output.std();
out_wav = wav.wavfile("output.wav", mode="w", sampling_rate = rate);
out_wav.write(output); # do I need to reshape, or is the automatic flattening the right thing?
out_wav.flush()
