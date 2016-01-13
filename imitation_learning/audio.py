#!/usr/bin/python

import ewave as wav
import numpy as np
import scipy.fftpack as fftp 
from scitools.std import *

rate = 10000 # sampling rate in Hz
bins = 100

def load_wave(fname): 
    tutor_wav = wav.wavfile(fname);
    aud_raw = np.array(tutor_wav.read());

    figure(1);
    clf
    legend("Audio Input -- raw")
    plot(aud_raw[0:100])

#    aud_raw -= aud_raw.mean();
    #   aud_raw /= aud_raw.std();
    aud_sample = np.reshape(aud_raw, (bins, -1)) # samples into 100 time bins.
    aud_dct = fftp.dct(aud_sample.real.astype(float), type=2, norm='ortho').T # matrix right way round?

    # Normalise:
    aud_dct -= aud_dct.mean(); 
    aud_dct /= aud_dct.std();

    figure(2);
    clf;
    legend("FFT");
    plot(aud_dct[:,0])
         

    return aud_dct;

def save_wave(fname, song):
    output = fftp.idct(x=song.T, type=2, norm='ortho');
    output2 = (output).ravel();
    figure(3)
    clf
    legend("Audio out")
    plot(output2[0:100]);
    out_wav = wav.wavfile(fname, mode="w", sampling_rate = rate, dtype = u'f', nchannels = 1);
    out_wav.write(data=output2, scale=True); # do I need to reshape, or is the automatic flattening the right thing?
    out_wav.flush()

    return output2

transf = load_wave("startreck.wav")
signal = save_wave("output.wav", transf)

# use this shell command to filter the 100 Hz component out:
# sox output.wav filtered.wav highpass 200 [norm?]



