#!/usr/bin/python

import ewave as wav
import numpy as np
import scipy.fftpack as fftp 
from scitools.std import *

rate = 10000 # sampling rate in Hz
samples_per_bin = 100

def load_wave(fname): 
    tutor_wav = wav.wavfile(fname);
    aud_raw = np.array(tutor_wav.read());

    figure(4);
    clf
    #plot(aud_raw[0:samples_per_bin])
    legend("Audio Input -- Audio Samples -- First Bin")

#    aud_raw -= aud_raw.mean();
    #   aud_raw /= aud_raw.std();
    aud_sample = np.reshape(aud_raw, (-1, samples_per_bin)) # 100 samples go into 1 bin
    aud_dct = fftp.dct(aud_sample.real.astype(float), type=2, norm='ortho')#.T # matrix right way round?

    # Normalise:
    aud_dct -= aud_dct.mean(); 
    aud_dct /= aud_dct.std();

    figure(5);
    clf;
    #plot(aud_dct[0])
    legend("FFT of the first bin");

    figure(6);
    clf;
    #plot(aud_dct[:,0])
    legend("Amplitude of first frequency of FFT across all bins");
         
    return aud_dct.T; # we need matrix in format that we have the 100 frequency as rows.

def save_wave(fname, song):

    output = fftp.idct(x=song.T, type=2, norm='ortho');
    #output = fftp.idct(x=song, type=2, norm='ortho');
    output2 = (output).ravel();

    maximum = (abs(output2)).max()

#    output2 *= (30000) / maximum; # scale to about 80% of 16 bit range

    figure(7)
    clf
    #plot(output2[0:100]);
    legend("Audio out -- first 100 sample points")
    out_wav = wav.wavfile(fname, mode="w", sampling_rate = rate, dtype = u'f', nchannels = 1);
    out_wav.write(data=output2, scale=True); # do I need to reshape, or is the automatic flattening the right thing?    
    #out_wav.write(data=output2); # do I need to reshape, or is the automatic flattening the right thing?
    out_wav.flush()

    return output2

#transf = load_wave("sinus_3s.wav")
#signal = save_wave("output.wav", transf)

# use this shell command to filter the 100 Hz component out:
# sox output.wav filtered.wav highpass 200 [norm?]

# use this shell command to record new files:
# arecord -r 10000 -d 1 -f S16_LE hallo.wav 



