'''PySoundTool: visualization and analysis of sound in Python
Copyright (C) 2019  Aislyn Rose

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.
'''

import sys
from scipy.io.wavfile import read
import numpy as np
from python_speech_features import mfcc, logfbank
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hamming, hann, resample

from .soundprep import resample_audio


def create_signal(freq=200, amplitude=0.4, samplerate=8000, dur_sec=0.25):
    #The variable `time` holds the expected number of measurements taken: 
    time = get_time_points(dur_sec, samplerate=samplerate)
    # unit circle: 2pi equals a full circle
    full_circle = 2 * np.pi
    #TODO: add namedtuple
    sinewave_samples = amplitude * np.sin((freq*full_circle)*time)
    return sinewave_samples, samplerate

def get_time_points(dur_sec,samplerate):
    #duration in seconds multiplied by the sampling rate. 
    time = np.linspace(0, dur_sec, int(dur_sec*samplerate))
    return time

def create_noise(num_samples, amplitude=0.025, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    noise = amplitude * np.random.randn(num_samples)
    return noise

        
def load_sound(sound, samplerate=None):
    if isinstance(sound, str):
        sr, data = read(sound)
        if samplerate and samplerate != sr:
            data, sr = resample_audio(data, sr, samplerate)
    else:
        sr = samplerate
        data = sound
    return data, sr
     
     
def visualize_signal(sound, samplerate=None, save_pic=False, name4pic=None):
    data, sr = load_sound(sound, samplerate=samplerate)
    dur_sec = len(data) / sr
    time_sec = get_time_points(dur_sec, sr)
    plt.clf()
    plt.plot(time_sec, data)
    plt.xlabel('Time (sec)')
    plt.ylabel('Amplitude')
    plt.title('Sound wave')
    if save_pic:
        outputname = name4pic or 'soundsignal'
        plt.savefig('{}.png'.format(outputname))
    else:
        plt.show()
    

def visualize_feats(sound, features='fbank', win_size_ms = 20, \
    win_shift_ms = 10,num_filters=40,num_mfcc=40, samplerate=None,\
        save_pic=False, name4pic=None):
    '''Visualize feature extraction depending on set parameters
    
    Parameters
    ----------
    sound : str or numpy.ndarray
        If str, wavfile (must be compatible with scipy.io.wavfile). Otherwise 
        the samples of the sound data. Note: in the latter case, `samplerate`
        must be declared.
    features : str
        Either 'mfcc' or 'fbank' features. MFCC: mel frequency cepstral
        coefficients; FBANK: mel-log filterbank energies (default 'fbank')
    win_size_ms : int or float
        Window length in milliseconds for Fourier transform to be applied
        (default 20)
    win_shift_ms : int or float 
        Window overlap in milliseconds; default set at 50% window size 
        (default 10)
    num_filters : int
        Number of mel-filters to be used when applying mel-scale. For 
        'fbank' features, 20-128 are common, with 40 being very common.
        (default 40)
    num_mfcc : int
        Number of mel frequency cepstral coefficients. First coefficient
        pertains to loudness; 2-13 frequencies relevant for speech; 13-40
        for acoustic environment analysis or non-linguistic information.
        Note: it is not possible to choose only 2-13 or 13-40; if `num_mfcc`
        is set to 40, all 40 coefficients will be included.
        (default 40). 
    samplerate : int, optional
        The sample rate of the sound data or the desired sample rate of
        the wavfile to be loaded. (default None)
    '''
    data, sr = load_sound(sound, samplerate=samplerate)
    win_samples = int(win_size_ms * sr // 1000)
    if 'fbank' in features:
        feats = logfbank(data,
                         samplerate=sr,
                         winlen=win_size_ms * 0.001,
                         winstep=win_shift_ms * 0.001,
                         nfilt=num_filters,
                         nfft=win_samples)
        axis_feature_label = 'Mel Filters'
    elif 'mfcc' in features:
        feats = mfcc(data,
                     samplerate=sr,
                     winlen=win_size_ms * 0.001,
                     winstep=win_shift_ms * 0.001,
                     nfilt=num_filters,
                     numcep=num_mfcc,
                     nfft=win_samples)
        axis_feature_label = 'Mel Freq Cepstral Coefficients'
    feats = feats.T
    plt.clf()
    plt.pcolormesh(feats)
    plt.xlabel('Frames (each {} ms)'.format(win_size_ms))
    plt.ylabel('Num {}'.format(axis_feature_label))
    plt.title('{} Features'.format(features.upper()))
    if save_pic:
        outputname = name4pic or 'visualize{}feats'.format(features.upper())
        plt.savefig('{}.png'.format(outputname))
    else:
        plt.show()
