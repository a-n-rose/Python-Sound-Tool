import sys
from scipy.io.wavfile import read
import numpy as np
from python_speech_features import mfcc, logfbank
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hamming, hann, resample
import librosa

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
    time = np.linspace(0, dur_sec, int(np.floor(dur_sec*samplerate)))
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
     
    
# TODO Test for multiple channels   
def visualize_feats(feature_matrix, feature_type, 
                    save_pic=False, name4pic=None, scale=None,
                    title=None, sample_rate=None):
    '''Visualize feature extraction; frames on x axis, features on y axis. Uses librosa to scale the data if scale applied.
    
    Parameters
    ----------
    feature_matrix : np.ndarray [shape=(len(data),), (len(data), num_features)]
        or (len(data), num_channels), dtype=np.float].
        Matrix of features. If the features are not of type 'signal' and the
        shape is 1 D, one dimension will be added to be plotted with a colormesh.
    feature_type : str
        Options: 'signal', 'stft', 'mfcc', or 'fbank' features, or 
        what user would like to name the feature set.
        signal: the 1 D samples of sound.
        STFT: short-time Fourier transform
        MFCC: mel frequency cepstral coefficients.
        FBANK: mel-log filterbank energies (default 'fbank').
    save_pic : bool
        True to save image as .png; False to just plot it.
    name4pic : str, optional
        If `save_pic` set to True, the name the image should be saved under.
    scale : str, optional
        If features need to be adjusted, e.g. from power to decibels. 
        Default is None.
    title : str, optional
        The title for the graph. If None, `feature_type` is used.
    sample_rate : int, optional
        Useful for plotting a signal type feature matrix. Allows x-axis to be
        presented as time in seconds.
    '''
    # ensure real numbers
    if isinstance(feature_matrix[0], np.complex):
        feature_matrix = feature_matrix.real
    # features presented via colormesh need 2D format.
    if len(feature_matrix.shape) == 1:
        feature_matrix = np.expand_dims(feature_matrix, axis=1)
    if 'fbank' in feature_type:
        axis_feature_label = 'Num Mel Filters'
    elif 'mfcc' in feature_type:
        axis_feature_label = 'Num Mel Freq Cepstral Coefficients'
    elif 'stft' in feature_type:
        axis_feature_label = 'Number of frames'
    elif 'signal' in feature_type:
        axis_feature_label = 'Amplitude'
    else:
        axis_feature_label = 'Energy'
    if scale is None or feature_type == 'signal':
        energy_label = 'energy'
        pass
    elif scale == 'power_to_db':
        feature_matrix = librosa.power_to_db(feature_matrix)
        energy_label = 'decicels'
    elif scale == 'db_to_power':
        feature_matrix = librosa.db_to_power(feature_matrix)
        energy_label = 'power'
    elif scale == 'amplitude_to_db':
        feature_matrix = librosa.amplitude_to_db(feature_matrix)
        energy_label = 'decibels'
    elif scale == 'db_to_amplitude':
        feature_matrix = librosa.db_to_amplitude(feature_matrix)
        energy_label = 'amplitude'
    plt.clf()
    if feature_type != 'signal':
        x_axis_label = 'Frequency bins'
    else:
        x_axis_label = 'Samples over time'
    if feature_type == 'signal':
        # transpose matrix if second dimension is larger - probably 
        # because channels are in first dimension. Expect in second dimension
        if not feature_matrix.shape[0] > feature_matrix.shape[1]:
            feature_matrix = feature_matrix.T
        if sample_rate is not None:
            x_axis_label = 'Time (sec)'
            dur_sec = len(feature_matrix) / sample_rate
            time_sec = get_time_points(dur_sec, sample_rate)
            for channel in range(feature_matrix.shape[1]):
                data = feature_matrix[:,channel]
                # overlay the channel data
                plt.plot(time_sec, data)
        else:
            for channel in range(feature_matrix.shape[1]):
                data = feature_matrix[:,channel]
                # overlay the channel data
                plt.plot(data)
        x_axis_label += ' across {} channel(s)'.format(channel+1)
    else:
        plt.pcolormesh(feature_matrix.T)
        plt.colorbar(label=energy_label)
    plt.xlabel(x_axis_label)
    plt.ylabel(axis_feature_label)
    if feature_matrix.shape[1] > 1:
        plt.xlabel('Number of processed frames')
        plt.ylabel('Frequency bins')
    if title is None:
        plt.title('{} Features'.format(feature_type.upper()))
    else:
        plt.title(title)
    if save_pic:
        outputname = name4pic or 'visualize{}feats'.format(feature_type.upper())
        plt.savefig('{}.png'.format(outputname))
    else:
        plt.show()

def visualize_audiofile(audiofile, feature_type='fbank', win_size_ms = 20, \
    win_shift_ms = 10, num_filters=40, num_mfcc=40, samplerate=None,\
        save_pic=False, name4pic=None):
    '''Visualize feature extraction depending on set parameters. Does not use Librosa.
    
    Parameters
    ----------
    audiofile : str or numpy.ndarray
        If str, wavfile (must be compatible with scipy.io.wavfile). Otherwise 
        the samples of the sound data. Note: in the latter case, `samplerate`
        must be declared.
    feature_type : str
        Options: 'signal', 'mfcc', or 'fbank' features. 
        MFCC: mel frequency cepstral
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
    if isinstance(audiofile, str):
        data, sr = load_sound(audiofile, samplerate=samplerate)
    else:
        data = audiofile
        sr = samplerate
    win_samples = int(win_size_ms * sr // 1000)
    if 'signal' in feature_type:
        dur_sec = len(data) / sr
        time_sec = get_time_points(dur_sec, sr)
        plt.clf()
        plt.plot(time_sec, data)
        plt.xlabel('Time (sec)')
        plt.ylabel('Amplitude')
        plt.title('Sound wave')
    else:
        if 'fbank' in feature_type:
            feats = logfbank(data,
                            samplerate=sr,
                            winlen=win_size_ms * 0.001,
                            winstep=win_shift_ms * 0.001,
                            nfilt=num_filters,
                            nfft=win_samples)
            axis_feature_label = 'Mel Filters'
        elif 'mfcc' in feature_type:
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
        plt.title('{} Features'.format(feature_type.upper()))
    if save_pic:
        outputname = name4pic or 'visualize{}feats'.format(feature_type.upper())
        plt.savefig('{}.png'.format(outputname))
    else:
        plt.show()
