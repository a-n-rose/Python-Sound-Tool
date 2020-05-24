
import sys
from scipy.io.wavfile import read
import numpy as np
from python_speech_features import mfcc, logfbank
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hamming, hann, resample
import librosa
import librosa.display as display

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
     
    
# TODO Clean up   
# TODO add graph for each channel? For all feature types?
def visualize_feats(feature_matrix, feature_type, 
                    save_pic=False, name4pic=None, scale=None,
                    title=None, sample_rate=None):
    '''Visualize feature extraction; frames on x axis, features on y axis. Uses librosa to scale the data if scale applied.
    
    Parameters
    ----------
    feature_matrix : np.ndarray [shape=(num_samples,), (num_samples, num_channels), or (num_features, num_frames), dtype=np.float].
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
            dur_sec = feature_matrix.shape[0] / sample_rate
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
                ##display.waveplot(data,sr=sample_rate)
        x_axis_label += ' across {} channel(s)'.format(channel+1)
    else:
        plt.pcolormesh(feature_matrix.T)
        ## display.specshow(feature_matrix.T, sr=sample_rate)
        plt.colorbar(label=energy_label)
    plt.xlabel(x_axis_label)
    plt.ylabel(axis_feature_label)
    # if feature_matrix has multiple frames, not just one
    if feature_matrix.shape[1] > 1 and 'signal' not in feature_type:
        # the xticks basically show time but need to be multiplied by 0.01
        plt.xlabel('Time (sec)') 
        locs, labels = plt.xticks()
        new_labels=[str(round(i*0.01,1)) for i in locs]
        plt.xticks(ticks=locs,labels=new_labels)
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

def get_feats(sound, 
              features='fbank', 
              win_size_ms = 20, 
              win_shift_ms = 10,
              num_filters=40,
              num_mfcc=None, 
              samplerate=None, 
              limit=None,
              mono=None):
    '''Feature extraction depending on set parameters; frames on y axis, features x axis
    
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
        (default None). 
    samplerate : int, optional
        The sample rate of the sound data or the desired sample rate of
        the wavfile to be loaded. (default None)
    limit : float, optional
        Time in seconds to limit in loading a signal. (default None)
    mono: bool, optional
        For loading an audiofile, True will result in only one channel of 
        data being loaded; False will allow additional channels be loaded. 
        (default None, which results in mono channel data)
        
    Returns
    -------
    feats : tuple (num_samples, sr) or np.ndarray [size (num_frames, num_filters) dtype=np.float or np.complex]
        Feature data. If `feature_type` is 'signal', returns a tuple containing samples and sampling rate. If `feature_type` is of another type, returns np.ndarray with shape (num_frames, num_filters/features)
    '''
    if isinstance(sound, str):
        if mono is None:
            mono = True
        data, sr = librosa.load(sound, sr=samplerate, duration=limit, mono=mono)
        if mono is False and len(data.shape) > 1:
            index_samples = np.argmax(data.shape)
            index_channels = np.argmin(data.shape)
            num_channels = data.shape[index_channels]
            # transpose data to be (samples, num_channels) rather than (num_channels, samples)
            if index_channels == 0:
                data = data.T 
            # remove additional channel for 'stft', 'fbank' etc. feature
            # extraction
            if 'signal' not in features and num_channels > 1:
                data = data[:,0]
    else:
        if samplerate is None:
            raise ValueError('No samplerate given. Either provide '+\
                'filename or appropriate samplerate.')
        data, sr = sound, samplerate
    try:
        if 'fbank' in features:
            feats = librosa.feature.melspectrogram(
                data,
                sr = sr,
                n_fft = int(win_size_ms * sr // 1000),
                hop_length = int(win_shift_ms*0.001*sr),
                n_mels = num_filters).T
            # have found better results if not conducted:
            # especially if recreating audio signal later
            #feats -= (np.mean(feats, axis=0) + 1e-8)
        elif 'mfcc' in features:
            if num_mfcc is None:
                num_mfcc = num_filters
            feats = librosa.feature.mfcc(
                data,
                sr = sr,
                n_mfcc = num_mfcc,
                n_fft = int(win_size_ms * sr // 1000),
                hop_length = int(win_shift_ms*0.001*sr),
                n_mels = num_filters).T
            #feats -= (np.mean(feats, axis=0) + 1e-8)
        elif 'stft' in features:
            feats = librosa.stft(
                data,
                n_fft = int(win_size_ms * sr // 1000),
                hop_length = int(win_shift_ms*0.001*sr)).T
            #feats -= (np.mean(feats, axis=0) + 1e-8)
        elif 'signal' in features:
            feats = (data, sr)
    except librosa.ParameterError as e:
        feats = get_feats(np.asfortranarray(data),
                          features = features,
                          win_size_ms = win_size_ms,
                          win_shift_ms = win_shift_ms,
                          num_filters = num_filters,
                          num_mfcc = num_mfcc,
                          samplerate = sr,
                          limit = limit,
                          mono = mono)
    return feats

def visualize_audio(audiodata, feature_type='fbank', win_size_ms = 20, \
    win_shift_ms = 10, num_filters=40, num_mfcc=40, samplerate=None,\
        save_pic=False, name4pic=None, power_scale=None, mono=None):
    '''Visualize feature extraction depending on set parameters. Does not use Librosa.
    
    Parameters
    ----------
    audiodata : str or numpy.ndarray
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
    mono : bool, optional
        When loading an audiofile, True will limit number of channels to
        one; False will allow more channels to be loaded. (default None, 
        which results in mono channel loading.)
    '''
    feats = get_feats(audiodata, features=feature_type, 
                      win_size_ms = win_size_ms, win_shift_ms = win_shift_ms,
                      num_filters=num_filters, num_mfcc = num_mfcc, samplerate=samplerate,
                      mono=mono)
    if 'signal' in feature_type:
        feats, sr = feats
    else:
        sr = None
    visualize_feats(feats, feature_type=feature_type, sample_rate=sr,
                    save_pic = save_pic, name4pic=name4pic, scale=power_scale)
