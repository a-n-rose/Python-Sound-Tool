'''dsp module contains functions pertaining to the actual generation,
manipulation, and analysis of sound. This ranges from generating sounds to
calculating sound to noise ratio.
'''
###############################################################################
import sys, os
import inspect
import numpy as np
from numpy.fft import fft, rfft, ifft, irfft
from scipy.signal import hamming, hann, resample
import librosa
import librosa.display as display
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)

import pysoundtool as pyst

def generate_sound(freq=200, amplitude=0.4, sr=8000, dur_sec=0.25):
    '''Generates a sound signal with the provided parameters. Signal begins at 0.
    
    Parameters
    ----------
    freq : int, float
        The frequency in Hz the signal should have (default 200 Hz). This pertains
        to the number of ossicliations per second.
    amplitude : int, float
        The parameter controling how much energy the signal should have. 
        (default 0.4)
    sr : int 
        The sampling rate of the signal, or how many samples make up the signal 
        per second. (default 8000)
        
    Returns
    -------
    sound_samples : np.ndarray [size = ()]
        The samples of the generated sound 
    sr : int 
        The sample rate of the generated signal
        
    Examples
    --------
    >>> sound, sr = generate_sound(freq=5, amplitude=0.5, sr=5, dur_sec=1)
    >>> sound 
    array([ 0.000000e+00,  5.000000e-01,  3.061617e-16, -5.000000e-01, -6.123234e-16])
    >>> sr
    5
    '''
    #The variable `time` holds the expected number of measurements taken: 
    time = get_time_points(dur_sec, sr=sr)
    # unit circle: 2pi equals a full circle
    full_circle = 2 * np.pi
    #TODO: add namedtuple
    sound_samples = amplitude * np.sin((freq*full_circle)*time)
    return sound_samples, sr

def get_time_points(dur_sec,sr):
    '''Get evenly spaced time points from zero to length of `dur_sec`.
    
    The time points align with the provided sample rate, making it easy 
    to plot a signal with a time line in seconds.
    
    Parameters
    ----------
    dur_sec : int, float
        The amount of time in seconds 
    sr : int
        The sample rate relevant for the signal
    
    Returns
    -------
    time : np.ndarray [size = (num_time_points,)]
    
    Examples
    --------
    >>> # 50 milliseconds at sample rate of 100 (100 samples per second)
    >>> x = get_time_points(0.05,100)
    >>> x.shape
    (5,)
    >>> x
    array([0.    , 0.0125, 0.025 , 0.0375, 0.05  ])
    '''
    time = np.linspace(0, dur_sec, int(np.floor(dur_sec*sr)))
    return time

def generate_noise(num_samples, amplitude=0.025, random_seed=None):
    '''Generates noise to be of a certain amplitude and number of samples.
    
    Useful for adding noise to another signal of length `num_samples`.
    
    Parameters
    ----------
    num_samples : int
        The number of total samples making up the noise signal.
    amplitude : float 
        Allows the noise signal to be louder or quieter. (default 0.025)
    random_seed : int, optional
        Useful for repeating 'random' noise samples.
        
    Examples
    --------
    >>> noise = generate_noise(10, random_seed = 0)
    >>> noise
    array([0.04410131, 0.01000393, 0.02446845, 0.05602233, 0.04668895])
    '''
    if random_seed is not None:
        np.random.seed(random_seed)
    noise = amplitude * np.random.randn(num_samples)
    return noise

def set_signal_length(samples, numsamps):
    '''Sets audio signal to be a certain length. Zeropads if too short.
    
    Useful for setting signals to be a certain length, regardless of how 
    long the audio signal is.
    
    Parameters
    ----------
    samples : np.ndarray [size = (num_samples, num_channels), or (num_samples,)]
        The array of sample data to be zero padded.
    numsamps : int 
        The desired number of samples. 
        
    Returns
    -------
    data : np.ndarray [size = (numsamps, num_channels), or (numsamps,)]
        Copy of samples zeropadded or limited to `numsamps`.
        
    Examples
    --------
    >>> import numpy as np
    >>> input_samples = np.array([1,2,3,4,5])
    >>> output_samples = set_signal_length(input_samples, numsamps = 8)
    >>> output_samples
    array([1, 2, 3, 4, 5, 0, 0, 0])
    >>> output_samples = set_signal_length(input_samples, numsamps = 4)
    >>> output_samples
    array([1, 2, 3, 4])
    '''
    data = samples.copy()
    if data.shape[0] < numsamps:
        diff = numsamps - data.shape[0]
        if len(data.shape) > 1:
            signal_zeropadded = np.zeros(
                (data.shape[0] + int(diff),data.shape[1]))
        else:
            signal_zeropadded = np.zeros(
                (data.shape[0] + int(diff),))
        for i, row in enumerate(data):
            signal_zeropadded[i] = row
        data = signal_zeropadded
    else:
        if len(data.shape) > 1:
            data = data[:numsamps,:]
        else:
            data = data[:numsamps]
    # ensure returned data same dtype as input 
    data = pyst.utils.match_dtype(data, samples)
    return data

def scalesound(data,min_val=-1,max_val=1):
    '''Scales the input array to range between `min_val` and `max_val`. 
    
    Parameters
    ----------
    data : np.ndarray [size = (num_samples,)]
        Original samples
    min_val : int, float
        The minimum value the dataset is to range from (default -1)
    max_val : int, float
        The maximum value the dataset is to range from (default 1)
    
    Returns
    -------
    samples : np.ndarray [size = (num_samples,)]
        Copy of original data, scaled to the min and max values.
        
    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> input_samples = np.random.random_sample((5,))
    >>> input_samples
    array([0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ])
    >>> # default setting: between -1 and 1
    >>> output_samples = scalesound(input_smaples)
    >>> output_samples 
    array([-0.14138 ,1., 0.22872961, -0.16834299, -1.])
    >>> # range between -100 and 100
    >>> output_samples = scalesound(input_samples, min_val = -100, max_val = 100)
    >>> output_samples
    array([ -14.13800026,100., 22.87296052,-16.83429866,-100.])
    '''
    samples = data.copy()
    samples = np.interp(samples,(samples.min(), samples.max()),(min_val, max_val))
    return samples

def normalize(data, max_val=None, min_val=None):
    '''Normalizes data.
    
    This is usefule if you have predetermined max and min values you want to normalize
    new data with. Should work with stereo sound: TODO test for stereo sound.
    
    Parameters
    ----------
    data : np.ndarray
        Data to be normalized
    max_val : int or float, optional
        Predetermined maximum value. If None, will use max value
        from `data`.
    min_val : int or float, optional
        Predetermined minimum value. If None, will use min value
        from `data`.
        
    Returns
    -------
    normed_data : np.ndarray [size = (num_samples,)]
    
    Examples
    --------
    >>> # using the min and max of a previous dataset:
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> input_samples = np.random.random_sample((5,))
    >>> input_samples
    array([0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548 ])
    >>> np.random.seed(40)
    >>> previous_samples = np.random.random_sample((5,))
    >>> previous_samples
    array([0.40768703, 0.05536604, 0.78853488, 0.28730518, 0.45035059])
    >>> max_prev = np.max(previous_samples)
    >>> min_prev = np.min(previous_samples)
    >>> output_samples = normalize(input_samples, min_val = min_prev, max_val = max_prev)
    >>> output_samples
    array([0.67303388, 0.89996095, 0.74661839, 0.66767314, 0.50232462])
    '''
    if max_val is None:
         normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        normed_data = (data - min_val) / (max_val - min_val)
    return normed_data

def resample_audio(samples, sr_original, sr_desired):
    '''Allows audio samples to be resampled to desired sample rate.
    
    Parameters
    ----------
    samples : np.ndarray [size = (num_samples,)]
        The samples to be resampled.
    sr_original : int 
        The orignal sample rate of the samples.
    sr_desired : int 
        The desired sample rate of the samples.
        
    Returns
    -------
    resampled : np.ndarray [size = (num_samples_resampled,)]
        The resampled samples.
    sr_desired : int 
        The newly applied sample rate
        
    Examples
    --------
    >>> import numpy as np
    >>> # example samples from 5 millisecond signal with sr 100 and frequency 10
    >>> input_samples = np.array([0.00e+00, 2.82842712e-01, 4.000e-01, 2.82842712e-01, 4.89858720e-17]
    >>> # we want to resample to 80 instead of 100 (for this example's sake)
    >>> output_samples = resample_audio(input_samples, sr_original = 100, sr_desired = 80)
    >>> output_samples
    array([-2.22044605e-17, 3.35408001e-01, 3.72022523e-01, 6.51178161e-02])
    '''
    time_sec = len(samples)/sr_original 
    num_samples = int(time_sec * sr_desired)
    resampled = resample(samples, num_samples)
    return resampled, sr_desired

def stereo2mono(data):
    '''If sound data has multiple channels, reduces to first channel
    
    Parameters
    ----------
    data : numpy.ndarray
        The series of sound samples, with 1+ columns/channels
    
    Returns
    -------
    data_mono : numpy.ndarray
        The series of sound samples, with first column
        
    Examples
    --------
    >>> import numpy as np
    >>> data = np.linspace(0,20)
    >>> data_2channel = data.reshape(25,2)
    >>> data_2channel[5]
    array([[0.        , 0.40816327],
       [0.81632653, 1.2244898 ],
       [1.63265306, 2.04081633],
       [2.44897959, 2.85714286],
       [3.26530612, 3.67346939]])
    >>> data_mono = stereo2mono(data_2channel)
    >>> data_mono[:5]
    array([0.        , 0.81632653, 1.63265306, 2.44897959, 3.26530612])
    '''
    data_mono = data.copy()
    if len(data.shape) > 1 and data.shape[1] > 1:
        data_mono = np.take(data,0,axis=-1) 
    return data_mono

# TODO add snr instead of scale
def add_backgroundsound(audio_main, audio_background, scale_background=1,
                        delay_mainsound_sec = None, total_len_sec=None):
    '''Adds a sound (i.e. background noise) to a target signal.
    
    If the sample rates of the two audio samples do not match, the sample
    rate of `audio_main` will be applied. (i.e. the `audio_background` will
    be resampled)
    
    Parameters
    ----------
    audio_main : str, pathlib.PosixPath, or tuple [size=((num_samples,), sr)]
        Sound file of the main sound (will not be modified; only delayed if 
        specified). If not path or string, should be a tuple containing the 
        audio samples and corresponding sample rate.
    audio_background : str, pathlib.PosixPath, or tuple [size=((num_samples,), sr)]
        Sound file of the background sound (will be modified /repeated to match
        or extend the length indicated). If not of type pathlib.PosixPath or
        string, should be a tuple containing the audio samples and corresponding 
        sample rate.
    scale_background : int, float
        The loudness of the sound to be added (default 1). The audio samples
        will be multiplied by this number.
    delay_mainsound_sec : int or float, optional
        Length of time in seconds the main sound will be delayed. For example, 
        if `delay_mainsound_sec` is set to 1, one second of the
        `audio_background` will be played before `audio_main` starts. 
        (default None)
    total_len_sec : int or float, optional
        Total length of combined sound in seconds. If none, the sound will end
        after target sound ends (default None).
    
    Returns
    -------
    combined : numpy.ndarray
        The samples of the sounds added together
    sr : int 
        The sample rate of the samples
        
    Examples
    --------
    >>> import numpy as np
    >>> sound_samples = np.array([1,2,3,4,5])
    >>> background_samples = np.array([1,1,1,1,1])
    >>> sr = 5 # doesn't mean anything in this example
    >>> combined, sr = add_backgroundsound((sound_samples, sr), (background_samples, sr))
    >>> combined
    array([2, 3, 4, 5, 6])
    >>> sr
    5
    >>> # increase the scale of the sound
    >>> combined, sr = add_backgroundsound((sound_samples, sr), (background_samples, sr), scale_background=1.5)
    array([2.5, 3.5, 4.5, 5.5, 6.5])
    >>> # decrese the scale of the sound
    >>> combined, sr = add_backgroundsound((sound_samples, sr), (background_samples, sr), scale_background = 0.5)
    array([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> # delay `main_sound`
    >>> combined, sr = add_backgroundsound((sound_samples, sr), (background_samples, sr), delay_mainsound_sec=1)
    >>> combined 
    array([1, 1, 1, 1, 1, 2, 3, 4, 5, 6])
    >>> # set total length without delay
    >>> combined, sr = add_backgroundsound((sound_samples, sr), (background_samples, sr), total_len_sec = 3)
    >>> combined
    array([2, 3, 4, 5, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    >>> # set total length with delay
    >>> combined, sr = add_backgroundsound((sound_samples, sr), (background_samples, sr), total_len_sec = 3, delay_mainsound_sec=1)
    array([1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 1, 1, 1, 1, 1])
    >>> # set total length with delay: target sound will get cut off
    >>> combined, sr = add_backgroundsound((sound_samples, sr), (background_samples, sr), total_len_sec = 1.5, delay_mainsound_sec=1)
    array([1, 1, 1, 1, 1, 2, 3])
    >>> # also works with stereo sound
    >>> sound = np.zeros((3,2))
    >>> sound[:,0] = np.array([1,2,3])
    >>> sound[:,1] = np.array([0,1,2])
    >>> sound
    array([[1., 0.],
        [2., 1.],
        [3., 2.]])
    >>> noise = np.array([1,1,1,])
    >>> sr = 3
    >>> input1 = sound, sr
    >>> input2 = noise, sr
    >>> combined, sr2 = pyst.dsp.add_backgroundsound(input1, input2)
    >>> combined
    array([[2., 1.],
        [3., 2.],
        [4., 3.]])
    '''
    input_type_main = pyst.utils.path_or_samples(audio_main)
    input_type_background = pyst.utils.path_or_samples(audio_background)
    if 'path' in input_type_main:
        target, sr = pyst.loadsound(audio_main)
    elif 'samples' in input_type_main:
        target, sr = audio_main
    if 'path' in input_type_background:
        sound2add, sr2 = pyst.loadsound(audio_background)
    elif 'samples' in input_type_background:
        sound2add, sr2 = audio_background
    if sr != sr2:
        sound2add, sr2 = pyst.dsp.resample_audio(sound2add, sr2, sr)
        assert sr2 == sr
        
    # make background same shape as signal
    if len(target.shape) != len(sound2add.shape):
        # ensure in shape (num_samples,) or (num_samples, num_channels)
        target = pyst.utils.shape_samps_channels(target)
        if len(target.shape) > 1:
            num_channels = target.shape[1]
        else:
            num_channels = 1
        sound2add = apply_num_channels(sound2add, num_channels)
            
    # TODO add snr calculation instead of scale
    sound2add = sound2add * scale_background
    if delay_mainsound_sec is None:
        delay_mainsound_sec = 0
    if total_len_sec is not None:
        total_samps = int(sr*total_len_sec)
    else:
        total_samps = len(target) + int(sr*delay_mainsound_sec)
    if total_samps < len(target) + delay_mainsound_sec:
        import warnings
        warnings.warn('The length of `audio_main` and `delay_mainsound_sec `'+\
            'exceeds `total_len_sec`. Some of `audio_main` will be cut off in '+\
                'the `combined` audio signal.')
    # make the background sound match the length of total samples
    sound2add = pyst.dsp.apply_length(sound2add, total_samps)
    # separate samples to add to the target signal
    target_sound = sound2add[int(sr*delay_mainsound_sec):len(target)+int(sr*delay_mainsound_sec)]
    # If target is longer than indicated length, shorten it
    if len(target_sound) < len(target):
        target = target[:len(target_sound)]
    combined = target_sound + target
    if delay_mainsound_sec:
        # set aside samples for beginning delay (if there is one)
        beginning_sound = sound2add[:int(sr*delay_mainsound_sec)]
        combined = np.concatenate((beginning_sound,combined))
    if total_len_sec:
        # set aside ending samples for ending (if sound is extended)
        ending_sound = sound2add[len(target)+int(sr*delay_mainsound_sec):total_samps]
        combined = np.concatenate((combined, ending_sound))
    return combined, sr

def apply_num_channels(sound_data, num_channels):
    '''Ensures `data` has indicated `num_channels`. 
    
    To increase number of channels, the first column will be duplicated. To limit 
    channels, channels will simply be removed.
    
    Parameters
    ----------
    sound_data : np.ndarray [size= (num_samples,) or (num_samples, num_channels)]
        The data to adjust the number of channels
    num_channels : int 
        The number of channels desired
        
    Returns
    -------
    data : np.ndarray [size = (num_samples, num_channels)]
    
    Examples 
    --------
    >>> import numpy as np
    >>> data = np.array([1, 1, 1, 1])
    >>> data_3d = apply_num_channels(data, 3)
    >>> data_3d
    array([[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])
    >>> data_2d = apply_num_channels(data_3d, 2)
    >>> data_2d
    array([[1, 1],
        [1, 1],
        [1, 1],
        [1, 1]])
    '''
    if len(sound_data.shape)== 1:
        data = np.expand_dims(sound_data, axis=1)
    else:
        data = sound_data
    diff = num_channels - data.shape[1]
    if diff < 0:
        # limit number of channels
        data = data[:,:num_channels]
        return data
    elif diff == 0:
        # no change necessary
        return sound_data
    # add channels
    duplicated_data = np.expand_dims(data[:,0], axis=1)
    for i in range(diff):
        data = np.append(data, duplicated_data, axis=1)
    return data

def apply_length(data, target_len):
    '''Extends a sound by repeating it until its `target_len`.
    If the `target_len` is shorter than the length of `data`, `data`
    will be shortened to the specificed `target_len`
    
    This is perhaps useful when working with repetitive or
    stationary sounds.
    
    Parameters
    ----------
    data : np.ndarray [size = (num_samples,) or (num_samples, num_channels)] 
        The data to be checked or extended in length. If shape (num_channels, num_samples),
        the data will be reshaped to (num_samples, num_channels).
    target_len : int 
        The length of samples the input `data` should be.
    
    Returns
    -------
    new_data : np.ndarray [size=(target_len, ) or (target_len, num_channels)]

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1,2,3,4])
    >>> pyst.dsp.apply_length(data, 12)
    array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
    >>> # two channels
    >>> data = np.zeros((3,2))
    >>> data[:,0] = np.array([0,1,2])
    >>> data[:,1] = np.array([1,2,3])
    >>> data
    array([[0., 1.],
        [1., 2.],
        [2., 3.]])
    >>> pyst.dsp.apply_length(data,5)
    array([[0., 1.],
        [1., 2.],
        [2., 3.],
        [0., 1.],
        [1., 2.]])
    '''
    if len(data) > target_len:
        new_data = data[:target_len]
        return new_data
    elif len(data) == target_len:
        new_data = data
        return new_data
    if len(data.shape) > 1:
        # ensure stereo in correct format (num_samples, num_channels)
        data = pyst.utils.shape_samps_channels(data)
        num_channels = data.shape[1]
    else:
        num_channels = 0
    if num_channels:
        new_data = np.zeros((target_len, num_channels))
    else:
        new_data = np.zeros((target_len,))
    row_id = 0
    while row_id < len(new_data):
        if row_id + len(data) > len(new_data):
            diff = row_id + len(data) - len(new_data)
            new_data[row_id:] += data[:-diff]
        else:
            new_data[row_id:row_id+len(data)] = data
        row_id += len(data)
    new_data = pyst.utils.match_dtype(new_data, data)
    return new_data

# TODO: raise error or only warning if original data cut off?
def zeropad_sound(data, target_len, sr, delay_sec=None):
    '''If the sound data needs to be a certain length, zero pad it.
    
    Parameters
    ----------
    data : numpy.ndarray [size = (num_samples,) or (num_samples, num_channels)]
        The sound data that needs zero padding. Shape (len(data),). 
    target_len : int 
        The number of samples the `data` should have
    sr : int
        The samplerate of the `data`
    delay_sec : int, float, optional
        If the data should be zero padded also at the beginning.
        (default None)
    
    Returns
    -------
    signal_zeropadded : numpy.ndarray [size = (target_len,) or (target_len, num_channels)]
        The data zero padded.
        
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1,2,3,4])
    >>> # with 1 second delay (with sr of 4, that makes 4 sample delay)
    >>> x_zeropadded = zeropad_sound(x, target_len=10, sr=4, delay_sec=1)
    array([0., 0., 0., 0., 1., 2., 3., 4., 0., 0.])
    >>> # without delay
    >>> x_zeropadded = zeropad_sound(x, target_len=10, sr=4)
    array([1., 2., 3., 4., 0., 0., 0., 0., 0., 0.])
    >>> # if signal is longer than desired length:
    >>> x_zeropadded = zeropad_sound(x, target_len=3, sr=4)
    UserWarning: The signal cannot be zeropadded and will instead be truncated as length of `data` is 4 and `target_len` is 3.
  len(data), target_len))
    >>> x_zeropadded
    array([1, 2, 3])
    '''
    # ensure data follows shape of (num_samples,) or (num_samples, num_channels)
    data = pyst.utils.shape_samps_channels(data)
    num_channels = get_num_channels(data)
    if delay_sec is None:
        delay_sec = 0
    delay_samps = sr * delay_sec
    if target_len  < len(data) + delay_samps:
        import warnings
        # data must be truncated:
        remove_samples = len(data) - len(data[:target_len-delay_samps])
        if remove_samples >= len(data):
            warnings.warn('All data will be lost and replaced with zeros with the '+\
                'provided `target_len` and `delay_sec` settings. Data length is '+\
                    '{}, target_len is {}, and delay samples is {}.'.format(
                        len(data), target_len, delay_samps))
        data = data[:target_len-delay_samps]
        warnings.warn('The `target_len` is shorter than the `data` and `delay_sec`. '+\
            'Therefore the data will be cut off by {} sample(s).'.format(remove_samples))
    if len(data) < target_len:
        diff = target_len - len(data)
        signal_zeropadded = np.zeros((data.shape[0] + int(diff)))
        if num_channels > 1:
            signal_zeropadded = apply_num_channels(signal_zeropadded, num_channels)
            assert signal_zeropadded.shape[1] == data.shape[1]
        for i, row in enumerate(data):
            signal_zeropadded[i+delay_samps] += row
    else:
        import warnings
        warnings.warn('The signal cannot be zeropadded and will instead be truncated '+\
            'as length of `data` is {} and `target_len` is {}.'.format(
                len(data), target_len))
        signal_zeropadded = data[:target_len]
    return signal_zeropadded

def get_num_channels(data):
    if len(data.shape) > 1 and data.shape[1] > 1: 
        num_channels = data.shape[1]
    else:
        num_channels = 1
    return num_channels

# TODO clarify how length of output array is established
# TODO change time_delay_sec=1 to None default, as well as total_dur_sec
def combine_sounds(file1, file2, match2shortest=True, time_delay_sec=1,total_dur_sec=5):
    '''Combines sounds
    
    Parameters
    ----------
    file1 : str 
        One of two files to be added together
    file2 : str 
        Second of two files to be added together
    match2shortest : bool
        If the lengths of the addition should be limited by the shorter sound. 
        (defaul True)
    time_delay_sec : int, float, optional
        The amount of time in seconds before the sounds are added together. 
        The longer sound will play for this period of time before the shorter
        sound is added to it. (default 1)
    total_dur_sec : int, float, optional
        The total duration in seconds of the combined sounds. (default 5)
        
    Returns
    -------
    added_sound : numpy.ndarray
        The sound samples of the two soundfiles added together
    sr1 : int 
        The sample rate of the original signals and added sound
    '''
    data1, sr1 = pyst.dsp.loadsound(file1)
    data2, sr2 = pyst.dsp.loadsound(file2)
    if sr1 != sr2:
        data2, sr2 = resample_audio(data2, sr2, sr1)
    if time_delay_sec:
        num_extra_samples = int(sr1*time_delay_sec)
    else:
        num_extra_samples = 0
    if len(data1) > len(data2):
        data_long = data1
        data_short = data2
    else:
        data_long = data2
        data_short = data1
    dl_copy = data_long.copy()
    ds_copy = data_short.copy()
    
    if match2shortest:
        data_short = zeropad_sound(data_short, len(ds_copy) + num_extra_samples, sr1, delay_sec= time_delay_sec)
        data_long = data_long[:len(ds_copy)+num_extra_samples]
    else:
        data_short = zeropad_sound(data_short,len(dl_copy), sr1, delay_sec= time_delay_sec)
    added_sound = data_long + data_short
    
    if total_dur_sec:
        added_sound = added_sound[:sr1*total_dur_sec]
    return added_sound, sr1


def calc_frame_length(dur_frame_millisec, sr):
    """Calculates the number of samples necessary for each frame

    Parameters
    ----------
    dur_frame_millisec : int or float
        time in milliseconds each frame should be
    sr : int
        sampling rate of the samples to be framed

    Returns
    -------
    frame_length : int
        the number of samples necessary to fill a frame

    Examples
    --------
    >>> calc_frame_length(dur_frame_millisec=20, sr=1000)
    20
    >>> calc_frame_length(dur_frame_millisec=20, sr=48000)
    960
    >>> calc_frame_length(dur_frame_millisec=25.5, sr=22500)
    573
    """
    frame_length = int(dur_frame_millisec * sr // 1000)
    return frame_length

def calc_num_overlap_samples(samples_per_frame, percent_overlap):
    """Calculate the number of samples that constitute the overlap of frames

    Parameters
    ----------
    samples_per_frame : int 
        the number of samples in each window / frame
    percent_overlap : int, float
        either an integer between 0 and 100 or a decimal between 0.0 and 1.0
        indicating the amount of overlap of windows / frames

    Returns
    -------
    num_overlap_samples : int 
        the number of samples in the overlap

    Examples
    --------
    >>> calc_num_overlap_samples(samples_per_frame=100,percent_overlap=0.10)
    10
    >>> calc_num_overlap_samples(samples_per_frame=100,percent_overlap=10)
    10
    >>> calc_num_overlap_samples(samples_per_frame=960,percent_overlap=0.5)
    480
    >>> calc_num_overlap_samples(samples_per_frame=960,percent_overlap=75)
    720
    """
    if percent_overlap > 1:
        percent_overlap *= 0.01
    num_overlap_samples = int(samples_per_frame * percent_overlap)
    return num_overlap_samples

def calc_num_subframes(tot_samples, frame_length, overlap_samples, zeropad=False):
    """Assigns total frames needed to process entire noise or target series

    This function calculates the number of full frames that can be 
    created given the total number of samples, the number of samples in 
    each frame, and the number of overlapping samples.

    Parameters
    ----------
    tot_samples : int 
        total number of samples in the entire series
    frame_length : int 
        total number of samples in each frame / processing window
    overlap_samples : int
        number of samples in overlap between frames
    zeropad : bool, optional
        If False, number of subframes limited to full frames. If True, 
        number of subframes extended to zeropad the last partial frame.
        (default False)

    Returns
    -------
    None

    Examples
    --------
    >>> calc_num_subframes(30,10,5)
    5
    >>> calc_num_subframes(30,20,5)
    3
    """
    import math
    trim = frame_length - overlap_samples
    totsamps_adjusted = tot_samples-trim
    if zeropad:
        subframes = int(math.ceil(totsamps_adjusted / overlap_samples))
    else:
        subframes = int(totsamps_adjusted / overlap_samples)
    return subframes

def create_window(window_type, frame_length):
    """Creates window according to set window type and frame length

    the Hamming window tapers edges to around 0.08 while the Hann window
    tapers edges to 0.0. Both are commonly used in noise filtering.

    Parameters 
    ----------
    window_type : str
        type of window to be applied (default 'hamming')

    Returns
    -------
    window : ndarray
        a window fitted to the class attribute 'frame_length'

    Examples
    --------
    >>> #create Hamming window
    >>> hamm_win = create_window('hamming', frame_length=5)
    >>> hamm_win
    array([0.08, 0.54, 1.  , 0.54, 0.08])
    >>> #create Hann window
    >>> hann_win = create_window('hann',frame_length=5)
    >>> hann_win
    array([0. , 0.5, 1. , 0.5, 0. ])
    """
    if window_type.lower() == 'hamming':
        window = hamming(frame_length)
    elif 'hann' in window_type.lower():
        window = hann(frame_length)
    return window

def apply_window(samples, window, zeropad=False):
    """Applies predefined window to a section of samples

    The length of the samples must be the same length as the window. 

    Parameters
    ----------
    samples : ndarray
        series of samples with the length of input window
    window : ndarray
        window to be applied to the signal

    Returns
    -------
    samples_win : ndarray
        series with tapered sides according to the window provided

    Examples
    --------
    >>> import numpy as np
    >>> input_signal = np.array([ 0.        ,  0.36371897, -0.302721,
    ...                         -0.1117662 ,  0.3957433 ])
    >>> window_hamming = np.array([0.08, 0.54, 1.  , 0.54, 0.08])
    >>> apply_window(input_signal, window_hamming)
    array([ 0.        ,  0.19640824, -0.302721  , -0.06035375,  0.03165946])
    >>> window_hann = np.array([0. , 0.5, 1. , 0.5, 0. ])
    >>> apply_window(input_signal, window_hann)
    array([ 0.        ,  0.18185948, -0.302721  , -0.0558831 ,  0.        ])
    """
    if zeropad:
        if samples.shape != window.shape:
            temp_matrix = create_empty_matrix(
                window.shape)
            temp_matrix[:len(samples)] = samples
            samples = temp_matrix
    samples_win = samples * window
    return samples_win

def calc_fft(signal_section, real_signal=None, norm=False):
    """Calculates the fast Fourier transform of a 1D time series

    The length of the signal_section determines the number of frequency
    bins analyzed. Therefore, if there are higher frequencies in the 
    signal, the length of the `signal_section` should be long enough to 
    accommodate those frequencies. 

    The frequency bins with energy levels at around zero denote frequencies 
    not prevelant in the signal;the frequency bins with prevalent energy 
    levels relate to frequencies as well as their amplitudes that are in 
    the signal.


    Parameters
    ----------
    signal_section : ndarray
        the series that the fft will be applied to
    norm : bool
        whether or not normalization should be applied (default False)


    Returns
    -------
    fft_vals : ndarray (complex)
        the series transformed into the frequency domain with the same
        shape as the input series
    """
    if norm:
        norm = 'ortho'
    else:
        norm = None
    if real_signal:
        fft_vals = rfft(signal_section, norm=norm)
    else:
        fft_vals = fft(signal_section, norm=norm)
    return fft_vals

# TODO: https://github.com/biopython/biopython/issues/1496
# Fix numpy array repr for Doctest. 
def calc_power(fft_vals):
    '''Calculates the power of fft values

    Parameters
    ----------
    fft_vals : ndarray (complex or floats)
        the fft values of a windowed section of a series

    Returns
    -------
    power_spec : ndarray
        the squared absolute value of the input fft values

    Example
    -------
    >>> import numpy as np
    >>> matrix = np.array([[1,1,1],[2j,2j,2j],[-3,-3,-3]],
    ...                     dtype=np.complex_)
    >>> calc_power(matrix)
    array([[0.33333333, 0.33333333, 0.33333333],
           [1.33333333, 1.33333333, 1.33333333],
           [3.        , 3.        , 3.        ]])        
    '''
    power_spec = np.abs(fft_vals)**2 / len(fft_vals)
    return power_spec

def calc_average_power(matrix, num_iters):
    '''Divides matrix values by the number of times power values were added. 

    This function assumes the power values of n-number of series were 
    calculated and added. It divides the values in the input matrix by n, 
    i.e. 'num_iters'. 

    Parameters
    ----------
    matrix : ndarray
        a collection of floats or ints representing the sum of power 
        values across several series sets
    num_iters : int 
        an integer denoting the number of times power values were added to
        the input matrix

    Returns
    -------
    matrix : ndarray
        the averaged input matrix

    Examples
    --------
    >>> matrix = np.array([[6,6,6],[3,3,3],[1,1,1]])
    >>> ave_matrix = calc_average_power(matrix, 3)
    >>> ave_matrix
    array([[2.        , 2.        , 2.        ],
           [1.        , 1.        , 1.        ],
           [0.33333333, 0.33333333, 0.33333333]])
    '''
    if matrix.dtype == 'int64':
        matrix = matrix.astype('float')
    for i in range(len(matrix)):
        matrix[i] /= num_iters
    return matrix

def calc_phase(fft_matrix, radians=False):
    '''Calculates phase from complex fft values.
    
    Parameters
    ----------
    fft_vals : np.ndarray [shape=(d, t), dtype=complex]
        matrix with fft values
    radians : boolean
        False and complex values are returned. True and radians are returned.
        (Default False)
        
    Returns
    -------
    phase : np.ndarray [shape=(d, t)]
        Phase values for fft_vals. If radians is set to False, dtype = complex.
        If radians is set to True, dtype = float. 
        
    Examples
    --------
    >>> import numpy as np 
    >>> frame_length = 10
    >>> time = np.arange(0, 10, 0.1)
    >>> signal = np.sin(time)[:frame_length]
    >>> fft_vals = np.fft.fft(signal)
    >>> phase = calc_phase(fft_vals, radians=False)
    >>> phase[:2]
    array([ 1.        +0.j        , -0.37872566+0.92550898j])
    >>> phase = calc_phase(fft_vals, radians=True)
    >>> phase[:2]
    array([0.        , 1.95921533])
    '''
    if not radians:
        __, phase = librosa.magphase(fft_matrix)
    else:
        # in radians 
        #if normalization:
            #phase = np.angle(fft_matrix) / (frame_length * norm_win)
        #else:
        phase = np.angle(fft_matrix)
    return phase

def reconstruct_whole_spectrum(band_reduced_noise_matrix, n_fft=None):
    '''Reconstruct whole spectrum by mirroring complex conjugate of data.
    
    Parameters
    ----------
    band_reduced_noise_matrix : np.ndarray [size=(n_fft,), dtype=np.float or np.complex_]
        Matrix with either power or fft values of the left part of the fft. The whole
        fft can be provided; however the right values will be overwritten by a mirrored
        left side.
    n_fft : int, optional
        If None, `n_fft` set to length of `band_reduced_noise_matrix`. `n_fft` defines
        the size of the mirrored vector.
        
    Returns
    -------
    output_matrix : np.ndarray [size = (n_fft,), dtype=np.float or np.complex_]
        Mirrored vector of input data.
        
    Examples
    --------
    >>> x = np.array([3.,2.,1.,0.])
    >>> # double the size of x
    >>> x_rec = pyst.dsp.reconstruct_whole_spectrum(x, n_fft=int(len(x)*2))
    >>> x_rec
    array([3., 2., 1., 0., 0., 1., 2., 3.])
    >>> # overwrite right side of data
    >>> x = np.array([3.,2.,1.,0.,0.,2.,3.,5.])
    >>> x_rec = pyst.dsp.reconstruct_whole_spectrum(x, n_fft=len(x))
    >>> x_rec
    array([3., 2., 1., 0., 0., 1., 2., 3.])
    '''
    # expects 1d data
    if len(band_reduced_noise_matrix.shape) > 1:
        band_reduced_noise_matrix = band_reduced_noise_matrix.reshape((
            band_reduced_noise_matrix.shape[0]))
    if n_fft is None:
        n_fft = len(band_reduced_noise_matrix)
    if isinstance(band_reduced_noise_matrix[0], np.complex):
        complex_vals = True
    else:
        complex_vals = False
    total_rows = n_fft
    output_matrix = create_empty_matrix((total_rows,), complex_vals=complex_vals)
    if band_reduced_noise_matrix.shape[0] < n_fft:
        temp_matrix = create_empty_matrix((total_rows,), complex_vals=complex_vals)
        temp_matrix[:len(band_reduced_noise_matrix)] += band_reduced_noise_matrix
        band_reduced_noise_matrix = temp_matrix
    # flip up-down
    flipped_matrix = np.flip(band_reduced_noise_matrix, axis=0)
    output_matrix[0:n_fft//2+1,] += band_reduced_noise_matrix[0:n_fft//2+1]
    output_matrix[n_fft//2+1:,] += flipped_matrix[n_fft//2+1:]
    assert output_matrix.shape == (n_fft,)
    return output_matrix

# TODO
def vad():
    '''voice activity detection
    
    Determines whether speech exists or not in the signal
    '''
    pass

# TODO
def snr():
    '''measures the sound to noise ratio in signal
    '''
    pass


def apply_original_phase(spectrum, phase):
    '''Multiplies phase to power spectrum
    
    Parameters
    ----------
    spectrum : np.ndarray [shape=(n,), dtype=np.float or np.complex]
        Magnitude or power spectrum
    phase : np.ndarray [shape=(n,), dtype=np.float or np.complex]
        Phase to be applied to spectrum
    '''
    # ensure 1d dimensions
    if len(spectrum.shape) > 1:
        spectrum = spectrum.reshape((
            spectrum.shape[0],))
    if len(phase.shape) > 1:
        phase = phase.reshape((
            phase.shape[0],))
    assert spectrum.shape == phase.shape
    # Whether or not phase is represented in radians or a spectrum.
    if isinstance(phase[0], np.complex):
        radians = False
    else:
        radians = True
    if not radians:
        spectrum_complex = spectrum * phase
    else:
        import cmath
        phase_prepped = (1/2) * np.cos(phase) + cmath.sqrt(-1) * np.sin(phase)
        spectrum_complex = spectrum**(1/2) * phase_prepped
    
    return spectrum_complex

def calc_posteri_snr(target_power_spec, noise_power_spec):
    """Calculates and updates signal to noise ratio of current frame

    Parameters
    ----------
    target_power_spec : ndarray
        matrix of shape with power values of target 
        signal
    noise_power_spec : ndarray
        matrix of shape with power values of noise
        signal

    Returns 
    -------
    posteri_snr : ndarray
        matrix containing the signal to noise ratio 

    Examples
    --------
    >>> sig_power = np.array([6,6,6,6])
    >>> noise_power = np.array([2,2,2,2])
    >>> calc_posteri_snr(sig_power, noise_power)
    array([3., 3., 3., 3.])
    """
    posteri_snr = np.zeros(target_power_spec.shape)
    for i in range(len(target_power_spec)):
        posteri_snr[i] += target_power_spec[i] / noise_power_spec[i]
    return posteri_snr

def calc_posteri_prime(posteri_snr):
    """Calculates the posteri prime 

    Parameters
    ----------
    posteri_snr : ndarray
        The signal-to-noise ratio of the noisey signal, frame by frame.

    Returns 
    -------
    posteri_prime : ndarray
        The primed posteri_snr, calculated according to the reference paper.

    References
    ----------
    Scalart, P. and Filho, J. (1996). Speech enhancement based on a priori 
    signal to noise estimation. Proc. IEEE Int. Conf. Acoust., Speech, Signal
    Processing, 629-632.
    """
    posteri_prime = posteri_snr - 1
    posteri_prime[posteri_prime < 0] = 0
    return posteri_prime

def calc_prior_snr(snr, snr_prime, smooth_factor=0.98, first_iter=None, gain=None):
    """Estimates the signal-to-noise ratio of the previous frame

    Depending on the `first_iter` argument, the prior snr is calculated 
    according to different algorithms. If `first_iter` is None, prior snr is 
    calculated according to Scalart and Filho (1996); if `first_iter` 
    is True or False, snr prior is calculated according to Loizou (2013).

    Parameters
    ----------
    snr : ndarray
        The sound-to-noise ratio of target vs noise power/energy levels.
    snr_prime : ndarray
        The prime of the snr (see Scalart & Filho (1996))
    smooth_factor : float
        The value applied to smooth the signal. (default 0.98)
    first_iter : None, True, False
        If None, snr prior values are estimated the same, no matter if it is
        the first iteration or not (Scalart & Filho (1996))
        If True, snr prior values are estimated without gain (Loizou 2013)
        If False, snr prior values are enstimed with gain (Loizou 2013) 
        (default None)
    gain : None, ndarray
        If None, gain will not be used. If gain, it is a previously calculated
        value from the previous frame. (default None)

    Returns
    -------
    prior_snr : ndarray
        Estimation of signal-to-noise ratio of the previous frame of target signal.

    References
    ----------
    C Loizou, P. (2013). Speech Enhancement: Theory and Practice. 
    
    Scalart, P. and Filho, J. (1996). Speech enhancement based on a 
    priori signal to noise estimation. Proc. IEEE Int. Conf. Acoust., 
    Speech, Signal Processing, 629-632.
    """
    if first_iter is None:
        # calculate according to apriori SNR equation (6) in paper
        # Scalart, P. and Filho, J. (1996)
        first_arg = (1 - smooth_factor) * snr_prime
        second_arg = smooth_factor * snr
        prior_snr = first_arg + second_arg
    elif first_iter is True:
        # calculate according to Loizou (2013)
        # don't yet have previous gain or snr values to apply
        first_arg = smooth_factor
        second_arg = (1-smooth_factor) * snr_prime
        prior_snr = first_arg + second_arg
    elif first_iter is False:
        # now have previous gain and snr values
        first_arg = smooth_factor * (gain**2) * snr
        second_arg = (1 - smooth_factor) * snr_prime
        prior_snr = first_arg + second_arg
    return prior_snr

def calc_gain(prior_snr):
    '''Calculates the gain (i.e. attenuation) values to reduce noise.

    Parameters
    ----------
    prior_snr : ndarray
        The prior signal-to-noise ratio estimation

    Returns
    -------
    gain : ndarray
        An array of attenuation values to be applied to the signal (stft) array
        at the current frame.
        
    References
    ----------
    C Loizou, P. (2013). Speech Enhancement: Theory and Practice. 
    
    Scalart, P. and Filho, J. (1996). Speech enhancement based on a 
    priori signal to noise estimation. Proc. IEEE Int. Conf. Acoust., 
    Speech, Signal Processing, 629-632.
    '''
    gain = np.sqrt(prior_snr/(1+prior_snr))
    return gain

def apply_gain_fft(fft_vals, gain):
    '''Reduces noise by applying gain values to the stft / fft array of the 
    target signal

    Parameters
    ----------
    fft_vals : ndarray(complex)
        Matrix containing complex values (i.e. stft values) of target signal
    gain : ndarray(real)
        Matrix containing calculated attenuation values to apply to 'fft_vals'

    Returns
    -------
    enhanced_fft : ndarray(complex)
        Matrix with attenuated noise in target (stft) values
    '''
    enhanced_fft = fft_vals * gain
    assert enhanced_fft.shape == fft_vals.shape
    return enhanced_fft

def calc_ifft(signal_section, real_signal=None, norm=False):
    """Calculates the inverse fft of a series of fft values

    The real values of the ifft can be used to be saved as an audiofile

    Parameters
    ----------
    signal_section : ndarray [shape=(num_freq_bins,) 
        The frame of fft values to apply the inverse fft to
    num_fft : int, optional
        The number of total fft values applied when calculating the original fft. 
        If not given, length of `signal_section` is used. 
    norm : bool
        Whether or not the ifft should apply 'ortho' normalization
        (default False)

    Returns 
    -------
    ifft_vals : ndarray(complex)
        The inverse Fourier transform of filtered audio data
    """
    if norm:
        norm = 'ortho'
    else:
        norm = None
    if real_signal:
        ifft_vals = irfft(signal_section, norm=norm)
    else:
        ifft_vals = ifft(signal_section, norm=norm)
    return ifft_vals

def control_volume(samples, max_limit):
    """Keeps max volume of samples to within a specified range.

    Parameters
    ----------
    samples : ndarray
        series of audio samples
    max_limit: float
        maximum boundary of the maximum value of the audio samples

    Returns
    -------
        samples with volume adjusted (if need be).

    Examples
    --------
    >>> import numpy as np
    >>> #low volume example: increase volume to desired window
    >>> x = np.array([-0.03, 0.04, -0.05, 0.02])
    >>> x = control_volume(x, max_limit=0.25)
    >>> x
    array([-0.13888889,  0.25      , -0.25      ,  0.13888889])
    >>> #high volume example: decrease volume to desired window
    >>> y = np.array([-0.3, 0.4, -0.5, 0.2])
    >>> y = control_volume(y, max_limit=0.15)
    >>> y
    array([-0.08333333,  0.15      , -0.15      ,  0.08333333])
    """
    if max(samples) != max_limit:
        samples = np.interp(samples,
                            (samples.min(), samples.max()),
                            (-max_limit, max_limit))
    return samples

######### Functions related to postfilter###############

def calc_power_ratio(original_powerspec, noisereduced_powerspec):
    '''Calc. the ratio of original vs noise reduced power spectrum.
    '''
    power_ratio = sum(noisereduced_powerspec) / \
        sum(original_powerspec)/len(noisereduced_powerspec)
    return power_ratio

def calc_noise_frame_len(SNR_decision, threshold, scale):
    '''Calc. window length for calculating moving average. 
    
    Note: lower SNRs require larger window.
    '''
    if SNR_decision < 1:
        soft_decision = 1 - (SNR_decision/threshold)
        soft_decision_scaled = round((soft_decision) * scale)
        noise_frame_len = 2 * soft_decision_scaled + 1
    else:
        noise_frame_len = SNR_decision
    return noise_frame_len

def calc_linear_impulse(noise_frame_len, num_freq_bins):
    '''Calc. the post filter coefficients to be applied to gain values.
    '''
    linear_filter_impulse = np.zeros((num_freq_bins,))
    for i in range(num_freq_bins):
        if i < noise_frame_len:
            linear_filter_impulse[i] = 1 / noise_frame_len
        else:
            linear_filter_impulse[i] = 0
    return linear_filter_impulse

def adjust_volume(samples, vol_range):
    samps = samples.copy()
    adjusted_volume = np.interp(samps,
                                (samps.min(), samps.max()),
                                (-vol_range, vol_range))
    return adjusted_volume

def spread_volumes(samples, vol_list = [0.1,0.3,0.5]):
    '''Returns samples with a range of volumes. 
    
    This may be useful in applying to training data (transforming data).
    
    Parameters
    ----------
    samples : ndarray
        Series belonging to acoustic signal.
    vol_list : list 
        List of floats or ints representing the volumes the samples
        are to be oriented towards. (default [0.1,0.3,0.5])
        
    Returns
    -------
    volrange_dict : tuple 
        Tuple of `volrange_dict` values containing `samples` at various vols.
    '''
    if samples is None or len(samples) == 0:
        raise ValueError('No Audio sample data recognized.')
    max_vol = max(samples)
    if round(max_vol) > 1:
        raise ValueError('Audio data not normalized.')
    volrange_dict = {}
    for i, vol in enumerate(vol_list):
        volrange_dict[i] = adjust_volume(samples, vol) 
    return tuple(volrange_dict.values())

def create_empty_matrix(shape, complex_vals=False):
    '''Allows creation of a matrix filled with real or complex zeros.

    In digital signal processing, complex numbers are common; it is 
    important to note that if complex_vals=False and complex values are
    inserted into the matrix, the imaginary part will be removed.

    Parameters
    ----------
    shape : tuple or int
        tuple or int indicating the shape or length of desired matrix or
        vector, respectively
    complex_vals : bool
        indicator of whether or not the matrix will receive real or complex
        values (default False)

    Returns
    ----------
    matrix : ndarray
        a matrix filled with real or complex zeros

    Examples
    ----------
    >>> matrix = create_empty_matrix((3,4))
    >>> matrix
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])
    >>> matrix_complex = create_empty_matrix((3,4),complex_vals=True)
    >>> matrix_complex
    array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
    >>> vector = create_empty_matrix(5,)
    >>> vector
    array([0., 0., 0., 0., 0.])
    '''
    if complex_vals:
        matrix = np.zeros(shape, dtype=np.complex_)
    else:
        matrix = np.zeros(shape, dtype=float)
    return matrix

# TODO test this in applications (currently not implemented)
def overlap_add(enhanced_matrix, frame_length, overlap, complex_vals=False):
    '''Overlaps and adds windowed sections together to form 1D signal.
    
    Parameters
    ----------
    enhanced_matrix : np.ndarray [shape=(frame_length, num_frames), dtype=float]
        Matrix with enhance values
    frame_length : int 
        Number of samples per frame 
    overlap : int 
        Number of samples that overlap
        
    Returns
    -------
    new_signal : np.ndarray [shape=(frame_length,), dtype=float]
        Length equals (frame_length - overlap) * enhanced_matrix.shape[1] + overlap
        
    Examples
    --------
    >>> import numpy as np
    >>> enhanced_matrix = np.ones((4, 4))
    >>> frame_length = 4
    >>> overlap = 1
    >>> sig = overlap_add(enhanced_matrix, frame_length, overlap)
    >>> sig
    [1. 1. 1. 2. 1. 1. 2. 1. 1. 2. 1. 1. 1.]
    '''
    try:
        assert enhanced_matrix.shape[0] == frame_length
    except AssertionError as e:
        raise TypeError('The first dimension of the enhance matrix should '+ \
            'match the frame length. {} does not match frame length {}'.format(
                enhanced_matrix.shape[0], frame_length))
    increments = frame_length - overlap
    start= increments
    mid= start + overlap
    stop= start + frame_length
    
    expected_len = (frame_length - overlap) * enhanced_matrix.shape[1] + overlap
    new_signal = create_empty_matrix(
        (expected_len,),
        complex_vals=complex_vals)
    
    for i in range(enhanced_matrix.shape[1]):
        if i == 0:
            new_signal[:frame_length] += enhanced_matrix[:frame_length,i]
        else:
            new_signal[start:mid] += enhanced_matrix[:overlap,i] 
            new_signal[mid:stop] += enhanced_matrix[overlap:frame_length,i]
            start += increments
            mid = start+overlap
            stop = start+frame_length
    return new_signal


if __name__ == '__main__':
    import doctest
    doctest.testmod()
