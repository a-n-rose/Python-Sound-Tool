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
from python_speech_features import logfbank, mfcc
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
    data = data.astype(samples.dtype)
    return data

def scalesound(data,min_val=-1,max_val=1):
    '''Scales the input array to range between `min_val` and `max_val`. 
    
    Parameters
    ----------
    data : np.ndarray [size = (num_samples,)]
        Original samples
    Returns
    -------
    samples : np.ndarray [size = (num_samples,)]
        Copy of original data, scaled to the min and max values.
    '''
    samples = data.copy()
    samples = np.interp(samples,(samples.min(), samples.max()),(min_val, max_val))
    return samples

def normalize(data, max_val=None, min_val=None):
    '''Normalizes data.
    
    This is usefule if you have predetermined max and min values you want to normalize
    new data with.
    
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
    '''
    if max_val is None:
         normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        normed_data = (data - min_val) / (max_val - min_val)
    return normed_data

def resample_audio(samples, sr_original, sr_desired):
    '''Allows audio samples to be resampled to desired sample rate.
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
    >>> data_mono = stereo2mono(data)
    >>> data_mono[:5]
    array([0.        , 0.81632653, 1.63265306, 2.44897959, 3.26530612])
    '''
    data_mono = data.copy()
    if len(data.shape) > 1 and data.shape[1] > 1:
        data_mono = np.take(data,0,axis=-1) 
    return data_mono

def combine_sounds(signal, sound, scale=1, delay_target_sec = 1, total_len_sec=None):
    '''Adds a sound (i.e. background noise) to a target signal 
    
    Parameters
    ----------
    signal : str 
        Sound file of the target sound
    sound : str 
        Sound file of the background noise or sound
    scale : int or float, optional
        The loudness of the sound to be added (default 1)
    delay_target_sec : int or float, optional
        Length of time in seconds the sound will be played before the target
        (default 1)
    total_len_sec : int or float, optional
        Total length of combined sound in seconds. If none, the sound will end
        after target sound ends (default None)
    
    Returns
    -------
    combined : numpy.ndarray
        The samples of the sounds added together
    sr : int 
        The sample rate of the samples
    '''
    target, sr = pyst.dsp.loadsound(signal)
    sound2add, sr2 = pyst.dsp.loadsound(sound)
    if sr != sr2:
        sound2add, sr = resample_audio(sound2add, sr2, sr)
    sound2add *= scale
    if total_len_sec:
        total_samps = int(sr*total_len_sec)
    else:
        total_samps = int(sr*(len(target)+delay_target_sec))
    if len(sound2add) < total_samps:
        sound2add = extend_sound(sound2add, total_samps)
    beginning_delay = sound2add[:int(sr*delay_target_sec)]
    noise2mix = sound2add[int(sr*delay_target_sec):len(target)+int(sr*delay_target_sec)]
    ending = sound2add[len(target)+int(sr*delay_target_sec):total_samps]
    combined = noise2mix + target
    if delay_target_sec:
        combined = np.concatenate((beginning_delay,combined))
    if total_len_sec:
        combined = np.concatenate((combined, ending))
    return combined, sr

def extend_sound(data, target_len):
    '''Extends a sound by repeating it until its `target_len`
    
    This is perhaps useful when working with repetitive or
    stationary sounds.
    '''
    new_data = np.zeros((target_len,))
    row_id = 0
    while row_id < len(new_data):
        if row_id + len(data) > len(new_data):
            diff = row_id + len(data) - len(new_data)
            new_data[row_id:] += data[:-diff]
        else:
            new_data[row_id:row_id+len(data)] += data
            row_id += len(data)
    return new_data


def zeropad_sound(data, target_len, sr, delay_sec=1):
    '''If the sound data needs to be a certain length, zero pad it.
    
    Parameters
    ----------
    data : numpy.ndarray
        The sound data that needs zero padding. Shape (len(data),). 
        Expects mono channel.
    target_len : int 
        The number of samples the `data` should have
    sr : int
        The samplerate of the `data`
    delay_sec : int, float, optional
        If the data should be zero padded also at the beginning.
        (default 1)
    
    Returns
    -------
    signal_zeropadded : numpy.ndarray
        The data zero padded to the shape (target_len,)
    '''
    if len(data.shape) > 1 and data.shape[1] > 1: 
        data = stereo2mono(data)
    delay_samps = sr * delay_sec
    if len(data) < target_len:
        diff = target_len - len(data)
        signal_zeropadded = np.zeros((data.shape[0] + int(diff)))
        for i, row in enumerate(data):
            signal_zeropadded[i+delay_samps] += row
    return signal_zeropadded

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

def calc_num_subframes(tot_samples, frame_length, overlap_samples):
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
    trim = frame_length - overlap_samples
    totsamps_adjusted = tot_samples-trim
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

def apply_window(samples, window):
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

def collect_features(samples, feature_type='mfcc', sr=48000, window_size_ms=20,
                     window_shift_ms=10, num_filters=40, num_mfcc=40,
                     window_function=None):
    '''Collects fbank and mfcc features.
    '''
    if not window_function:
        # default for python_speech_features:
        def window_function(x): return np.ones((x,))
    else:
        if 'hamming' in window_function:
            window_function = hamming
        elif 'hann' in window_function:
            window_function = hann
        else:
            # default for python_speech_features:
            def window_function(x): return np.ones((x,))
    if len(samples)/sr*1000 < window_size_ms:
        window_size_ms = len(samples)/sr*1000
    frame_length = calc_frame_length(window_size_ms, sr)
    if 'fbank' in feature_type:
        feats = logfbank(samples,
                         sr=sr,
                         winlen=window_size_ms * 0.001,
                         winstep=window_shift_ms * 0.001,
                         nfilt=num_filters,
                         nfft=frame_length)
    elif 'mfcc' in feature_type:
        feats = mfcc(samples,
                     sr=sr,
                     winlen=window_size_ms * 0.001,
                     winstep=window_shift_ms * 0.001,
                     nfilt=num_filters,
                     numcep=num_mfcc,
                     nfft=frame_length)
    return feats, frame_length, window_size_ms

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
