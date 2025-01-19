import math
import numpy as np
from numpy.fft import fft, rfft, ifft, irfft

from .org import SoundObj, VADhandler,FFTsettings, VADsettings

class VAD:
    def __init__(
            self,
            origin:SoundObj,
            settings:VADhandler=None,
    ):
        if settings is None:
            # Ensure default setting num fft_bins documented
            settings=VADhandler(
                fft_settings=FFTsettings(fft_bins=int(origin.sr)//2),
                vad_settings=VADsettings(),
            )
        self.origin=origin
        self.settings=settings
        self.frames_vad, self.frames_idx = vad(
            data=self.origin.data,
            sr=self.origin.sr,
            **self.settings.fft_settings,
            **self.settings.vad_settings,
        )
        self.startidx=np.where(self.frames_vad == 1)[0][0]
        self.endidx=np.where(self.frames_vad == 1)[0][-1]


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


# works for stereo sound (raw signal data)
def scalesound(data, max_val = 1, min_val=None):
    '''Scales the input array to range between `min_val` and `max_val`. 
    
    Parameters
    ----------
    data : np.ndarray [size = (num_samples,) or (num_samples, num_channels)]
        Original samples
    
    max_val : int, float
        The maximum value the dataset is to range from (default 1)
    
    min_val : int, float, optional
        The minimum value the dataset is to range from. If set to None,
        will be set to the opposiite of `max_val`. E.g. if `max_val` is set to
        0.8, `min_val` will be set to -0.8. (default None)
    
    
    Returns
    -------
    samples : np.ndarray [size = (num_samples,) or (num_samples, num_channels)]
        Copy of original data, scaled to the min and max values.
    
    
    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> input_samples = np.random.random_sample((5,))
    >>> input_samples
    array([0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ])
    >>> input_samples.max()
    0.7151893663724195
    >>> input_samples.min()
    0.4236547993389047
    >>> # default setting: between -1 and 1
    >>> output_samples = scalesound(input_samples)
    >>> output_samples 
    array([-0.14138 ,1., 0.22872961, -0.16834299, -1.])
    >>> output_samples.max()
    1.0
    >>> output_samples.min()
    -1.0
    >>> # range between -100 and 100
    >>> output_samples = scalesound(input_samples, max_val = 100, min_val = -100)
    >>> output_samples
    array([ -14.13800026,100., 22.87296052,-16.83429866,-100.])
    >>> output_samples.max()
    100.0
    >>> output_samples.min()
    -100.0
    '''
    if min_val is None:
        min_val = -max_val
    samples = data.copy()
    if isinstance(samples, np.ndarray):
        samples = np.interp(samples,
                            (samples.min(), samples.max()),
                            (min_val, max_val))
    else:
        samples = np.interp(samples,
                            (min(samples), max(samples)),
                            (min_val, max_val))
    return samples


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
    subframes : int 
        The number of subframes necessary to fully process the audio samples
        at given `frame_length`, `overlap_samples`, and `zeropad`.


    Examples
    --------
    >>> calc_num_subframes(30,10,5)
    5
    >>> calc_num_subframes(30,20,5)
    3
    """
    import math
    if overlap_samples == 0:
        if zeropad:
            subframes = int(math.ceil(tot_samples/frame_length))
        else:
            subframes = int(tot_samples/frame_length)
        return subframes
    trim = frame_length - overlap_samples
    totsamps_adjusted = tot_samples-trim
    if zeropad:
        subframes = int(math.ceil(totsamps_adjusted / overlap_samples))
    else:
        subframes = int(totsamps_adjusted / overlap_samples)
    return subframes


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


def create_window(window_type, frame_length):
    """Creates window according to set window type and frame length

    the Hamming window tapers edges to around 0.08 while the Hann window
    tapers edges to 0.0. Both are commonly used in noise filtering.


    Parameters 
    ----------
    window_type : str
        type of window to be applied (default 'hamming'). Options:
        bartlett, blackman, hamming, hanning, kaiser(beta).


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
        window = np.hamming(frame_length)
    elif 'hann' in window_type.lower():
        window = np.hanning(frame_length)
    elif 'bartlett' in window_type.lower():
        window = np.bartlett(frame_length)
    elif 'blackman' in window_type.lower():
        window = np.blackman(frame_length)
    elif 'kaiser' in window_type.lower():
        window = np.kaiser(frame_length)
    return window


def add_channels(samples, channels_total):
    '''Copies columns of `samples` to create additional channels.
    
    Parameters
    ----------
    samples : np.ndarray [shape=(num_samples) or (num_samples,num_channels)]
        The samples to add channels to.
    
    channels_total : int 
        The total number of channels desired. For example, if `samples` already has
        2 channels and you want it to have 3, set `channels_total` to 3.
        
    Returns
    -------
    x : np.ndarray [shape = (num_samples, channels_total)]
        A copy of `samples` with desired number of channels.
        
    Examples
    --------
    >>> import numpy as np
    >>> samps_mono = np.array([1,2,3,4,5])
    >>> samps_stereo2 = add_channels(samps_mono, 2)
    >>> samps_stereo2
    array([[1, 1],
    ...    [2, 2],
    ...    [3, 3],
    ...    [4, 4],
    ...    [5, 5]])
    >>> samps_stereo5 = add_channels(samps_stereo2, 5)
    >>> samps_stereo5
    array([[1, 1, 1, 1, 1],
    ...    [2, 2, 2, 2, 2],
    ...    [3, 3, 3, 3, 3],
    ...    [4, 4, 4, 4, 4],
    ...    [5, 5, 5, 5, 5]])
    
    Warning
    -------
    If `channels_total` is less than or equal to the number of channels already presesnt
    in `samples`. No channels added in those cases.
    '''
    y = samples.copy()
    if len(y.shape) == 1:
        for i in range(channels_total):
            if i == 0:
                x = np.vstack((y, y))
            elif i < channels_total-1:
                x = np.vstack((x, y)) 
    else:
        if y.shape[1] == channels_total:
            import warnings
            msg = '\n\nWarning: provided `samples` already has {} channels.'.format(channels_total)+\
                '\nTo add 1 channel to samples shaped (5, 1), set `channels_total` to 2.' +\
                    '\nNo channels added.\n'
            warnings.warn(msg)
            return y
        elif y.shape[1] > channels_total:
            import warnings
            msg = '\n\nWarning: function add_channels adds channels. '+\
                '`samples` currently has {} channels and provided`'.format(samples.shape[1]) +\
                    ' `channels_total` is less than that: {}'.format(channels_total) +\
                        '\nTo add 1 channel to samples shaped (5, 2), set `channels_total` to 3.'+\
                            '\nNo channels added.\n'
            warnings.warn(msg)
        x = y.T
        extra_channels = channels_total//samples.shape[1]
        for i in range(extra_channels):
            x = np.vstack((x, y.T))
    x = x.T 
    if x.shape[1] > channels_total:
        x = x[:,:channels_total]
    return x


# TODO: remove nested function
def apply_window(samples, window, zeropad=False):
    """Applies predefined window to a section of samples. Mono or stereo sound checked.

    The length of the samples must be the same length as the window. 

    Parameters
    ----------
    samples : ndarray [shape=(num_samples,) or (num_samples, num_channels)]
        series of samples with the length of input window
    
    window : ndarray [shape=(num_samples,) or (num_samples, num_channels)]
        window to be applied to the  If window does not match number of 
        channels of sample data, the missing channels will be applied to the window,
        repeating the first channel.


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
    if len(samples.shape) == 1:
        if len(window.shape) > 1:
            window = window[:,0]
    elif len(samples.shape) != len(window.shape) or samples.shape[1] != window.shape[1]:
        window = add_channels(window, samples.shape[1])
        # in the off chance the window has more channels than samples:
        if window.shape[1] > samples.shape[1]:
            window = window[:,:samples.shape[1]]
    if zeropad:
        if samples.shape != window.shape:
            temp_matrix = create_empty_matrix(
                window.shape)
            temp_matrix[:len(samples)] = samples
            samples = temp_matrix
    samples_win = samples * window
    return samples_win

def shape_samps_channels(data):
    '''Returns data in shape (num_samps, num_channels)
    
    Parameters
    ----------
    data : np.ndarray [size= (num_samples,) or (num_samples, num_channels), or 
        (num_channels, num_samples)] The data that needs to be checked for 
        correct format 
    
    Returns
    -------
    data : np.ndarray [size = (num_samples,) or (num_samples, num_channels)]
    '''
    if len(data.shape) == 1:
        return data
    if len(data.shape) > 2:
        raise ValueError('Expected 2 dimensional data: (num_samples, num_channels,) not '+\
            'shape {}'.format(data.shape))
    if data.shape[0] < data.shape[1]:
        # assumes number of samples will be greater than number of channels
        data = data.T
    assert data.shape[0] > data.shape[1] 
    return data

# TODO remove nested function
def ismono(data):
    # ensure channels last
    data = shape_samps_channels(data) 
    if len(data.shape) > 1 and data.shape[1] > 1:
        return False
    else:
        return True

def calc_fft(signal_section, real_signal=None, fft_bins = None, **kwargs):
    """Calculates the fast Fourier transform of a time series. Should work with stereo signals.

    The length of the signal_section determines the number of frequency
    bins analyzed if `fft_bins` not set. Therefore, if there are higher frequencies in the 
    signal, the length of the `signal_section` should be long enough to 
    accommodate those frequencies. 

    The frequency bins with energy levels at around zero denote frequencies 
    not prevelant in the signal;the frequency bins with prevalent energy 
    levels relate to frequencies as well as their amplitudes that are in 
    the signal.


    Parameters
    ----------
    signal_section : ndarray [shape = (num_samples) or (num_samples, num_channels)]
        the series that the fft will be applied to. If stereo sound, will return a FFT 
        for each channel.
        
    real_signal : bool 
        If True, only half of the fft will be returned (the fft is mirrored). Otherwise the 
        full fft will be returned.
        
    kwargs : additional keyword arguments
        keyword arguments for numpy.fft.fft or nump.fft.rfft

    Returns
    -------
    fft_vals : ndarray [shape=(num_fft_bins), or (num_fft_bins, num_channels), dtype=np.complex_]
        the series transformed into the frequency domain with the same
        shape as the input series
    """
    if ismono(signal_section):
        if real_signal:
            fft_vals = rfft(signal_section, n = fft_bins, **kwargs)
        else:
            fft_vals = fft(signal_section, n = fft_bins, **kwargs)
    else:
        for channel in range(signal_section.shape[1]):
            if channel == 0:
                if real_signal:
                    fft_vals = rfft(signal_section[:,channel], n = fft_bins, **kwargs)
                else:
                    fft_vals = fft(signal_section[:,channel], n = fft_bins, **kwargs)
                x = fft_vals
            else:
                if real_signal:
                    fft_vals = rfft(signal_section[:,channel], n = fft_bins, **kwargs)
                else:
                    fft_vals = fft(signal_section[:,channel], n = fft_bins, **kwargs)
                x = np.stack((x, fft_vals), axis=-1)
        fft_vals = x
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


def short_term_energy(signal_windowed):
    '''
    Expects `signal` to be scaled (-1, 1) as well as windowed.
    
    References
    ----------
    http://vlab.amrita.edu/?sub=3&brch=164&sim=857&cnt=1
    '''
    ste = sum(np.abs(signal_windowed)**2)
    return ste


def get_dom_freq(power_values):
    '''If real_signal (i.e. half fft bins), might mess up values.
    '''
    freq_bin = np.argmax(power_values)
    return freq_bin

# TODO test
def spectral_flatness_measure(spectrum):
    import scipy
    spectrum = np.abs(spectrum)
    g = scipy.stats.gmean(spectrum)
    a = np.mean(spectrum)
    if not np.isfinite(a).all():
        raise TypeError(a)
    elif not np.isfinite(a).any():
        raise ValueError(a)
    sfm = 10 * np.log10(g/(a+1e-6))
    return sfm


# TODO not allow 'None' defaults
# TODO Doesn't catch end of VAD well; loses speech info at end, seemingly
def vad(data: np.ndarray, sr:int, win_size_ms: int = 50, percent_overlap:float = 0, 
        real_signal:bool = False, fft_bins:int = None, window:str = 'hann',
        energy_thresh:int = 40, freq_thresh:int = 185, sfm_thresh:int = 5,
        min_energy:int = None, min_freq:int = None, min_sfm = None, use_beg_ms:int = 120):
    '''
    Warning: this VAD works best with sample rates at or above 44100 Hz.
    
    Parameters
    ----------
    energy_thresh : int, float
        The minimum amount of energy for speech detection.
    
    freq_thresh : int, float 
        The maximum frequency threshold.
    
    sfm_thresh : int, float 
        The spectral flatness measure threshold. 
    
    References
    ----------
    M. H. Moattar and M. M. Homayounpour, "A simple but efficient real-time Voice 
    Activity Detection algorithm," 2009 17th European Signal Processing Conference, 
    Glasgow, 2009, pp. 2549-2553.
    '''
    # if real signal, divide freq_thresh in half, as only half freq_bins are used
    if real_signal:
        freq_thresh /= 2 
    if fft_bins is None: 
        fft_bins = int(sr // 2)
    if fft_bins % 2 != 0:
        fft_bins += 1
    max_freq = int(sr//2)
    # ensure even
    if not max_freq % 2 == 0:
        max_freq += 1
    if sr < 44100:
        import warnings
        msg = '\nWarning: VAD works best with sample rates at or above 44100 Hz.'
        warnings.warn(msg)
    # first scale samples to be between -1 and 1
    data = scalesound(data, max_val = 1)
    frame_length = calc_frame_length(win_size_ms, sr)
    num_overlap_samples = int(frame_length * percent_overlap)
    num_subframes = calc_num_subframes(
        len(data),
        frame_length = frame_length,
        overlap_samples = num_overlap_samples,
        zeropad = True,
    )
    
    # limit number of beginning frames to measure background noise:
    measure_noise_samples = calc_frame_length(use_beg_ms, sr)
    measure_noise_frames = calc_num_subframes(
        measure_noise_samples,
        frame_length = frame_length,
        overlap_samples = num_overlap_samples,
        zeropad = True,
    )
    
    # initialize empty matrix to vad values into
    vad_matrix = create_empty_matrix(
        (num_subframes,),
         complex_vals = False,
    )
    vadidx_matrix = create_empty_matrix(
        (num_subframes,),
         complex_vals = False,
    ).astype(int)
    section_start = 0
    speech = 0
    silence = 0
    min_energy_array = np.zeros(measure_noise_frames)
    min_freq_array = np.zeros(measure_noise_frames)
    min_sfm_array = np.zeros(measure_noise_frames)
    window_frame = create_window(window, frame_length)
    for frame in range(num_subframes):
        # include signal idx for current frame
        vadidx_matrix[frame] += int(section_start)
        samples = data[section_start:section_start+frame_length]
        samples_windowed = apply_window(samples, window_frame, zeropad = True)
        
        # apply dft to large window - increase frequency resolution during warping
        section_fft = calc_fft(
            samples_windowed, 
            real_signal = real_signal,
            fft_bins = fft_bins,
        )
        # limit exploration of dominant frequency to max frequency (Nyquist Theorem)
        section_fft = section_fft[:max_freq]
        section_power = calc_power(section_fft)
        # set minimum values if not yet set
        if min_energy is None and frame == 0:
            min_energy = short_term_energy(samples)
        #elif frame < measure_noise_frames:
            #new_min_energy = short_term_energy(samples)
            #min_energy_array[frame] = new_min_energy
            #min_energy = np.mean(min_energy_array[:frame+1])
        ste = short_term_energy(samples)
        if ste < min_energy and frame < measure_noise_frames:
            min_energy = ste
            
        if min_freq is None and frame == 0:
            min_freq = get_dom_freq(section_power)
        #elif frame < measure_noise_frames:
            #new_min_freq = get_dom_freq(section_power)
            #min_freq_array[frame] = new_min_freq
            #min_freq = np.mean(min_freq_array[:frame+1])
        f = get_dom_freq(section_power)
        if f < min_freq and frame < measure_noise_frames:
            min_freq = f
        
        # TODO fix when NAN value
        if min_sfm is None and frame == 0:
            try:
                min_sfm = spectral_flatness_measure(section_fft)
            except TypeError:
                min_sfm = 0
        #elif frame < measure_noise_frames:
            #new_min_sfm = spectral_flatness_measure(section_fft)
            #min_sfm_array[frame] = new_min_sfm
            #min_sfm= np.mean(min_sfm_array[:frame+1])
        
        # TODO fix when NAN value
        try:
            sfm = spectral_flatness_measure(section_fft)
        except TypeError:
            if frame == 0:
                sfm = 0
            else:
                pass # it should already be defined
        if sfm < min_sfm  and frame < measure_noise_frames:
            min_sfm = sfm         
        # decide if speech or silence 
        # not finding this helpful
        if frame < measure_noise_frames:
            thresh_e = energy_thresh * np.log(min_energy)
        else:
            # what if frame == 0 and measure_noise_frames == 0?
            # still samples to process
            thresh_e = energy_thresh # TODO: test this fix
        
        counter = 0
        if ste - min_energy > thresh_e:
            counter += 1
        if f - min_freq > freq_thresh:
            counter += 1
        if sfm - min_sfm > sfm_thresh:
            counter += 1
        if counter >= 1:
            vad_matrix[frame] += 1
            speech += 1
            # set silence back to 0 if at least 5 speech frames
            if speech > 5:
                silence = 0
        else:
            vad_matrix[frame] += 0
            # update min energy and silence count
            silence += 1
            # not finding this helpful
            if frame < measure_noise_frames:
                min_energy = ((silence * min_energy) + ste) / \
                    silence + 1
            # if silence has been longer than 10 frames, set speech to 0
            if silence > 10:
                speech = 0
        # not finding this helpful
        if frame < measure_noise_frames:
            thresh_e = energy_thresh * np.log(min_energy)
        section_start += (frame_length - num_overlap_samples)
    #TODO is relevant sending back min_energy, min_freq, min_sfm?
    return vad_matrix, vadidx_matrix, #(sr, min_energy, min_freq, min_sfm)
