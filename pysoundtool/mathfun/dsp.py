#!/bin/bash
# Copyright 2019 Peggy Sylopp und Aislyn Rose GbR
# All rights reserved
# This file is part of the  NoIze-framework
# The NoIze-framework is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by the  
# Free Software Foundation, either version 3 of the License, or (at your option) 
# any later version.
#
#@author Aislyn Rose
#@version 0.1
#@date 31.08.2019
#
# The  NoIze-framework  is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details. 
#
# You should have received a copy of the GNU AFFERO General Public License 
# along with the NoIze-framework. If not, see http://www.gnu.org/licenses/.
"""
Script with functions useful in filtering / digital signal processing
"""
###############################################################################
import numpy as np
from scipy.signal import hamming, hann, resample
from scipy.io.wavfile import read
from numpy.fft import fft, ifft
from python_speech_features import logfbank, mfcc


def load_signal(wav, sampling_rate=48000, dur_sec=None):
    '''Loads wavfile, resamples if necessary, and normalizes signal.
    '''
    sr, samps = read(wav)
    if len(samps.shape) > 1:
        samps = np.take(samps,0,axis=-1) 
    if sr != sampling_rate:
        samps, sr = resample_audio(samps, sr, sampling_rate)
    if dur_sec:
        numsamps = int(dur_sec * sampling_rate)
    else:
        numsamps = len(samps)
    # zero pad if signal is too short:
    if len(samps) < numsamps:
        diff = numsamps - len(samps)
        signal_zeropadded = np.zeros(
            (samps.shape[0] + int(diff)))
        for i, row in enumerate(samps):
            signal_zeropadded[i] += row
        samps = signal_zeropadded
    elif len(samps) > numsamps:
        samps = samps[:numsamps]
    #ensure max and min are between 1 and -1
    samps = np.interp(samps,(samps.min(), samps.max()),(-1, 1))
    return samps, sr

def resample_audio(samples, sr_original, sr_desired):
    '''Allows audio samples to be resampled to desired sample rate.
    '''
    time_sec = len(samples)/sr_original 
    num_samples = int(time_sec * sr_desired)
    resampled = resample(samples, num_samples)
    return resampled, sr_desired

def calc_frame_length(dur_frame_millisec, sampling_rate):
    """Calculates the number of samples necessary for each frame

    Parameters
    ----------
    dur_frame_millisec : int or float
        time in milliseconds each frame should be
    sampling_rate : int
        sampling rate of the samples to be framed

    Returns
    -------
    frame_length : int
        the number of samples necessary to fill a frame

    Examples
    --------
    >>> calc_frame_length(dur_frame_millisec=20, sampling_rate=1000)
    20
    >>> calc_frame_length(dur_frame_millisec=20, sampling_rate=48000)
    960
    >>> calc_frame_length(dur_frame_millisec=25.5, sampling_rate=22500)
    573
    """
    frame_length = int(dur_frame_millisec * sampling_rate // 1000)
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

def calc_fft(signal_section, norm=False):
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

def calc_ifft(signal_section, norm=False):
    """Calculates the inverse fft of a series of fft values

    The real values of the ifft can be used to be saved as an audiofile

    Parameters
    ----------
    signal_section : ndarray
        The frame of fft values to apply the inverse fft to
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
                         samplerate=sr,
                         winlen=window_size_ms * 0.001,
                         winstep=window_shift_ms * 0.001,
                         nfilt=num_filters,
                         nfft=frame_length)
    elif 'mfcc' in feature_type:
        feats = mfcc(samples,
                     samplerate=sr,
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

def postfilter(original_powerspec, noisereduced_powerspec, gain,
               threshold=0.4, scale=10):
    '''Apply filter that reduces musical noise resulting from other filter.
    
    If it is estimated that speech (or target signal) is present, reduced
    filtering is applied.

    References 
    ----------
    
    T. Esch and P. Vary, "Efficient musical noise suppression for speech enhancement 
    system," Proceedings of IEEE International Conference on Acoustics, Speech and 
    Signal Processing, Taipei, 2009.
    '''
    power_ratio_current_frame = calc_power_ratio(
        original_powerspec,
        noisereduced_powerspec)
    # is there speech? If so, SNR decision = 1
    if power_ratio_current_frame < threshold:
        SNR_decision = power_ratio_current_frame
    else:
        SNR_decision = 1
    noise_frame_len = calc_noise_frame_len(SNR_decision, threshold, scale)
    # apply window
    postfilter_coeffs = calc_linear_impulse(
        noise_frame_len,
        num_freq_bins=original_powerspec.shape[0])
    gain_postfilter = np.convolve(gain, postfilter_coeffs, mode='valid')
    return gain_postfilter

if __name__ == '__main__':
    import doctest
    doctest.testmod()
