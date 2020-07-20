'''Feats module includes functions related to converting audio sample data 
to features for analysis, filtering, machine learning, or visualization.  
''' 


###############################################################################
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)

import numpy as np
import math
import librosa
from scipy.signal import hann, hamming
import pathlib
from python_speech_features import fbank, mfcc
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib.pyplot as plt
import pysoundtool as pyst




# TODO Clean up   
def plot(feature_matrix, feature_type, 
                    save_pic=False, name4pic=None, energy_scale='power_to_db',
                    title=None, sr=None, win_size_ms=None, percent_overlap=None,
                    x_label=None, y_label=None, use_scipy = False):
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
    energy_scale : str, optional
        If features need to be adjusted, e.g. from power to decibels. 
        Default is 'power_to_db'.
    title : str, optional
        The title for the graph. If None, `feature_type` is used.
    sr : int, optional
        Useful for plotting a signal type feature matrix. Allows x-axis to be
        presented as time in seconds.
    '''
    # ensure real numbers
    if feature_matrix.dtype == np.complex64 or feature_matrix.dtype == np.complex128:
        feature_matrix = np.abs(feature_matrix)
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
    if energy_scale is None or feature_type == 'signal':
        energy_label = 'energy'
        energy_scale = None
    if energy_scale == 'power_to_db':
        feature_matrix = librosa.power_to_db(feature_matrix)
        energy_label = 'decicels'
    elif energy_scale == 'db_to_power':
        feature_matrix = librosa.db_to_power(feature_matrix)
        energy_label = 'power'
    elif energy_scale == 'amplitude_to_db':
        feature_matrix = librosa.amplitude_to_db(feature_matrix)
        energy_label = 'decibels'
    elif energy_scale == 'db_to_amplitude':
        feature_matrix = librosa.db_to_amplitude(feature_matrix)
        energy_label = 'amplitude'
    plt.clf()
    if 'signal' not in feature_type:
        x_axis_label = 'Frequency bins'
    else:
        x_axis_label = 'Samples over time'
    if 'signal' in feature_type:
        # transpose matrix if second dimension is larger - probably 
        # because channels are in first dimension. Expect in second dimension
        if not feature_matrix.shape[0] > feature_matrix.shape[1]:
            feature_matrix = feature_matrix.T
        if sr is not None:
            x_axis_label = 'Time (sec)'
            dur_sec = feature_matrix.shape[0] / sr
            time_sec = pyst.dsp.get_time_points(dur_sec, sr)
            for channel in range(feature_matrix.shape[1]):
                data = feature_matrix[:,channel]
                # overlay the channel data
                if len(time_sec) > len(data):
                    time_sec = time_sec[:len(data)]
                elif len(time_sec) < len(data):
                    data = data[:len(time_sec)]
                plt.plot(time_sec, data)
        else:
            for channel in range(feature_matrix.shape[1]):
                data = feature_matrix[:,channel]
                # overlay the channel data
                plt.plot(data)
                ##display.waveplot(data,sr=sr)
        x_axis_label += ' across {} channel(s)'.format(channel+1)
    else:
        plt.pcolormesh(feature_matrix.T)
        ## display.specshow(feature_matrix.T, sr=sr)
        plt.colorbar(label=energy_label)
    plt.xlabel(x_axis_label)
    plt.ylabel(axis_feature_label)
    # if feature_matrix has multiple frames, not just one
    if feature_matrix.shape[1] > 1: 
        if 'signal' not in feature_type \
            and win_size_ms is not None and percent_overlap is not None:
            # the xticks basically show time but need to be multiplied by 0.01
            plt.xlabel('Time (sec)') 
            locs, labels = plt.xticks()
            new_labels=[str(round(i*0.001*win_size_ms*percent_overlap,1)) for i in locs]
            plt.xticks(ticks=locs,labels=new_labels)
        else:
            plt.xlabel('Number frames')
        plt.ylabel('Frequency bins')
    if title is None:
        plt.title('{} Features'.format(feature_type.upper()))
    else:
        plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if save_pic:
        outputname = name4pic or 'visualize{}feats'.format(feature_type.upper())
        outputname = pyst.utils.string2pathlib(outputname)
        if outputname.suffix:
            if outputname.suffix != '.png':
                # add .png as extension
                fname = outputname.name + '.png'
                outputname = outputname.parent.joinpath(fname)
        else:
            fname = outputname.stem + '.png'
            outputname = outputname.parent.joinpath(fname)
        plt.savefig(outputname)
    else:
        plt.show()

def plotsound(audiodata, feature_type='fbank', win_size_ms = 20, \
    percent_overlap = 0.5, fft_bins = None, num_filters=40, num_mfcc=40, sr=None,\
        save_pic=False, name4pic=None, energy_scale='power_to_db', mono=None, **kwargs):
    '''Visualize feature extraction depending on set parameters. Does not use Librosa.
    
    Parameters
    ----------
    audiodata : str or numpy.ndarray
        If str, wavfile (must be compatible with scipy.io.wavfile). Otherwise 
        the samples of the sound data. Note: in the latter case, `sr`
        must be declared.
    feature_type : str
        Options: 'signal', 'mfcc', or 'fbank' features. 
        MFCC: mel frequency cepstral
        coefficients; FBANK: mel-log filterbank energies (default 'fbank')
    win_size_ms : int or float
        Window length in milliseconds for Fourier transform to be applied
        (default 20)
    percent_overlap : int or float 
        Amount of overlap between processing windows. For example, if `percent_overlap`
        is set at 0.5, the overlap will be half that of `win_size_ms`. (default 0.5) 
        If an integer is provided, it will be converted to a float between 0 and 1.
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
    sr : int, optional
        The sample rate of the sound data or the desired sample rate of
        the wavfile to be loaded. (default None)
    mono : bool, optional
        When loading an audiofile, True will limit number of channels to
        one; False will allow more channels to be loaded. (default None, 
        which results in mono channel loading.)
    **kwargs : additional keyword arguments
        Keyword arguments for pysoundtool.feats.plot
    '''
    percent_overlap = check_percent_overlap(percent_overlap)
    feats = pyst.feats.get_feats(audiodata, feature_type=feature_type, 
                      win_size_ms = win_size_ms, percent_overlap = percent_overlap,
                      fft_bins = fft_bins, num_filters=num_filters, num_mfcc = num_mfcc,
                      sr=sr, mono = mono)
    pyst.feats.plot(feats, feature_type=feature_type, sr=sr,
                    save_pic = save_pic, name4pic=name4pic, energy_scale = energy_scale,
                    win_size_ms = win_size_ms, percent_overlap = percent_overlap,
                    **kwargs)

# TODO test duration limit on all settings
def get_feats(sound,
              sr = None, 
              feature_type = 'fbank', 
              win_size_ms = 20, 
              percent_overlap = 0.5,
              window = 'hann',
              fft_bins = None,
              num_filters = 40,
              num_mfcc = None, 
              dur_sec = None,
              mono = None,
              rate_of_change = False,
              rate_of_acceleration = False,
              subtract_mean = False,
              use_scipy = False,
              **kwargs):
    '''Collects raw signal data, stft, fbank, or mfcc features via librosa.
    
    Parameters
    ----------
    sound : str or numpy.ndarray [size=(num_samples,) or (num_samples, num_channels)]
        If str, wavfile (must be compatible with scipy.io.wavfile). Otherwise 
        the samples of the sound data. Note: in the latter case, `sr`
        must be declared.
    
    sr : int, optional
        The sample rate of the sound data or the desired sample rate of
        the wavfile to be loaded. (default None)
    
    feature_type : str
        Options include 'signal', 'stft', 'powspec', 'fbank',  or 'mfcc' data 
        (default 'fbank').
        signal: energy/amplitude measurements along time
        STFT: short-time fourier transform
        powspec : power spectrum (absolute value of stft, squared)
        FBANK: mel-log filterbank energies 
        MFCC: mel frequency cepstral coefficients 
    
    win_size_ms : int or float
        Window length in milliseconds for Fourier transform to be applied
        (default 20)
    
    percent_overlap : int or float 
        Amount of overlap between processing windows. For example, if `percent_overlap`
        is set at 0.5, the overlap will be half that of `win_size_ms`. (default 0.5) 
        If an integer is provided, it will be converted to a float between 0 and 1. 
    
    window : str or np.ndarray [size (n_fft, )]
        The window function to be applied to each window. (Default 'hann')
    
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
    
    dur_sec : float, optional
        Time in seconds to limit in loading a signal. (default None)
    
    mono: bool, optional
        For loading an audiofile, True will result in only one channel of 
        data being loaded; False will allow additional channels be loaded. 
        (default None, which results in mono channel data)
    
    rate_of_change : bool 
        If True, the first derivative of spectral data will be concatenated 
        to the features.
        This is applicable for all feature types except 'signal'.
        
    rate_of_acceleration : bool 
        If True, the second derivative of spectral data will be concatenated 
        to the features.
        This is applicable for all feature types except 'signal'.
        
    subtract_mean : bool 
        If True, the mean of each feature column will be subtracted from
        each row. This is applicable for all feature types except 'signal'.
    
    **kwargs : additional keyword arguments
        Additional keyword arguments for librosa.filters.mel.
        
    Returns
    -------
    feats : tuple (num_samples, sr) or np.ndarray [size (num_frames, num_filters) dtype=np.float or np.complex]
        Feature data. If `feature_type` is 'signal', returns a tuple containing samples and sampling rate. If `feature_type` is of another type, returns np.ndarray with shape (num_frames, num_filters/features)
    '''
    # load data
    if isinstance(sound, str) or isinstance(sound, pathlib.PosixPath):
        if mono is None:
            mono = True
        data, sr = pyst.loadsound(sound, sr = sr, dur_sec = dur_sec, mono = mono)
        if mono is False and len(data.shape) > 1:
            index_samples = np.argmax(data.shape)
            index_channels = np.argmin(data.shape)
            num_channels = data.shape[index_channels]
            # transpose data to be (samples, num_channels) rather than (num_channels, samples)
            if index_channels == 0:
                data = data.T 
            # remove additional channel for 'stft', 'fbank' etc. feature
            # extraction
            if 'signal' not in feature_type and num_channels > 1:
                import Warnings
                Warnings.warn('Only one channel is used for {}'.format(feature_type)+\
                    ' feature extraction. Removing extra channels.')
                data = data[:,0]
    else:
        if sr is None:
            raise ValueError('No samplerate given. Either provide '+\
                'filename or appropriate samplerate.')
        data, sr = sound, sr
        if dur_sec:
            data = data[:int(sr*dur_sec)]
    # ensure percent overlap is between 0 and 1
    percent_overlap = check_percent_overlap(percent_overlap)
    win_shift_ms = win_size_ms * percent_overlap
    try:
        if fft_bins is None:
            fft_bins = int(win_size_ms * sr // 1000)
        if 'fbank' in feature_type:
            if use_scipy:
                feats = pyst.feats.get_mfcc_fbank(
                    data,
                    feature_type = 'fbank',
                    sr = sr,
                    win_size_ms = win_size_ms,
                    percent_overlap = percent_overlap,
                    num_filters = num_filters,
                    fft_bins = fft_bins,
                    window_function = window)
            else:
                feats = librosa.feature.melspectrogram(
                    data,
                    sr = sr,
                    n_fft = fft_bins,
                    hop_length = int(win_shift_ms*0.001*sr),
                    n_mels = num_filters, window=window,
                    **kwargs).T
        elif 'mfcc' in feature_type:
            if num_mfcc is None:
                num_mfcc = num_filters
            if use_scipy:
                feats = pyst.feats.get_mfcc_fbank(
                    data,
                    feature_type = 'mfcc',
                    sr = sr,
                    win_size_ms = win_size_ms,
                    percent_overlap = percent_overlap,
                    num_filters = num_filters,
                    num_mfcc = num_mfcc,
                    fft_bins = fft_bins,
                    window_function = window)
            else:
                feats = librosa.feature.mfcc(
                    data,
                    sr = sr,
                    n_mfcc = num_mfcc,
                    n_fft = fft_bins,
                    hop_length = int(win_shift_ms*0.001*sr),
                    n_mels = num_filters,
                    window=window,
                    **kwargs).T
        elif 'stft' in feature_type or 'powspec' in feature_type:
            if use_scipy:
                feats = pyst.feats.get_stft(
                    data,
                    sr = sr, 
                    win_size_ms = win_size_ms,
                    percent_overlap = percent_overlap,
                    real_signal = False,
                    fft_bins = fft_bins,
                    window = window)
            else:
                feats = librosa.stft(
                    data,
                    n_fft = fft_bins,
                    hop_length = int(win_shift_ms*0.001*sr),
                    window=window).T
            if 'powspec' in feature_type:
                feats = np.abs(feats)**2
        elif 'signal' in feature_type:
            feats = data
        else:
            raise ValueError('Feature type "{}" not recognized. '.format(feature_type)+\
                'Please ensure one of the following is included in `feature_type`:\n- '+\
                    '\n- '.join(pyst.feats.list_available_features()))
        if 'signal' not in feature_type:
            if rate_of_change or rate_of_acceleration:
                d, d_d = pyst.feats.get_change_acceleration_rate(feats)
                if rate_of_change:
                    feats = np.concatenate((feats, d), axis=1)
                if rate_of_acceleration:
                    feats = np.concatenate((feats, d_d), axis=1)
            if subtract_mean:
                feats -= (np.mean(feats, axis=0) + 1e-8)
    except librosa.ParameterError as e:
        # potential error in handling fortran array
        feats = get_feats(np.asfortranarray(data),
                          sr = sr,
                          feature_type = feature_type,
                          win_size_ms = win_size_ms,
                          percent_overlap = percent_overlap,
                          fft_bins = fft_bins,
                          num_filters = num_filters,
                          num_mfcc = num_mfcc,
                          dur_sec = dur_sec,
                          mono = mono,
                          rate_of_change = rate_of_change,
                          rate_of_acceleration = rate_of_acceleration,
                          subtract_mean = subtract_mean,
                          use_scipy = use_scipy
                          **kwargs)
    return feats

# allows for more control over fft bins / resolution of each iteration.
def get_stft(sound, sr=16000, win_size_ms = 50, percent_overlap = 0.5,
                real_signal = False, fft_bins = 1024, 
                window = 'hamming'):
    if isinstance(sound, np.ndarray):
        data = sound
    else:
        data, sr2 = pyst.loadsound(sound, sr=sr)
        assert sr2 == sr
    frame_length = pyst.dsp.calc_frame_length(win_size_ms, sr)
    num_overlap_samples = int(frame_length * percent_overlap)
    num_subframes = pyst.dsp.calc_num_subframes(len(data),
                                                frame_length = frame_length,
                                                overlap_samples = num_overlap_samples,
                                                zeropad = True)
    total_rows = fft_bins
    stft_matrix = pyst.dsp.create_empty_matrix((num_subframes, total_rows),
                                              complex_vals = True)
    section_start = 0
    window_frame = pyst.dsp.create_window(window, frame_length)
    for frame in range(num_subframes):
        section = data[section_start:section_start+frame_length]
        section = pyst.dsp.apply_window(section, 
                                        window_frame, 
                                        zeropad = True)
        
        section_fft = pyst.dsp.calc_fft(section, 
                                        real_signal = real_signal,
                                        fft_bins = total_rows,
                                        )
        stft_matrix[frame] = section_fft
        section_start += (frame_length - num_overlap_samples)
    
    return stft_matrix[:,:fft_bins//2]

def plot_dom_freq(sound, energy_scale = 'power_to_db',**kwargs):
    stft_matrix = pyst.feats.get_stft(sound, **kwargs)
    pitch = pyst.dsp.get_pitch(sound, **kwargs)
    stft_matrix = librosa.power_to_db(stft_matrix)
    plt.pcolormesh(stft_matrix.T)
    color = 'yellow'
    linestyle = ':'
    plt.plot(pitch, 'ro', color=color)
    plt.show()
    
def plot_vad(sound, energy_scale = 'power_to_db',**kwargs):
    stft_matrix = pyst.feats.get_stft(sound, **kwargs)
    vad, x,y,z = pyst.dsp.vad(sound, **kwargs)
    stft_matrix = librosa.power_to_db(stft_matrix)
    y_axis = stft_matrix.shape[1]
    vad = pyst.dsp.scalesound(vad, max_val = y_axis, min_val = 0)
    plt.pcolormesh(stft_matrix.T)
    color = 'yellow'
    linestyle = ':'
    plt.plot(vad, 'ro', color=color)
    plt.show()

def get_change_acceleration_rate(spectro_data):
    '''Gets first and second derivatives of spectral data.
    
    This is useful particularly for speech recognition.
    
    Parameters
    ----------
    spectro_data : np.ndarray [shape = (num_samples, num_features)]
    
    Returns
    -------
    delta : np.ndarray [shape = (num_samples, num_features)]
        The first order derivative of spectral data. Reflects rate of change in signal.
        
    delta_delta : np.ndarray [shape = (num_samples, num_features)]
        The second order derivative of spectral data. Reflects rate of acceleration in signal.
    '''
    spectro_data = spectro_data.T
    #first derivative = delta (rate of change)
    delta = librosa.feature.delta(spectro_data)
    #second derivative = delta delta (acceleration changes)
    delta_delta = librosa.feature.delta(spectro_data,order=2)
    delta = delta.T
    delta_delta = delta_delta.T
    return delta, delta_delta

def get_mfcc_fbank(samples, feature_type='mfcc', sr=48000, win_size_ms=20,
                     percent_overlap=0.5, num_filters=40, num_mfcc=40,
                     fft_bins = None, window_function = None, zeropad = True, **kwargs):
    '''Collects fbank or mfcc features via python speech features.
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
    if len(samples)/sr*1000 < win_size_ms:
        if zeropad:
            samples = pyst.dsp.zeropad_sound(samples, win_size_ms * sr / 1000, sr = sr)
        else:
            win_size_ms = len(samples)/sr*1000
    frame_length = pyst.dsp.calc_frame_length(win_size_ms, sr)
    percent_overlap = check_percent_overlap(percent_overlap)
    window_shift_ms = win_size_ms * percent_overlap
    if 'fbank' in feature_type:
        feats, energy = fbank(samples,
                         samplerate = sr,
                         winlen = win_size_ms * 0.001,
                         winstep = window_shift_ms * 0.001,
                         nfilt = num_filters,
                         nfft = fft_bins,
                         winfunc = window_function, 
                         **kwargs)
    elif 'mfcc' in feature_type:
        feats = mfcc(samples,
                     samplerate = sr,
                     winlen = win_size_ms * 0.001,
                     winstep = window_shift_ms * 0.001,
                     nfilt = num_filters,
                     numcep = num_mfcc,
                     nfft = fft_bins,
                     winfunc = window_function,
                     **kwargs)
    return feats

def zeropad_features(feats, desired_shape, complex_vals = False):
    '''Applies zeropadding to a copy of feats. 
    '''
    # to avoid UFuncTypeError:
    if feats.dtype == np.complex or feats.dtype == np.complex64 or \
        feats.dtype == np.complex128:
            complex_vals = True
    fts = feats.copy()
    if feats.shape != desired_shape:
        if complex_vals:
            dtype = np.complex
        else:
            dtype = np.float
        empty_matrix = np.zeros(desired_shape, dtype = dtype)
        try:
            if len(desired_shape) == 1:
                empty_matrix[:feats.shape[0]] += feats
            elif len(desired_shape) == 2:
                empty_matrix[:feats.shape[0], 
                            :feats.shape[1]] += feats
            elif len(desired_shape) == 3:
                empty_matrix[:feats.shape[0], 
                            :feats.shape[1],
                            :feats.shape[2]] += feats
            elif len(desired_shape) == 4:
                empty_matrix[:feats.shape[0], 
                            :feats.shape[1],
                            :feats.shape[2],
                            :feats.shape[3]] += feats
            elif len(desired_shape) == 5:
                empty_matrix[:feats.shape[0], 
                            :feats.shape[1],
                            :feats.shape[2],
                            :feats.shape[3],
                            :feats.shape[4]] += feats
            else:
                raise TypeError('Zeropadding columns requires a matrix with '+\
                    'a minimum of 1 dimension and maximum of 5 dimensions.')
            fts = empty_matrix
        except ValueError as e:
            print(e)
            raise ValueError('The desired shape is smaller than the original shape.'+ \
                ' No zeropadding necessary.')
        except IndexError as e:
            print(e)
            raise IndexError('The dimensions do not align. Zeropadding '+ \
                'expects same number of dimensions.')
    assert fts.shape == desired_shape
    return fts

def reduce_num_features(feats, desired_shape, complex_vals = False):
    '''Limits number features of a copy of feats. 
    '''
    fts = feats.copy()
    if feats.shape != desired_shape:
        if complex_vals:
            dtype = np.complex_
        else:
            dtype = np.float
        empty_matrix = np.zeros(desired_shape, dtype = dtype)
        try:
            if len(desired_shape) == 1:
                empty_matrix += feats[:empty_matrix.shape[0]]
            elif len(desired_shape) == 2:
                empty_matrix += feats[:empty_matrix.shape[0], 
                            :empty_matrix.shape[1]]
            elif len(desired_shape) == 3:
                empty_matrix += feats[:empty_matrix.shape[0], 
                            :empty_matrix.shape[1],
                            :empty_matrix.shape[2]]
            elif len(desired_shape) == 4:
                empty_matrix += feats[:empty_matrix.shape[0], 
                            :empty_matrix.shape[1],
                            :empty_matrix.shape[2],
                            :empty_matrix.shape[3]]
            elif len(desired_shape) == 5:
                empty_matrix += feats[:empty_matrix.shape[0], 
                            :empty_matrix.shape[1],
                            :empty_matrix.shape[2],
                            :empty_matrix.shape[3],
                            :empty_matrix.shape[4]]
            else:
                raise TypeError('Reducing items in columns requires a matrix with'+\
                    ' a minimum of 1 dimension and maximum of 5 dimensions.')
            fts = empty_matrix
        except ValueError as e:
            print(e)
            raise ValueError('The desired shape is larger than the original shape.'+ \
                ' Perhaps try zeropadding.')
        except IndexError as e:
            print(e)
            raise IndexError('The dimensions do not align. Zeropadding '+ \
                'expects same number of dimensions.')
    assert fts.shape == desired_shape
    return fts

# TODO remove warning for 'operands could not be broadcast together with shapes..'
# TODO test
def adjust_shape(data, desired_shape, change_dims = False):
    if len(data.shape) != len(desired_shape):
        if not change_dims:
            raise ValueError('Cannot adjust data to a different number of '+\
                'dimensions.\nOriginal data shape: '+str(data.shape)+ \
                    '\nDesired shape: '+str(desired_shape))
        total_desired_samples = 1
        for i in desired_shape:
            total_desired_samples *= i
        data_flattened = data.flatten()
        if len(data_flattened) < total_desired_samples:
            data_flattened = pyst.feats.zeropad_features(data_flattened, 
                                                         desired_shape = (total_desired_samples,))
        if len(data_flattened) > total_desired_samples:
            data_flattened = data_flattened[:total_desired_samples]
        data_prepped = data_flattened.reshape(desired_shape)
        return data_prepped
        
    # attempt to zeropad data:
    try:
        data_prepped = pyst.feats.zeropad_features(data, 
                                                  desired_shape = desired_shape)
    # if zeropadding is smaller than data.shape/features:
    except ValueError:
        # remove extra data/columns to match desired_shape:
        data_prepped = pyst.feats.reduce_num_features(data, 
                                                 desired_shape = desired_shape)
    return data_prepped

def check_percent_overlap(percent_overlap):
    '''Ensures percent_overlap is between 0 and 1.
    '''
    if percent_overlap > 1:
        percent_overlap *= 0.01
        if percent_overlap > 1:
            raise ValueError('The percent overlap value '+str(percent_overlap)+\
                ' is too large. Please use a value between 0 and 1 or 0 and 100.')
    return percent_overlap

def separate_dependent_var(matrix):
    '''Separates matrix into features and labels. Expects 3D array.

    Assumes the last column of the last dimension of the matrix constitutes
    the dependent variable (labels), and all other columns the indpendent variables
    (features). Additionally, it is assumed that for each block of data, 
    only one label is needed; therefore, just the first label is taken for 
    each block.

    Parameters
    ----------
    matrix : numpy.ndarray [size = (num_samples, num_frames, num_features)]
        The `matrix` holds the numerical data to separate. num_features is
        expected to be at least 2.

    Returns
    -------
    X : numpy.ndarray [size = (num_samples, num_frames, num_features -1)]
        A matrix holding the (assumed) independent variables
    y : numpy.ndarray, numpy.int64, numpy.float64 [size = (num_samples,)]
        A vector holding the labels assigned to the independent variables.
        If only one value in array, just the value inside is returned

    Examples
    --------
    >>> import numpy as np
    >>> #vector
    >>> separate_dependent_var(np.array([1,2,3,4]))
    (array([1, 2, 3]), 4)
    >>> #simple matrix
    >>> matrix = np.arange(4).reshape(2,2)
    >>> matrix
    array([[0, 1],
           [2, 3]])
    >>> X, y = separate_dependent_var(matrix)
    >>> X
    array([[0],
           [2]])
    >>> y 
    1
    >>> #more complex matrix
    >>> matrix = np.arange(20).reshape((2,2,5))
    >>> matrix
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9]],
    <BLANKLINE>
           [[10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19]]])
    >>> X, y = separate_dependent_var(matrix)
    >>> X
    array([[[ 0,  1,  2,  3],
            [ 5,  6,  7,  8]],
    <BLANKLINE>
           [[10, 11, 12, 13],
            [15, 16, 17, 18]]])
    >>> y
    array([ 4, 14])
    '''
    # get last column
    if matrix.shape[-1] == 1:
        raise ValueError('Expects input matrix to be size (num_samples, num_frames, ' + \
                         'num_features). Number of features must exceed 1 in order ' + \
                         'to separate into X and y arrays.')
    y_step1 = np.take(matrix, -1, axis=-1)
    # because the label is the same for each block of data, just need the first
    # row,  not all the rows, as they are the same label.
    y = np.take(y_step1, 0, axis=-1)
    # get features:
    X = np.delete(matrix, -1, axis=-1)
    return X, y

# TODO: perhaps remove - just use np.expand_dims() 
# TODO: https://github.com/biopython/biopython/issues/1496
# Fix numpy array repr for Doctest. 
def add_tensor(matrix):
    '''Adds tensor / dimension to input ndarray (e.g. features).

    Keras requires an extra dimension at some layers, which represents 
    the 'tensor' encapsulating the data. 

    Further clarification taking the example below. The input matrix has 
    shape (2,3,4). Think of it as 2 different events, each having
    3 sets of measurements, with each of those having 4 features. So, 
    let's measure differences between 2 cities at 3 different times of
    day. Let's take measurements at 08:00, 14:00, and 19:00 in... 
    Magic City and Never-ever Town. We'll measure.. 1) tempurature, 
    2) wind speed 3) light level 4) noise level.

    How I best understand it, putting our measurements into a matrix
    with an added dimension/tensor, this highlights the separate 
    measurements, telling the algorithm: yes, these are 4 features
    from the same city, BUT they occur at different times. Or it's 
    just how Keras set up the code :P 

    Parameters
    ----------
    matrix : numpy.ndarray
        The `matrix` holds the numerical data to add a dimension to.

    Returns
    -------
    matrix : numpy.ndarray
        The `matrix` with an additional dimension.

    Examples
    --------
    >>> import numpy as np
    >>> matrix = np.arange(24).reshape((2,3,4))
    >>> matrix.shape
    (2, 3, 4)
    >>> matrix
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    <BLANKLINE>
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
    >>> matrix_2 = add_tensor(matrix)
    >>> matrix_2.shape
    (2, 3, 4, 1)
    >>> matrix_2
    array([[[[ 0],
             [ 1],
             [ 2],
             [ 3]],
    <BLANKLINE>
            [[ 4],
             [ 5],
             [ 6],
             [ 7]],
    <BLANKLINE>
            [[ 8],
             [ 9],
             [10],
             [11]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[12],
             [13],
             [14],
             [15]],
    <BLANKLINE>
            [[16],
             [17],
             [18],
             [19]],
    <BLANKLINE>
            [[20],
             [21],
             [22],
             [23]]]])
    '''
    if isinstance(matrix, np.ndarray) and len(matrix) > 0:
        matrix = matrix.reshape(matrix.shape + (1,))
        return matrix
    elif isinstance(matrix, np.ndarray):
        raise ValueError('Input matrix is empty.')
    else:
        raise TypeError('Expected type numpy.ndarray, recieved {}'.format(
            type(matrix)))
    
# TODO improve / remove.. move to data module?
def scale_X_y(matrix, is_train=True, scalars=None):
    '''Separates and scales data into X and y arrays. Adds dimension for keras.
    
    Assumes the last column of the last dimension is the y or label data.
    
    Parameters
    ----------
    matrix : np.ndarray [size = (num_samples, num_frames, num_features)]
        Matrix with X and y data
    is_train : bool
        Relevant for the `scalars` parameter. If the data is training
        data (i.e. True), the `scalars` will be created. If the data
        is test data (i.e. False), the function expects `scalars` to 
        be provided. (default True)
    scalars : dict, optional
        Dictionary with scalars to be applied to non-training data.
        
    Returns
    -------
    X : np.ndarray [size = (num_sampls, num_frames, num_features-1, 1)]
        Scaled features with extra dimension
    y : np.ndarray [size = (num_samples, 1, 1)]
        Scaled independent variable with extra dimension
    scalars : dict
        The scalars either created or previously loaded.
    '''
    X, y = pyst.feats.separate_dependent_var(matrix)
    if is_train:
        scalars = {}
    elif scalars is None:
        raise TypeError('If non-train data, `scalars` cannot be of type None.')
    if len(X.shape) != 3:
        raise ValueError('Expected 3d input, not input of shape {}.'.format(
            matrix.shape))
    if X.dtype == np.complex_:
        # convert stft to power spectrum
        print('\nTaking absolute value and power of complex data..'+\
            '\ni.e. Removing complex values.')
        X = np.abs(X)**2
    for j in range(X.shape[2]):
        if is_train:
            scalars[j] = StandardScaler()
            X[:, :, j] = scalars[j].fit_transform(X[:, :, j])
        else:
            X[:, :, j] = scalars[j].transform(X[:, :, j])
        X[:, :, j] = normalize(X[:, :, j])
    # Keras needs an extra dimension as a tensor / holder of data
    X = pyst.feats.add_tensor(X)
    y = pyst.feats.add_tensor(y)
    return X, y, scalars

# TODO move to data module??
# TODO test
def normalize(data):
    # need to take power - complex values in stft
    if data.dtype == np.complex_:
        # take power of absoulte value of stft
        data = np.abs(data)**2
    # add epsilon to avoid division by zero error
    eps = 2**-52
    data = (data - np.min(data)) / \
        ((np.max(data) - np.min(data)) + eps)
    return data

# TODO test for all these features:
def list_available_features():
    return ['stft', 'powspec', 'fbank', 'mfcc', 'signal']

def save_features_datasets(datasets_dict, datasets_path2save_dict, dur_sec,
                                    feature_type='fbank', num_feats=None, sr=22050, 
                                    win_size_ms=20, percent_overlap=0.5, n_fft = None,
                                    frames_per_sample=None,labeled_data=False, 
                                    subsection_data=False, divide_factor=None,
                                    visualize=False, vis_every_n_frames=50, 
                                    use_librosa=True, center=True, mode='reflect', 
                                    log_settings=True, decode_dict = None, **kwargs):
    '''Extracts and saves audio features, sectioned into datasets, to indicated locations.
    
    If MemoryError, the provided dataset dicts will be adjusted to allow data to be subsectioned.
    
    Parameters
    ----------
    datasets_dict : dict 
        Dictionary with keys representing datasets and values the audifiles making up that dataset.
        E.g. {'train':['1.wav', '2.wav', '3.wav'], 'val': ['4.wav'], 'test':['5.wav']} for unlabled
        data or  {'train':[(0, '1.wav'), (1, '2.wav'), (0, '3.wav')], 'val': [(1, '4.wav')], 
        'test':[(0, '5.wav')]} for labeled data.
    datasets_path2save_dict : dict
        Dictionary with keys representing datasets and values the pathways of where extracted 
        features of that dataset will be saved.
        E.g. {'train': './data/train.npy', 'val': './data/val.npy', 'test': './data/test.npy'}
    feature_type : str 
        String including only one of the following: 'signal', 'stft', 'powspec', 'fbank', and 'mfcc'. 
        'signal' currently only supports mono channel data. TODO test for stereo
        'powspec' and 'stft' are basically the same; 'powspec' is the 'stft' except without 
        complex values and squared. E.g 'mfcc_noisy' or 'stft_train'.
    sr : int 
        The sample rate the audio data should be loaded with.
    n_fft : int 
        The number of frequency bins used for the Fast Fourier Transform (fft)
    dur_sec : int or float
        The desired duration of how long the audio data should be. This is used to calculate 
        size of feature data and is therefore necessary, as audiofiles tend to differe in length.
        If audiofiles are longer or shorter, they will be cut or zeropadded respectively.
    num_feats : int 
        The number of mfcc coefficients (mfcc), mel filters (fbank), or frequency bins (stft).
    win_size_ms : int 
        The desired window size in milliseconds to process audio samples.
    percent_overlap : float
        The amount audio samples should overlap as each window is processed.
    frames_per_sample : int, optional 
        If you want to section each audio file feature data into smaller frames. This might be 
        useful for speech related contexts. (Can avoid this by simply reshaping data later)
    labeled_data : bool 
        If True, expects each audiofile to be accompanied by an integer label. See example 
        given for `datasets_dict`.
    subsection_data : bool 
        If you have a large dataset, you may want to divide it into subsections. See 
        pysoundtool.datasets.subsection_data. If datasets are large enough to raise a MemoryError, 
        this will be applied automatically.
    divide_factor : int, optional
        The number of subsections to divide data into. Only large enough sections will be divided.
        If smaller datasets (i.e. validation and test datasets) are as large or smaller than 
        the new subsectioned larger dataset(s) (i.e. train), they will be left unchanged.
        (defaults to 5)
    visualize : bool
        If True, periodic plots of the features will be saved throughout the extraction process. (default False)
    vis_every_n_frames : int 
        How often visuals should be made: every 10 samples, every 100, etc. (default 50)
    use_librosa : bool 
        If True, librosa is used to load and extract features. As of now, no other option is 
        available. TODO: add other options. :P I just wanted to be clear that some elements
        of this function are unique to using librosa. (default True)
    center : bool 
        Relevant for librosa and feature extraction. (default True)
    mode : str 
        Relevant for librosa and feature extraction. (default 'reflect')
    log_settings : bool
        If True, a .csv file will be saved in the feature extraction directory with 
        most of the feature settings saved. (default True)
    decode_dict : dict, optional
        The dictionary to get the label given the encoded label. This is for plotting 
        purposes. (default None)
    **kwargs : additional keyword arguments
        Keyword arguments for `pysoundtool.feats.get_feats`.
    
    Returns
    -------
    datasets_dict : dict 
        The final dataset dictionary used in feature extraction. The datasets may 
        have been subdivided.
    datasets_path2save_dict : dict
        The final dataset feature pathway dict. The pathways will have been 
        adjusted if the datasets have been subdivided.
        
    See Also
    --------
    pysoundtool.feats.get_feats
        Extract features from audio file or audio data.
    '''
    # if dataset is large, may want to divide it into sections
    if divide_factor is None:
        divide_factor = 5
    if subsection_data:
        datasets_dict, datasets_path2save_dict = pyst.datasets.section_data(
            datasets_dict,
            datasets_path2save_dict,
            divide_factor=divide_factor)
    try:
        # depending on which packages one uses, shape of data changes.
        # for example, Librosa centers/zeropads data automatically
        # TODO see which shapes result from python_speech_features
        total_samples = pyst.dsp.calc_frame_length(dur_sec*1000, sr=sr)
        # if using Librosa:
        if use_librosa:
            frame_length = pyst.dsp.calc_frame_length(win_size_ms, sr)
            hop_length = int(win_size_ms*percent_overlap*0.001*sr)
            if n_fft is None:
                n_fft = frame_length
            # librosa centers samples by default, sligthly adjusting total 
            # number of samples
            if center:
                y_zeros = np.zeros((total_samples,))
                y_centered = np.pad(y_zeros, int(n_fft // 2), mode=mode)
                total_samples = len(y_centered)
            # each audio file 
            if 'signal' in feature_type:
                # don't apply fft to signal (not sectioned into overlapping windows)
                total_rows_per_wav = total_samples // frame_length
            else:
                # do apply fft to signal (via Librosa) - (will be sectioned into overlapping windows)
                total_rows_per_wav = int(1 + (total_samples - n_fft)//hop_length)
            # set defaults to num_feats if set as None:
            if num_feats is None:
                if 'mfcc' in feature_type or 'fbank' in feature_type:
                    num_feats = 40
                elif 'powspec' in feature_type or 'stft' in feature_type:
                    num_feats = int(1+n_fft/2)
                elif 'signal' in feature_type:
                    num_feats = frame_length
                    ### how many samples make up one window frame?
                    ###num_samps_frame = int(sr * win_size_ms * 0.001)
                    #### make divisible by 10
                    #### TODO: might not be necessary
                    ###if not num_samps_frame % 10 == 0:
                        ###num_samps_frame *= 0.1
                        #### num_feats is how many samples per window frame (here rounded up 
                        #### to the nearest 10)
                        ###num_feats = int(round(num_samps_frame, 0) * 10)
                    ### limit in seconds how many samples
                    ### is this necessary?
                    ### dur_sec = num_features * frames_per_sample * batch_size / sr
                else:
                    raise ValueError('Feature type "{}" '.format(feature_type)+\
                        'not understood.\nMust include one of the following: \n'+\
                            ', '.join(list_available_features()))

            # adjust shape for model
            # input_shape: the input shape for the model
            # desired_shape: the 2D shape of expected samples. This is used for zeropadding or
            # limiting the feats to this shape. Once this shape, feats can be reshaped into input_shape
            # TODO test for labeled data with frames_per_sample
            if frames_per_sample is not None:
                # want smaller windows, e.g. autoencoder denoiser or speech recognition
                batch_size = math.ceil(total_rows_per_wav/frames_per_sample)
                if labeled_data:
                    input_shape = (batch_size, frames_per_sample, num_feats + 1)
                    desired_shape = (input_shape[0] * input_shape[1], 
                                     input_shape[2]-1)
                else:
                    input_shape = (batch_size, frames_per_sample, num_feats)
                    desired_shape = (input_shape[0]*input_shape[1],
                                     input_shape[2])
            else:
                if labeled_data:
                    input_shape = (int(total_rows_per_wav), num_feats + 1)
                    desired_shape = (input_shape[0], input_shape[1]-1)
                else:
                    input_shape = (int(total_rows_per_wav), num_feats)
                    desired_shape = input_shape
            # set whether or not features will include complex values:
            if 'stft' in feature_type:
                complex_vals = True
            else:
                complex_vals = False
            # limit feat_type to the basic feature extracted
            # for example:
            # feature_type 'powspec' is actually 'stft' but with complex info removed.
            # the basic feat_type is still 'stft'
            if 'mfcc' in feature_type:
                feat_type = 'mfcc'
            elif 'fbank' in feature_type:
                feat_type = 'fbank'
            elif 'stft' in feature_type:
                feat_type = 'stft'
            elif 'powspec' in feature_type:
                feat_type = 'stft'
            elif 'signal' in feature_type:
                feat_type = 'signal'
            else:
                raise TypeError('Expected '+', '.join(list_available_features())+\
                    ' to be in `feature_type`, not {}'.format(feature_type))
            for key, value in datasets_dict.items():
                # get parent directory of where data should be saved (i.e. for saving pics)
                datapath = datasets_path2save_dict[key]
                if not isinstance(datapath, pathlib.PosixPath):
                    datapath = pathlib.Path(datapath)
                datadir = datapath.parent
                # when loading a dictionary, the value is a string
                if isinstance(value, str):
                    value = pyst.utils.restore_dictvalue(value)
                extraction_shape = (len(value),) + input_shape
                feats_matrix = pyst.dsp.create_empty_matrix(
                    extraction_shape, 
                    complex_vals=complex_vals)
                for j, audiofile in enumerate(value):
                    if labeled_data:
                        
                        label, audiofile = int(audiofile[0]), audiofile[1]
                    if isinstance(audiofile, str):
                        audiofile = pathlib.PosixPath(audiofile)
                    feats = pyst.feats.get_feats(audiofile,
                                                sr=sr,
                                                feature_type=feat_type,
                                                win_size_ms=win_size_ms,
                                                percent_overlap=percent_overlap,
                                                num_filters=num_feats,
                                                num_mfcc=num_feats,
                                                dur_sec=dur_sec,
                                                **kwargs)
                    # if power spectrum (remove complex values and squaring features)
                    if 'powspec' in feature_type:
                        feats = np.abs(feats)**2
                        
                    if visualize:
                        if labeled_data:
                            if decode_dict is not None:
                                try:
                                    label_plot = decode_dict[label].upper()
                                except KeyError:
                                    try:
                                        label_plot = decode_dict[str(label)].upper()
                                    except KeyError:
                                        label_plot = label
                            else:
                                label_plot = label
                        else:
                            label_plot = audiofile.parent.stem.upper()
                        # visualize features:
                        if 'mfcc' in feature_type or 'signal' in feature_type:
                            energy_scale = None
                        else:
                            energy_scale = 'power_to_db'
                        #visualize features only every n num frames
                        if j % vis_every_n_frames == 0:
                            save_pic_path = datadir.joinpath(
                                'images',key,'{}_sample{}'.format(
                                    feature_type, j))
                            # make sure this directory exists
                            save_pic_dir = pyst.utils.check_dir(save_pic_path.parent, make=True)
                            pyst.feats.plot(feats, 
                                            feature_type = feature_type,
                                            win_size_ms = win_size_ms,
                                            percent_overlap = percent_overlap,
                                            energy_scale = energy_scale,
                                            title='{} {} features: label {}'.format(
                                                key, feature_type.upper(),
                                                label_plot),
                                            save_pic=visualize, 
                                            name4pic=save_pic_path)
                        
                    # zeropad feats if too short:
                    if 'signal' in feat_type:
                        feats_zeropadded = np.zeros(desired_shape)
                        feats_zeropadded = feats_zeropadded.flatten()
                        if len(feats.shape) > 1:
                            feats_zeropadded = feats_zeropadded.reshape(feats_zeropadded.shape[0],
                                                                        feats.shape[1])
                        if len(feats) > len(feats_zeropadded):
                            feats = feats[:len(feats_zeropadded)]
                        feats_zeropadded[:len(feats)] += feats

                        # reshape here for training models to avoid memory issues later 
                        # (while training) if total samples is large
                        feats = feats_zeropadded.reshape(desired_shape)
                    
                    feats = pyst.feats.zeropad_features(
                        feats, 
                        desired_shape = desired_shape,
                        complex_vals = complex_vals)
                    
                    if labeled_data:
                        # create label column
                        label_col = np.zeros((len(feats),1)) + label
                        feats = np.concatenate([feats,label_col], axis=1)



                                            
                    feats = feats.reshape(extraction_shape[1:])
                    # fill in empty matrix with features from each audiofile

                    feats_matrix[j] = feats
                    pyst.utils.print_progress(iteration = j, 
                                            total_iterations = len(value),
                                            task = '{} {} feature extraction'.format(
                                                key, feature_type))
                # save data:
                np.save(datasets_path2save_dict[key], feats_matrix)
                print('\nFeatures saved at {}\n'.format(datasets_path2save_dict[key]))
            if log_settings:
                log_filename = datadir.joinpath('log_extraction_settings.csv')
                feat_settings = dict(dur_sec=dur_sec,
                                     feature_type=feature_type,
                                     feat_type=feat_type,
                                     complex_vals=complex_vals,
                                     sr=sr,
                                     num_feats=num_feats,
                                     n_fft=n_fft,
                                     win_size_ms=win_size_ms,
                                     frame_length=frame_length,
                                     percent_overlap=percent_overlap,
                                     frames_per_sample=frames_per_sample,
                                     labeled_data=labeled_data,
                                     visualize=visualize,
                                     # different for each dataset
                                     #total_samples=total_samples, 
                                     input_shape=input_shape,
                                     desired_shape=desired_shape,
                                     use_librosa=use_librosa,
                                     center=center,
                                     mode=mode,
                                     subsection_data=subsection_data,
                                     divide_factor=divide_factor,
                                     kwargs = kwargs
                                     )
                feat_settings_path = pyst.utils.save_dict(
                    dict2save = feat_settings,
                    filename = log_filename,
                    overwrite=True)
        else:
            raise ValueError('Sorry, this functionality is not yet supported. '+\
                'Set `use_librosa` to True.')
    except MemoryError as e:
        print('MemoryError: ',e)
        print('\nSectioning data and trying again.\n')
        datasets_dict, datasets_path2save_dict = pyst.datasets.section_data(
            datasets_dict, datasets_path2save_dict, divide_factor=divide_factor)
        datasets_dict, datasets_path2save_dict = save_features_datasets(
            datasets_dict = datasets_dict, 
            datasets_path2save_dict = datasets_path2save_dict,
            feature_type = feature_type, 
            sr = sr, 
            n_fft = n_fft, 
            dur_sec = dur_sec,
            num_feats = num_feats,
            win_size_ms = win_size_ms,
            percent_overlap = percent_overlap,
            use_librosa = use_librosa, 
            window = window,
            center = center,
            mode = mode,
            frames_per_sample = frames_per_sample,
            visualize = visualize, 
            vis_every_n_frames = vis_every_n_frames, 
            labeled_data = labeled_data,
            log_settings = log_settings,
            decode_dict = decode_dict,
            **kwargs)
    return datasets_dict, datasets_path2save_dict

def save_features_datasets_zipfiles(datasets_dict, datasets_path2save_dict, 
                                    extract_dir, dur_sec,
                                    feature_type='fbank', num_feats=None, sr=22050, 
                                    win_size_ms=20, percent_overlap=0.5, n_fft = None,
                                    frames_per_sample=None, labeled_data=False, 
                                    subsection_data=False, divide_factor=None,
                                    visualize=False, vis_every_n_frames=50, 
                                    use_librosa=True, center=True, mode='reflect', 
                                    log_settings=True, decode_dict = None,
                                    audiofile_lim = 10, **kwargs):
    '''Extracts and saves audio features, sectioned into datasets, to indicated locations.
    
    If MemoryError, the provided dataset dicts will be adjusted to allow data to be subsectioned.
    
    Parameters
    ----------
    datasets_dict : dict 
        Dictionary with keys representing datasets and values the audifiles making up that dataset.
        E.g. {'train':['1.wav', '2.wav', '3.wav'], 'val': ['4.wav'], 'test':['5.wav']} for unlabled
        data or  {'train':[(0, '1.wav'), (1, '2.wav'), (0, '3.wav')], 'val': [(1, '4.wav')], 
        'test':[(0, '5.wav')]} for labeled data.
    datasets_path2save_dict : dict
        Dictionary with keys representing datasets and values the pathways of where extracted 
        features of that dataset will be saved.
        E.g. {'train': './data/train.npy', 'val': './data/val.npy', 'test': './data/test.npy'}
    feature_type : str 
        String including only one of the following: 'signal', 'stft', 'powspec', 'fbank', and 'mfcc'. 
        'signal' currently only supports mono channel data. TODO test for stereo
        'powspec' and 'stft' are basically the same; 'powspec' is the 'stft' except without 
        complex values and squared. E.g 'mfcc_noisy' or 'stft_train'.
    sr : int 
        The sample rate the audio data should be loaded with.
    n_fft : int 
        The number of frequency bins used for the Fast Fourier Transform (fft)
    dur_sec : int or float
        The desired duration of how long the audio data should be. This is used to calculate 
        size of feature data and is therefore necessary, as audiofiles tend to differe in length.
        If audiofiles are longer or shorter, they will be cut or zeropadded respectively.
    num_feats : int 
        The number of mfcc coefficients (mfcc), mel filters (fbank), or frequency bins (stft).
    win_size_ms : int 
        The desired window size in milliseconds to process audio samples.
    percent_overlap : float
        The amount audio samples should overlap as each window is processed.
    frames_per_sample : int, optional 
        If you want to section each audio file feature data into smaller frames. This might be 
        useful for speech related contexts. (Can avoid this by simply reshaping data later)
    labeled_data : bool 
        If True, expects each audiofile to be accompanied by an integer label. See example 
        given for `datasets_dict`.
    subsection_data : bool 
        If you have a large dataset, you may want to divide it into subsections. See 
        pysoundtool.datasets.subsection_data. If datasets are large enough to raise a MemoryError, 
        this will be applied automatically.
    divide_factor : int, optional
        The number of subsections to divide data into. Only large enough sections will be divided.
        If smaller datasets (i.e. validation and test datasets) are as large or smaller than 
        the new subsectioned larger dataset(s) (i.e. train), they will be left unchanged.
        (defaults to 5)
    visualize : bool
        If True, periodic plots of the features will be saved throughout the extraction process. (default False)
    vis_every_n_frames : int 
        How often visuals should be made: every 10 samples, every 100, etc. (default 50)
    use_librosa : bool 
        If True, librosa is used to load and extract features. As of now, no other option is 
        available. TODO: add other options. :P I just wanted to be clear that some elements
        of this function are unique to using librosa. (default True)
    center : bool 
        Relevant for librosa and feature extraction. (default True)
    mode : str 
        Relevant for librosa and feature extraction. (default 'reflect')
    log_settings : bool
        If True, a .csv file will be saved in the feature extraction directory with 
        most of the feature settings saved. (default True)
    decode_dict : dict, optional
        The dictionary to get the label given the encoded label. This is for plotting 
        purposes. (default None)
    **kwargs : additional keyword arguments
        Keyword arguments for `pysoundtool.feats.get_feats`.
    
    Returns
    -------
    datasets_dict : dict 
        The final dataset dictionary used in feature extraction. The datasets may 
        have been subdivided.
    datasets_path2save_dict : dict
        The final dataset feature pathway dict. The pathways will have been 
        adjusted if the datasets have been subdivided.
        
    See Also
    --------
    pysoundtool.feats.get_feats
        Extract features from audio file or audio data.
    '''
    # if dataset is large, may want to divide it into sections
    if divide_factor is None:
        divide_factor = 5
    if subsection_data:
        datasets_dict, datasets_path2save_dict = pyst.datasets.section_data(
            datasets_dict,
            datasets_path2save_dict,
            divide_factor=divide_factor)
    try:
        # depending on which packages one uses, shape of data changes.
        # for example, Librosa centers/zeropads data automatically
        # TODO see which shapes result from python_speech_features
        total_samples = pyst.dsp.calc_frame_length(dur_sec*1000, sr=sr)
        # if using Librosa:
        if use_librosa:
            frame_length = pyst.dsp.calc_frame_length(win_size_ms, sr)
            hop_length = int(win_size_ms*percent_overlap*0.001*sr)
            if n_fft is None:
                n_fft = frame_length
            # librosa centers samples by default, sligthly adjusting total 
            # number of samples
            if center:
                y_zeros = np.zeros((total_samples,))
                y_centered = np.pad(y_zeros, int(n_fft // 2), mode=mode)
                total_samples = len(y_centered)
            # each audio file 
            if 'signal' in feature_type:
                # don't apply fft to signal (not sectioned into overlapping windows)
                total_rows_per_wav = total_samples // frame_length
            else:
                # do apply fft to signal (via Librosa) - (will be sectioned into overlapping windows)
                total_rows_per_wav = int(1 + (total_samples - n_fft)//hop_length)
            # set defaults to num_feats if set as None:
            if num_feats is None:
                if 'mfcc' in feature_type or 'fbank' in feature_type:
                    num_feats = 40
                elif 'powspec' in feature_type or 'stft' in feature_type:
                    num_feats = int(1+n_fft/2)
                elif 'signal' in feature_type:
                    num_feats = frame_length
                    ### how many samples make up one window frame?
                    ###num_samps_frame = int(sr * win_size_ms * 0.001)
                    #### make divisible by 10
                    #### TODO: might not be necessary
                    ###if not num_samps_frame % 10 == 0:
                        ###num_samps_frame *= 0.1
                        #### num_feats is how many samples per window frame (here rounded up 
                        #### to the nearest 10)
                        ###num_feats = int(round(num_samps_frame, 0) * 10)
                    ### limit in seconds how many samples
                    ### is this necessary?
                    ### dur_sec = num_features * frames_per_sample * batch_size / sr
                else:
                    raise ValueError('Feature type "{}" '.format(feature_type)+\
                        'not understood.\nMust include one of the following: \n'+\
                            ', '.join(list_available_features()))

            # adjust shape for model
            # input_shape: the input shape for the model
            # desired_shape: the 2D shape of expected samples. This is used for zeropadding or
            # limiting the feats to this shape. Once this shape, feats can be reshaped into input_shape
            # TODO test for labeled data with frames_per_sample
            if frames_per_sample is not None:
                # want smaller windows, e.g. autoencoder denoiser or speech recognition
                batch_size = math.ceil(total_rows_per_wav/frames_per_sample)
                if labeled_data:
                    input_shape = (batch_size, frames_per_sample, num_feats + 1)
                    desired_shape = (input_shape[0] * input_shape[1], 
                                     input_shape[2]-1)
                else:
                    input_shape = (batch_size, frames_per_sample, num_feats)
                    desired_shape = (input_shape[0]*input_shape[1],
                                     input_shape[2])
            else:
                if labeled_data:
                    input_shape = (int(total_rows_per_wav), num_feats + 1)
                    desired_shape = (input_shape[0], input_shape[1]-1)
                else:
                    input_shape = (int(total_rows_per_wav), num_feats)
                    desired_shape = input_shape
            # set whether or not features will include complex values:
            if 'stft' in feature_type:
                complex_vals = True
            else:
                complex_vals = False
            # limit feat_type to the basic feature extracted
            # for example:
            # feature_type 'powspec' is actually 'stft' but with complex info removed.
            # the basic feat_type is still 'stft'
            if 'mfcc' in feature_type:
                feat_type = 'mfcc'
            elif 'fbank' in feature_type:
                feat_type = 'fbank'
            elif 'stft' in feature_type:
                feat_type = 'stft'
            elif 'powspec' in feature_type:
                feat_type = 'stft'
            elif 'signal' in feature_type:
                feat_type = 'signal'
            else:
                raise TypeError('Expected '+', '.join(list_available_features())+\
                    ' to be in `feature_type`, not {}'.format(feature_type))
            for key, value in datasets_dict.items():
                # get parent directory of where data should be saved (i.e. for saving pics)
                datapath = datasets_path2save_dict[key]
                if not isinstance(datapath, pathlib.PosixPath):
                    datapath = pathlib.Path(datapath)
                datadir = datapath.parent
                # when loading a dictionary, the value is a string
                if isinstance(value, str):
                    value = pyst.utils.restore_dictvalue(value)
                extraction_shape = (len(value) * audiofile_lim,) + input_shape
                feats_matrix = pyst.dsp.create_empty_matrix(
                    extraction_shape, 
                    complex_vals=complex_vals)
                # count empty rows (if speaker doesn't have audiofile_lim data)
                empty_rows = 0
                for j, zipfile in enumerate(value):
                    if labeled_data:
                        label, zipfile = int(zipfile[0]), zipfile[1]
                    if isinstance(zipfile, str):
                        zipfile = pathlib.PosixPath(zipfile)
                    # extract `audiofile_lim` from zipfile:
                    extract_dir = pyst.utils.check_dir(extract_dir, make=True)
                    pyst.files.extract(zipfile, extract_path = extract_dir)
                    audiolist = pyst.files.collect_audiofiles(extract_dir,
                                                              recursive = True)
                    if audiofile_lim is not None:
                        for i in range(audiofile_lim):
                            if i == len(audiolist) and i < audiofile_lim:
                                print('Short number files: ', audiofile_lim - i)
                                empty_rows += audiofile_lim - i
                                break
                            feats = pyst.feats.get_feats(audiolist[i],
                                                        sr=sr,
                                                        feature_type=feat_type,
                                                        win_size_ms=win_size_ms,
                                                        percent_overlap=percent_overlap,
                                                        num_filters=num_feats,
                                                        num_mfcc=num_feats,
                                                        dur_sec=dur_sec,
                                                        **kwargs)
                            # if power spectrum (remove complex values and squaring features)
                            if 'powspec' in feature_type:
                                feats = np.abs(feats)**2
                                
                            if visualize:
                                if labeled_data:
                                    if decode_dict is not None:
                                        try:
                                            label_plot = decode_dict[label].upper()
                                        except KeyError:
                                            try:
                                                label_plot = decode_dict[str(label)].upper()
                                            except KeyError:
                                                label_plot = label
                                    else:
                                        label_plot = label
                                else:
                                    label_plot = audiofile.parent.stem.upper()
                                # visualize features:
                                if 'mfcc' in feature_type or 'signal' in feature_type:
                                    energy_scale = None
                                else:
                                    energy_scale = 'power_to_db'
                                #visualize features only every n num frames
                                if j % vis_every_n_frames == 0:
                                    save_pic_path = datadir.joinpath(
                                        'images',key,'{}_sample{}'.format(
                                            feature_type, j))
                                    # make sure this directory exists
                                    save_pic_dir = pyst.utils.check_dir(save_pic_path.parent, make=True)
                                    pyst.feats.plot(feats, 
                                                    feature_type = feature_type,
                                                    win_size_ms = win_size_ms,
                                                    percent_overlap = percent_overlap,
                                                    energy_scale = energy_scale,
                                                    title='{} {} features: label {}'.format(
                                                        key, feature_type.upper(),
                                                        label_plot),
                                                    save_pic=visualize, 
                                                    name4pic=save_pic_path)
                                
                            # zeropad feats if too short:
                            if 'signal' in feat_type:
                                feats_zeropadded = np.zeros(desired_shape)
                                feats_zeropadded = feats_zeropadded.flatten()
                                if len(feats.shape) > 1:
                                    feats_zeropadded = feats_zeropadded.reshape(feats_zeropadded.shape[0],
                                                                                feats.shape[1])
                                if len(feats) > len(feats_zeropadded):
                                    feats = feats[:len(feats_zeropadded)]
                                feats_zeropadded[:len(feats)] += feats

                                # reshape here for training models to avoid memory issues later 
                                # (while training) if total samples is large
                                feats = feats_zeropadded.reshape(desired_shape)
                            
                            feats = pyst.feats.zeropad_features(
                                feats, 
                                desired_shape = desired_shape,
                                complex_vals = complex_vals)
                            
                            if labeled_data:
                                # create label column
                                label_col = np.zeros((len(feats),1)) + label
                                feats = np.concatenate([feats,label_col], axis=1)



                                                    
                            feats = feats.reshape(extraction_shape[1:])
                            # fill in empty matrix with features from each audiofile

                            feats_matrix[j+i] = feats
                    # delete extracted data (not directories):
                    pyst.files.delete_dir_contents(extract_dir, remove_dir = False)
                    pyst.utils.print_progress(iteration = j, 
                                            total_iterations = len(value),
                                            task = '{} {} feature extraction'.format(
                                                key, feature_type))
                if empty_rows > 0:
                    print('\nFeatures have {} empty rows.\n'.format(empty_rows))
                    print(feats_matrix.shape)
                    feats_matrix = feats_matrix[:-empty_rows]
                    print('\nNow removing them:')
                    print(feats_matrix.shape)
                # save data:
                np.save(datasets_path2save_dict[key], feats_matrix)
                print('\nFeatures saved at {}\n'.format(datasets_path2save_dict[key]))
            if log_settings:
                log_filename = datadir.joinpath('log_extraction_settings.csv')
                feat_settings = dict(dur_sec=dur_sec,
                                     feature_type=feature_type,
                                     feat_type=feat_type,
                                     complex_vals=complex_vals,
                                     sr=sr,
                                     num_feats=num_feats,
                                     n_fft=n_fft,
                                     win_size_ms=win_size_ms,
                                     frame_length=frame_length,
                                     percent_overlap=percent_overlap,
                                     frames_per_sample=frames_per_sample,
                                     labeled_data=labeled_data,
                                     visualize=visualize,
                                     # different for each dataset
                                     #total_samples=total_samples, 
                                     input_shape=input_shape,
                                     desired_shape=desired_shape,
                                     use_librosa=use_librosa,
                                     center=center,
                                     mode=mode,
                                     subsection_data=subsection_data,
                                     divide_factor=divide_factor,
                                     kwargs = kwargs
                                     )
                feat_settings_path = pyst.utils.save_dict(
                    dict2save = feat_settings,
                    filename = log_filename,
                    overwrite=True)
        else:
            raise ValueError('Sorry, this functionality is not yet supported. '+\
                'Set `use_librosa` to True.')
    except MemoryError as e:
        print('MemoryError: ',e)
        print('\nSectioning data and trying again.\n')
        datasets_dict, datasets_path2save_dict = pyst.datasets.section_data(
            datasets_dict, datasets_path2save_dict, divide_factor=divide_factor)
        datasets_dict, datasets_path2save_dict = save_features_datasets_zipfiles(
            datasets_dict = datasets_dict, 
            datasets_path2save_dict = datasets_path2save_dict,
            extract_dir = extract_dir,
            audiofile_lim = audiofile_lim,
            feature_type = feature_type, 
            sr = sr, 
            n_fft = n_fft, 
            dur_sec = dur_sec,
            num_feats = num_feats,
            win_size_ms = win_size_ms,
            percent_overlap = percent_overlap,
            use_librosa = use_librosa, 
            window = window,
            center = center,
            mode = mode,
            frames_per_sample = frames_per_sample,
            visualize = visualize, 
            vis_every_n_frames = vis_every_n_frames, 
            labeled_data = labeled_data,
            log_settings = log_settings,
            decode_dict = decode_dict,
            **kwargs)
    return datasets_dict, datasets_path2save_dict

def prep_new_audiofeats(feats, desired_shape, input_shape):
    '''Prepares new audio data to feed to a pre-trained model.
    
    Parameters
    ----------
    feats : np.ndarray [shape = (num_frames, num_features)]
        The features to prepare for feeding to a model.
    
    desired_shape : tuple 
        The expected number of samples necessary to fulfill the expected
        `input_shape` for the model. The `feats` will be zeropadded or
        limited to match this `desired_shape`.
    
    input_shape : tuple 
        The `input_shape` the model expects a single sample of data to be.
        
    Returns
    -------
    feats_reshaped : np.ndarray [shape = (`input_shape`)]
        The features reshaped to what the model expects.
    '''
    feats_reshaped = pyst.feats.adjust_shape(feats, desired_shape)
    # reshape to input shape with a necessary "tensor" dimension
    feats_reshaped = feats_reshaped.reshape(input_shape+(1,))
    return feats_reshaped


def feats2audio(feats, feature_type, sr, win_size_ms,
                percent_overlap, phase=None):
    '''Prepares features into audio playable format.
    
    Parameters
    ----------
    feats : np.ndarray [shape = (num_frames, num_feats)]
        If the features are a signal, 
        [size = (batch_size * num_frames * num_features, 1)]. 
        Otherwise [size = (batch_size * num_frames, num_features)].
    feature_type : str
        Either 'stft', 'fbank', 'signal', or 'mfcc'. For the 'signal'
        feature, only mono channel is supported.
    sr : int 
        Sampling rate that the features were extracted with
    win_size_ms : int 
        The window size in milliseconds the features were extracted with
    percent_overlap : float
        The percent overlap between windows.
    phase : np.ndarray [shape = (num_frames, num_feats)], optional
        The original phase information of the reconstructed signal.
        
    Returns
    -------
    y : np.ndarray [shape = (num_samples, )]
        The reconstructed signal in samples.
    '''
    # (default) librosa handles data in shape (num_feats, num_frames)
    # while pysoundtool works with data in shape (num_frames, num_feats)
    if phase is not None:
        try:
            assert feats.shape == phase.shape
        except AssertionError:
            raise ValueError('Expected `feats` (shape {})'.format(feats.shape)+\
                ' and `phase` (shape {}) '.format(phase.shape) +\
                    'to have the same shape: (num_frames, num_features)')
    window_shift = win_size_ms * percent_overlap
    if 'signal' not in feature_type:
        # Will apply Librosa package to feats. Librosa expects data to have
        # shape (num_features, num_frames) not (num_frames, num_features)
        feats = feats.T 
        if phase is not None:
            phase = phase.T
    if 'fbank' in feature_type:
        y = librosa.feature.inverse.mel_to_audio(
            feats, 
            sr=sr, 
            n_fft = int(win_size_ms*0.001*sr), 
            hop_length=int(window_shift*0.001*sr))
    elif 'mfcc' in feature_type:
        feats = feats[:14,:]
        y = librosa.feature.inverse.mfcc_to_audio(
            feats, 
            sr=sr, 
            n_fft = int(win_size_ms*0.001*sr), 
            hop_length=int(window_shift*0.001*sr),
            n_mels=13)
    elif 'stft' in feature_type or 'powspec' in feature_type:
        # can use istft with phase information applied
        if phase is not None:
            feats = feats * phase
            y = librosa.istft(
                feats,
                hop_length=int(window_shift*0.001*sr),
                win_length = int(win_size_ms*0.001*sr))
        # if no phase information available:
        else:
            y = librosa.griffinlim(
                feats,
                hop_length=int(window_shift*0.001*sr),
                win_length = int(win_size_ms*0.001*sr))
    elif 'signal' in feature_type:
        y = feats.flatten()
    return y

def grayscale2color(image_matrix, colorscale=3):
    '''Expects grayscale image. Copies first channel into additional channels.
    
    This is useful for pre-trained models that require features
    to have rgb channels, not grayscale. Assumes last channel the colorscale 
    column.
    '''
    if len(image_matrix.shape) == 2:
        # if colorscale column not there, adds it
        image_matrix = image_matrix.reshape(image_matrix.shape + (1,))
    expected_shape = image_matrix.shape[:-1] + (colorscale,)
    # create extra empty channels to copy gray image to it:
    image_zeropadded = pyst.feats.zeropad_features(image_matrix, expected_shape)
    for i in range(colorscale):
        if i == 0:
            pass
        else:
            if len(image_zeropadded.shape) == 3:
                image_zeropadded[:,:,i] = image_zeropadded[:,:,0]
            elif len(image_zeropadded.shape) == 4:
                image_zeropadded[:,:,:,i] = image_zeropadded[:,:,:,0]
            elif len(image_zeropadded.shape) == 5:
                image_zeropadded[:,:,:,:,i] = image_zeropadded[:,:,:,:,0]
            elif len(image_zeropadded.shape) == 6:
                image_zeropadded[:,:,:,:,:,i] = image_zeropadded[:,:,:,:,:,0]
            else:
                raise ValueError('This function expects between 2 and 6 dimensions, '\
                    'not {} dimensions'.format(len(image_matrix.shape)))
    return image_zeropadded
    

if __name__ == "__main__":
    import doctest
    doctest.testmod()
