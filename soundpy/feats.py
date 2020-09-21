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

import collections
import numpy as np
import math
import random
import matplotlib
import librosa
from scipy.signal import hann, hamming
from scipy.fftpack import dct
import pathlib
from python_speech_features import fbank, mfcc, delta
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import soundpy as sp


# TODO Clean up   
# stereo sound only for plotting 'signal'; NOT for frequency features.
def plot(feature_matrix, feature_type, 
        save_pic=False, name4pic=None, energy_scale='power_to_db',
        title=None, sr=None, win_size_ms=None, percent_overlap=None,
        x_label=None, y_label=None, subprocess=False, overwrite=False):
    '''Visualize feature extraction; frames on x axis, features on y axis. Uses librosa to scale the data if scale applied.
    
    Note: can only take multiple channels if `feature_type` is 'signal'. For other 
    feature types, the plot will not work as expected.
    
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
        True to save image as .png; False to just plot it. If `subprocess` is 
        True, `save_pic` will automatically be set to True.

    name4pic : str, optional
        If `save_pic` set to True, the name the image should be saved under.
        
    energy_scale : str, optional
        If features need to be adjusted, e.g. from power to decibels. 
        Default is 'power_to_db'.
        
    title : str, optional
        The title for the graph. If None, `feature_type` is used.

    sr : int, optional
        Useful in plotting the time for features.
        
    win_size_ms : int, float, optional
        Useful in plotting the time for features in the frequency domain (e.g. 
        STFT, FBANK, MFCC features)
        
    percent_overlap : int, float, optional
        Useful in plotting the time for features in the frequency domain (e.g. 
        STFT, FBANK, MFCC features)
        
    x_label : str, optional 
        The label to be applied to the x axis. 
        
    y_label : str, optional 
        The label to be applied to the y axis.
        
    subprocess : bool 
        If `subprocess` is True, matplotlib will use backend 'Agg', which only allows plots to be saved.
        If `subprocess` is False, the default backend 'TkAgg' will be used, which allows plots to be 
        generated live as well as saved. The 'Agg' backend is useful if one wants to visualize sound
        while a main process is being performed, for example, while a model is being trained.
        (default False)
        
    overwrite : bool 
        If False, if .png file already exists under given name, a date tag will be 
        added to the .png filename to avoid overwriting the file. 
        (default False)
        
    Returns
    -------
    None
    '''
    if not subprocess:
        # can show plots
        # interferes with training models though
        matplotlib.use('TkAgg')
    else:
        # does not interfere with training models
        matplotlib.use('Agg')
        if save_pic is False:
            import warnings
            save_pic = True
            if name4pic is None:
                location = 'current working directory'
            else:
                location = name4pic
            msg = 'Due to matplotlib using AGG backend, cannot display plot. '+\
                'Therefore, the plot will be saved here: {}'.format(location)
            warnings.warn(msg)
    import matplotlib.pyplot as plt
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
            time_sec = sp.dsp.get_time_points(dur_sec, sr)
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
        x_axis_label += ' across {} channel(s)'.format(feature_matrix.shape[1])
    else:
        plt.pcolormesh(feature_matrix.T)
        plt.colorbar(label=energy_label)
    plt.xlabel(x_axis_label)
    plt.ylabel(axis_feature_label)
    # if feature_matrix has multiple frames, not just one
    if feature_matrix.shape[1] > 1 and 'signal' not in feature_type: 
        if win_size_ms is not None and percent_overlap is not None:
            # the xticks basically show time but need to be multiplied by 0.01
            plt.xlabel('Time (sec)') 
            locs, labels = plt.xticks()
            if percent_overlap == 0:
                new_labels=[str(round(i*0.001*win_size_ms,1)) for i in locs]
            else:
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
        outputname = name4pic or 'visualize{}feats_{}'.format(feature_type.upper(),
                                                              sp.utils.get_date())
        outputname = sp.utils.string2pathlib(outputname)
        if outputname.suffix:
            if outputname.suffix != '.png':
                # add .png as extension
                fname = outputname.name + '.png'
                outputname = outputname.parent.joinpath(fname)
        else:
            fname = outputname.stem + '.png'
            outputname = outputname.parent.joinpath(fname)
        if not overwrite:
            if os.path.exists(outputname):
                fname = outputname.stem
                fname += '_'+sp.utils.get_date()
                outputname = outputname.parent.joinpath(fname+outputname.suffix)
        plt.savefig(outputname)
    else:
        plt.show()

# tested for stereo sound
def plotsound(audiodata, feature_type='fbank', win_size_ms = 20, \
    percent_overlap = 0.5, fft_bins = None, num_filters=40, num_mfcc=40, sr=None,\
        save_pic=False, name4pic=None, energy_scale='power_to_db', mono=None, real_signal=False, **kwargs):
    '''Visualize feature extraction depending on set parameters. 
    
    Stereo sound can be graphed. If `feature_type` is 'signal', all channels will be 
    graphed on same plot. Otherwise, each channel will be plotted separately.
    
    Parameters
    ----------
    audiodata : str, numpy.ndarray [size=(num_samples,) or (num_samples, num_channels)]
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
        Keyword arguments for soundpy.feats.plot
    '''
    percent_overlap = check_percent_overlap(percent_overlap)
    if 'signal' not in feature_type:
        if isinstance(audiodata, np.ndarray) and len(audiodata.shape) > 1:
            for channel in range(audiodata.shape[1]):
                if name4pic is None:
                    name4pic = '{}_channel_{}'.format(feature_type, channel+1)
                else:
                    name4pic = sp.string2pathlib(name4pic)
                    name = name4pic.stem
                    if channel == 0:
                        name += '_channel{}'.format(channel+1)
                    else:
                        name = name[:-1]+'{}'.format(channel+1)
                    name4pic = name4pic.parent.joinpath(name+name4pic.suffix)
                if 'title' not in kwargs:
                    kwargs['title'] = '{} features\n(channel {})'.format(
                        feature_type, channel+1)
                else:
                    if channel == 0:
                        kwargs['title'] += '\n(channel {})'.format(channel+1)
                    else:
                        kwargs['title'] = kwargs['title'][:-2]+'{})'.format(
                            channel+1)
                feats = sp.feats.get_feats(
                    audiodata[:,channel], feature_type=feature_type, 
                    win_size_ms = win_size_ms, percent_overlap = percent_overlap,
                    fft_bins = fft_bins, num_filters=num_filters, num_mfcc = num_mfcc,
                    sr=sr, mono = mono, real_signal = real_signal)
                sp.feats.plot(
                    feats, feature_type=feature_type, sr=sr,
                    save_pic = save_pic, name4pic=name4pic, 
                    energy_scale = energy_scale,
                    win_size_ms = win_size_ms, percent_overlap = percent_overlap,
                    **kwargs)
            return None
    feats = sp.feats.get_feats(audiodata, feature_type=feature_type, 
                      win_size_ms = win_size_ms, percent_overlap = percent_overlap,
                      fft_bins = fft_bins, num_filters=num_filters, num_mfcc = num_mfcc,
                      sr=sr, mono = mono, real_signal = real_signal)
    sp.feats.plot(feats, feature_type=feature_type, sr=sr,
                    save_pic = save_pic, name4pic=name4pic, energy_scale = energy_scale,
                    win_size_ms = win_size_ms, percent_overlap = percent_overlap,
                    **kwargs)

# stereo sound with mono (True/False) works for 'signal' data
# only mono for frequency features
def get_feats(sound,
              sr = None, 
              feature_type = 'fbank', 
              win_size_ms = 20, 
              percent_overlap = 0.5,
              window = 'hann',
              fft_bins = None,
              num_filters = None,
              num_mfcc = None,
              remove_first_coefficient = False,
              sinosoidal_liftering = False,
              dur_sec = None,
              mono = None,
              rate_of_change = False,
              rate_of_acceleration = False,
              subtract_mean = False,
              real_signal = True,
              fmin = None, 
              fmax = None,
              zeropad = True):
    '''Collects raw signal data, stft, fbank, or mfcc features.
    
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
    
    fft_bins : int  
        Number of frequency bins to apply in fast Fourier transform. (default None) 
    
    num_filters : int
        Number of mel-filters to be used when applying mel-scale. For 
        'fbank' features, 20-128 are common, with 40 being very common. If None, will 
        be set to 40.
        (default None)
    
    num_mfcc : int
        Number of mel frequency cepstral coefficients. First coefficient
        pertains to loudness; 2-13 frequencies relevant for speech; 13-40
        for acoustic environment analysis or non-linguistic information.
        If None, will be set to `num_filters` or 40.
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
        
    Returns
    -------
    feats : tuple (num_samples, sr) or np.ndarray [size (num_frames, num_filters) dtype=np.float or np.complex]
        Feature data. If `feature_type` is 'signal', returns a tuple containing samples and sampling rate. If `feature_type` is of another type, returns np.ndarray with shape (num_frames, num_filters/features)
    '''
    # load data
    if isinstance(sound, str) or isinstance(sound, pathlib.PosixPath):
        if mono is None:
            mono = True
        data, sr = sp.loadsound(sound, sr = sr, dur_sec = dur_sec, mono = mono)
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
        if len(data.shape) > 1 and 'signal' not in feature_type:
            print('Only one channel can be currently used for feature '+\
                'extraction. Using the first channel.')
            data = data[:,0]
        elif len(data.shape) > 1 and mono:
            data = data[:,0]
        if dur_sec:
            data = data[:int(sr*dur_sec)]
    
    if isinstance(data, np.ndarray):
        if not np.isfinite(data).all():
            raise TypeError('NAN values found in loaded sound samples.')
    else:
        raise TypeError('Data is type {} and '.format(type(data))+\
            'a numpy.ndarray was expected.')
    # ensure percent overlap is between 0 and 1
    percent_overlap = check_percent_overlap(percent_overlap)
    win_shift_ms = win_size_ms - (win_size_ms * percent_overlap)
    if win_shift_ms <= 0:
        raise ValueError('`percent_overlap` {} is too high. '.format(percent_overlap)+\
            'The signal cannot be processed with 0 or negative window shift / hop length.')
    if fft_bins is None:
        # base on frame length / window: larger windows --> higher freq resolution
        fft_bins = int(win_size_ms * sr // 1000)
    if 'stft' in feature_type or 'powspec' in feature_type:
        feats = sp.feats.get_stft(
            sound = data, 
            sr = sr, 
            win_size_ms = win_size_ms,
            percent_overlap = percent_overlap,
            real_signal = real_signal,
            fft_bins = fft_bins,
            window = window,
            zeropad = zeropad
            )
    elif 'fbank' in feature_type:
        if num_filters is None:
            num_filters = 40
        feats = sp.feats.get_fbank(
            sound = data, 
            sr = sr, 
            num_filters = num_filters,
            win_size_ms = win_size_ms,
            percent_overlap = percent_overlap,
            real_signal = real_signal,
            fft_bins = fft_bins,
            window = window, 
            zeropad = zeropad
            )
        
    elif 'mfcc' in feature_type:
        if num_mfcc is None:
            if num_filters is not None:
                num_mfcc = num_filters
            else:
                num_mfcc = 40
        if num_filters is None:
            num_filters = 40
        feats = sp.feats.get_mfcc(
            sound = data, 
            sr = sr, 
            num_mfcc = num_mfcc,
            remove_first_coefficient = remove_first_coefficient,
            sinosoidal_liftering = sinosoidal_liftering,
            num_filters = num_filters,
            win_size_ms = win_size_ms,
            percent_overlap = percent_overlap,
            real_signal = real_signal,
            fft_bins = fft_bins,
            window = window, 
            zeropad = zeropad
            )
    elif 'signal' in feature_type:
        if data.dtype == np.complex128 or data.dtype == np.complex64:
            import Warnings
            msg = '\nWARNING: real raw signal features are being generated '+\
                'from a STFT matrix.'
            Warnings.warn(msg)
            feats = sp.feats.feats2audio(data, feature_type = 'stft',
                                            sr=sr, 
                                            win_size_ms = win_size_ms,
                                            percent_overlap = percent_overlap)
        else:
            feats = data
        
    # TODO test difference between python_speech_features and librosa
    if not 'signal' in feature_type:
        if subtract_mean is True:
            feats -= (np.mean(feats, axis=0) + 1e-8)
        if rate_of_change is True:
            #d1 = delta(feats, N=2)
            d1 = librosa.feature.delta(feats.T).T
            feats = np.concatenate((feats, d1), axis=1)
        if rate_of_acceleration is True:
            #d2 = delta(delta(feats, N=2), N=2)
            d2 = librosa.feature.delta(feats.T, order=2).T
            feats = np.concatenate((feats, d2), axis=1)

    if 'powspec' in feature_type:
        feats = sp.dsp.calc_power(feats)
    return feats


def load_feat_settings(feat_settings_dict):
    '''Loads feature settings into named tuple. Sets defaults if not present.
    TODO: test w previous version
    '''
    FeatureExtractionSettings = collections.namedtuple('FeatureExtractionSettings',
                                          ['sr', 'feature_type', 'win_size_ms',
                                           'percent_overlap', 'window', 'dur_sec',
                                           'num_filters','num_mfcc', 'fft_bins',
                                           'remove_first_coefficient','sinosoidal_liftering',
                                           'mono', 'rate_of_change', 'rate_of_acceleration',
                                           'subtract_mean', 'real_signal', 'fmin', 'fmax',
                                           'zeropad', 'input_shape', 'base_shape',
                                           'num_feats'])    

    # newer version of soundpy: 0.1.0a3
    # store get_feats kwargs under `kwargs` key
    if 'kwargs' in feat_settings_dict:
        kwargs = sp.utils.restore_dictvalue(feat_settings_dict['kwargs'])
        feat_settings_dict.update(kwargs)
    # if values saved as strings and should be a list or tuple, restore them to original type: 
    # see `soundpy.utils.restore_dictvalue`
    sr = sp.utils.restore_dictvalue(
        feat_settings_dict['sr'])
    feature_type = sp.utils.restore_dictvalue(
        feat_settings_dict['feature_type'])
    win_size_ms = sp.utils.restore_dictvalue(
        feat_settings_dict['win_size_ms'])
    percent_overlap = sp.utils.restore_dictvalue(
        feat_settings_dict['percent_overlap'])
    try:
        window = sp.utils.restore_dictvalue(feat_settings_dict['window'])
    except KeyError:
        # set default here...
        window = 'hann'
    dur_sec = sp.utils.restore_dictvalue(
        feat_settings_dict['dur_sec'])
    try: 
        num_filters = sp.utils.restore_dictvalue(feat_settings_dict['num_filters'])
    except KeyError:
        num_filters = None
    try: 
        num_mfcc = sp.utils.restore_dictvalue(feat_settings_dict['num_mfcc'])
    except KeyError:
        num_mfcc = None
    try:
        # newer version soundpy 0.1.0v3
        fft_bins = sp.utils.restore_dictvalue(feat_settings_dict['fft_bins'])
    except KeyError:
        # older version soundpy 0.1.0v2
        fft_bins = sp.utils.restore_dictvalue(feat_settings_dict['n_fft'])
    try: 
        remove_first_coefficient = sp.utils.restore_dictvalue(
            feat_settings_dict['remove_first_coefficient'])
    except KeyError:
        remove_first_coefficient = False
    try: 
        sinosoidal_liftering = sp.utils.restore_dictvalue(
            feat_settings_dict['sinosoidal_liftering'])
    except KeyError:
        sinosoidal_liftering = False
    try: 
        mono = sp.utils.restore_dictvalue(
            feat_settings_dict['mono'])
    except KeyError:
        mono = True # default setting
    try: 
        rate_of_change = sp.utils.restore_dictvalue(
            feat_settings_dict['rate_of_change'])
    except KeyError:
        rate_of_change = False
    try: 
        rate_of_acceleration = sp.utils.restore_dictvalue(
            feat_settings_dict['rate_of_acceleration'])
    except KeyError:
        rate_of_acceleration = False
    try: 
        subtract_mean = sp.utils.restore_dictvalue(
            feat_settings_dict['subtract_mean'])
    except KeyError:
        subtract_mean = False
    try: 
        real_signal = sp.utils.restore_dictvalue(
            feat_settings_dict['real_signal'])
    except KeyError:
        real_signal = True
    try: 
        fmin = sp.utils.restore_dictvalue(
            feat_settings_dict['fmin'])
    except KeyError:
        fmin = None
    try: 
        fmax = sp.utils.restore_dictvalue(
            feat_settings_dict['fmax'])
    except KeyError:
        fmax = None
    try: 
        zeropad = sp.utils.restore_dictvalue(
            feat_settings_dict['zeropad'])
    except KeyError:
        zeropad = True
    try:
        # older version of soundpy: 0.1.0a2
        input_shape = sp.utils.restore_dictvalue(
            feat_settings_dict['input_shape'])
    except KeyError:
        # newer version of soundpy: 0.1.0a3
        input_shape = sp.utils.restore_dictvalue(
            feat_settings_dict['feat_model_shape'])
    try:
        # older version of soundpy: 0.1.0a2
        base_shape = sp.utils.restore_dictvalue(
            feat_settings_dict['desired_shape'])
    except KeyError:
        # newer version of soundpy: 0.1.0a3
        base_shape = sp.utils.restore_dictvalue(
            feat_settings_dict['feat_base_shape'])
    try:
        # older version of soundpy: 0.1.0a2
        num_feats = sp.utils.restore_dictvalue(
            feat_settings_dict['num_feats'])
    except KeyError:
        # newer version of soundpy: 0.1.0a3
        num_feats = base_shape[-1]
        
    featsettings = FeatureExtractionSettings(
        sr = sr,
        feature_type = feature_type, 
        win_size_ms = win_size_ms,
        percent_overlap = percent_overlap,
        window = window,
        dur_sec = dur_sec, 
        num_filters = num_filters,
        num_mfcc = num_mfcc,
        fft_bins = fft_bins,
        remove_first_coefficient = remove_first_coefficient,
        sinosoidal_liftering = sinosoidal_liftering,
        mono = mono, 
        rate_of_change = rate_of_change,
        rate_of_acceleration = rate_of_acceleration,
        subtract_mean = subtract_mean,
        real_signal = real_signal,
        fmin = fmin, 
        fmax = fmax, 
        zeropad = zeropad,
        input_shape = input_shape,
        base_shape = base_shape,
        num_feats = num_feats)
    return featsettings

# allows for more control over fft bins / resolution of each iteration.
def get_stft(sound, sr=22050, win_size_ms = 50, percent_overlap = 0.5,
                real_signal = False, fft_bins = 1024, 
                window = 'hann', zeropad = True, **kwargs):
    '''Returns short-time Fourier transform matrix.
    
    This function allows more flexibility in number of `fft_bins` and `real_signal`
    settings. Additionally, this does not require the package librosa, making it 
    a bit easier to manipulate if desired. For an example, see
    `soundpy.augment.vtlp`.
    
    Parameters
    ----------
    sound : np.ndarray [shape=(num_samples,) or (num_samples, num_channels)], str, or pathlib.PosixPath
        If type np.ndarray, expect raw samples in mono or stereo sound. If type str or 
        pathlib.PosixPath, expect pathway to audio file.
        
    sr : int 
        The sample rate of `sound`. 
    
    win_size_ms : int, float 
        Window length in milliseconds for Fourier transform to be applied
        (default 50)
        
    percent_overlap : int or float 
        Amount of overlap between processing windows. For example, if `percent_overlap`
        is set at 0.5, the overlap will be half that of `win_size_ms`. (default 0.5) 
        If an integer is provided, it will be converted to a float between 0 and 1.
        
    real_signal : bool 
        If True, only half the FFT spectrum will be used; there should really be no difference
        as the FFT is symmetrical. If anything, setting `real_signal` to True may speed up 
        functionality / make functions more efficient.
        
    fft_bins : int 
        Number of frequency bins to use when applying fast Fourier Transform. (default 1024)
        
    window : str 
        The window function to apply to each window segment. Options are 'hann' and 'hamming'.
        (default 'hann')
        
    zeropad : bool 
        If True, samples will be zeropadded to fill any partially filled window. If False, the 
        samples constituting the partially filled window will be cut off.
        
    **kwargs : additional keyword arguments
        Keyword arguments for `soundpy.files.loadsound`.
        
    Returns
    -------
    stft_matrix : np.ndarray[size=(num_frames, fft_bins)]
    '''
    if isinstance(sound, np.ndarray):
        if sound.dtype == np.complex_:
            import warnings
            msg = '\nWARNING: data provided to `soundpy.feats.get_stft` is already'+\
                ' a STFT matrix. Returning original data.'
            warnings.warn(msg)
            return sound
        data = sound
    else:
        data, sr2 = sp.loadsound(sound, sr=sr, **kwargs)
        assert sr2 == sr
    frame_length = sp.dsp.calc_frame_length(win_size_ms, sr)
    num_overlap_samples = int(frame_length * percent_overlap)
    num_subframes = sp.dsp.calc_num_subframes(len(data),
                                                frame_length = frame_length,
                                                overlap_samples = num_overlap_samples,
                                                zeropad = zeropad)
    
    if fft_bins is None:
        fft_bins = int(win_size_ms * sr // 1000)
    total_rows = fft_bins // 2 + 1
    # if mono, only one channel; otherwise match num channels in sound signal
    if sp.dsp.ismono(data):
        stft_matrix = sp.dsp.create_empty_matrix(
            (num_subframes, total_rows),
            complex_vals = True)
    else:
        stft_matrix = sp.dsp.create_empty_matrix(
            (num_subframes, total_rows, data.shape[1]),
            complex_vals = True)
    section_start = 0
    window_frame = sp.dsp.create_window(window, frame_length)
    for frame in range(num_subframes):
        section = data[section_start:section_start+frame_length]
        section = sp.dsp.apply_window(section, 
                                        window_frame, 
                                        zeropad = zeropad)
        
        section_fft = sp.dsp.calc_fft(section, 
                                        real_signal = real_signal,
                                        fft_bins = fft_bins,
                                        )
        stft_matrix[frame] = section_fft[:total_rows]
        section_start += (frame_length - num_overlap_samples)
    return stft_matrix

def get_fbank(sound, sr, num_filters, fmin=None, fmax=None, fft_bins = None, **kwargs):
    '''Extract mel-filterbank energy features from audio.
    
    Parameters
    ----------
    sound : np.ndarray [size=(num_samples,) or (num_samples, num_features)], str, or pathlib.PosixPath
        Sound in raw samples, a power spectrum, or a short-time-fourier-transform. If type string or pathlib.PosixPath, expect pathway to audio file.
        
    sr : int 
        The sample rate of `sound`.
        
    num_filters : int 
        The number of mel-filters to use when extracting mel-filterbank energies.
        
    fmin : int or float, optional
        The minimum frequency of interest. If None, will be set to 0. (default None)
    
    fmax : int or float, optional
        The maximum frequency of interst. If None, will be set to half of `sr`.
        (default None)
        
    fft_bins : int, optional
        The number of frequency bins / fast Fourier transform bins used in calculating 
        the fast Fourier transform. If None, set depending on type of parameter `sound`.
        If `sound` is a raw signal or audio pathway, `fft_bins` will be set to 1024;
        if `sound` is a STFT or power spectrum, `fft_bins` will be set to 2 * length
        of `sound` feature column, or 2 * sound.shape[1].
        
    **kwargs : additional keyword arguments
        Keyword arguments for `soundpy.feats.get_stft`.
        
    Returns
    -------
    fbank : np.ndarray [shape=(num_samples, num_filters)]
        The mel-filterbank energeis extracted. The number of samples depends on 
        the parameters applied in `soundpy.feats.get_stft`.
    
    References
    ----------
    Fayek, H. M. (2016). Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What’s In-Between. Retrieved from https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    '''
    if isinstance(sound, np.ndarray):
        if sound.dtype == np.complex64 or sound.dtype == np.complex128:
            stft = True
        # probably a power spectrum without complex values...
        # TODO improve
        elif len(sound.shape) > 1 and sound.shape[1] > sound.shape[0]:
            stft = True
        else:
            stft = False
    else:
        # probably a pathway - load in get_stft
        stft = False
    if fmin is None:
        fmin = 0
    if fmax is None:
        fmax = sr/2

    mel_points = sp.dsp.fbank_filters(fmin, fmax, num_filters = num_filters)
    hz_points = sp.dsp.mel_to_hz(mel_points)
    
    if fft_bins is None:
        if stft is True:
            # assumes number of fft bins is the length of second column
            # https://librosa.org/doc/latest/generated/librosa.istft.html?highlight=istf#librosa.istft
            fft_bins = (sound.shape[1]-1) * 2 
        else:
            try:
                fft_bins = int(kwargs['win_size_ms'] * sr // 1000)
            except KeyError:
                fft_bins = 512
        
    freq_bins = np.floor((fft_bins + 1) * hz_points / sr)
    
    if stft:
        # use number of fft columns in stft as reference
        fbank = np.zeros((num_filters, sound.shape[1]))
    else:
        fbank = np.zeros((num_filters, int(np.floor(fft_bins / 2 + 1))))
    for m in range(1, num_filters + 1):
        f_m_minus = int(freq_bins[m - 1]) # left
        f_m = int(freq_bins[m]) # center
        f_m_plus = int(freq_bins[m + 1]) # right
        
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - freq_bins[m - 1]) / (freq_bins[m] - freq_bins[m -1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (freq_bins[m + 1] - k) / (freq_bins[m + 1] - freq_bins[m])
    if stft:
        if np.min(sound) < 0:
            powspec = sp.dsp.calc_power(sound)
        else:
            powspec = sound
    else:
        sound_stft = sp.feats.get_stft(sound, sr=sr, fft_bins = fft_bins, **kwargs)
        powspec = sp.dsp.calc_power(sound_stft)

    filter_banks = np.dot(powspec, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks) # numerical stability
    return filter_banks

def get_mfcc(sound, sr, num_mfcc, remove_first_coefficient=False, 
             sinosoidal_liftering = False, **kwargs):
    '''Extracts mel-frequency cepstral coefficients from audio.
    
    Parameters
    ----------
    sound : np.ndarray [size=(num_samples,) or (num_samples, num_features)] or str or pathlib.PosixPath
        If `sound` is a np.ndarray, expected as raw samples, a power spectrum or a
        short-time Fourier transform. If string or pathlib.PosixPath, should be the pathway
        to the audio file.
        
    sr : int 
        The sample rate of the `sound`.
        
    num_mfcc : int 
        The number of mel-frequency cepstral coefficients
        
    remove_first_coefficient : bool
        If True, the first coefficient, representing amplitude or volume of signal, is
        removed. Found to sometimes improve automatic speech recognition. 
        (default False)
        
    sinosoidal_liftering : bool 
        If True, reduces influence of higher coefficients, found to aid in handling 
        noise in background in automatic speech recognition. (default False)
        
    **kwargs : additional keyword arguments
        Keyword arguments for soundpy.feats.get_fbank()
        
    References
    ----------
    Fayek, H. M. (2016). Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What’s In-Between. Retrieved from https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    '''
    fbank = sp.feats.get_fbank(sound, sr=sr, **kwargs)
    mfcc = dct(fbank, type=2, axis=1, norm='ortho')
    if remove_first_coefficient is True:
        mfcc = mfcc[:,1:num_mfcc]
    else:
        mfcc = mfcc[:,:num_mfcc]
    return mfcc

def get_vad_stft(sound, sr=48000, win_size_ms = 50, percent_overlap = 0.5,
                          real_signal = False, fft_bins = 1024, 
                          window = 'hann', use_beg_ms = 120,
                          extend_window_ms = 0, energy_thresh = 40, 
                          freq_thresh = 185, sfm_thresh = 5, 
                          zeropad = True, **kwargs):
    '''Returns STFT matrix and VAD matrix. STFT matrix contains only VAD sections.
    
    Parameters
    ----------
    sound : str or numpy.ndarray [size=(num_samples,) or (num_samples, num_channels)]
        If str, wavfile (must be compatible with scipy.io.wavfile). Otherwise 
        the samples of the sound data. Note: in the latter case, `sr`
        must be declared.
    
    sr : int, optional
        The sample rate of the sound data or the desired sample rate of
        the wavfile to be loaded. (default None)
    
    win_size_ms : int or float
        Window length in milliseconds for Fourier transform to be applied
        (default 50)
    
    percent_overlap : int or float 
        Amount of overlap between processing windows. For example, if `percent_overlap`
        is set at 0.5, the overlap will be half that of `win_size_ms`. (default 0.5) 
        If an integer is provided, it will be converted to a float between 0 and 1. 
        
    real_signal : bool 
        If True, only half the FFT spectrum will be used; there should really be no difference
        as the FFT is symmetrical. If anything, setting `real_signal` to True may speed up 
        functionality / make functions more efficient.

    fft_bins : int 
        Number of frequency bins to use when applying fast Fourier Transform. (default 1024)
        
    window : str 
        The window function to apply to each window segment. Options are 'hann' and 'hamming'.
        (default 'hann')
        
    use_beg_ms : int 
        The amount of time in milliseconds to use from beginning of signal to estimate background
        noise.
        
    extend_window_ms : int 
        The amount of time in milliseconds to pad or extend the identified VAD segments. This 
        may be useful to include more speech / sound, if desired.
        
    energy_thresh : int 
        The threshold to set for measuring energy for VAD in the signal. (default 40)
        
    freq_thresh : int 
        The threshold to set for measuring frequency for VAD in the signal. (default 185)
        
    sfm_thresh : int 
        The threshold to set for measuring spectral flatness for VAD in the signal. (default 5)
        
    zeropad : bool 
        If True, samples will be zeropadded to fill any partially filled window. If False, the 
        samples constituting the partially filled window will be cut off.
    
    **kwargs : additional keyword arguments
        Keyword arguments for `soundpy.files.loadsound`
        
    Returns
    -------
    stft_matrix : np.ndarray [size=(num_frames_vad, fft_bins//2+1), dtype=np.complex_]
        The STFT matrix frames of where voice activity has been detected.
        
    vad_matrix_extwin : np.ndarray [size=(num_frames,)]
        A vector containing indices of the full STFT matrix for frames of where voice activity 
        was detected or not.
    '''
    # raise ValueError if percent_overlap is not supported
    if percent_overlap != 0 and percent_overlap < 0.5:
        raise ValueError('For this VAD function, `percent_overlap` ' +\
            'set to {} is not currently supported.\n'.format(percent_overlap) +\
                'Suggested to set at either 0 or 0.5')
    if percent_overlap > 0.5:
        import warnings
        msg = '\nWarning: for this VAD function, parameter `percent_overlap` has most success '+\
            'when set at 0 or 0.5'
    # raise warnings if sample rate lower than 44100 Hz
    if sr < 44100:
        import warnings
        msg = '\nWarning: voice-activity-detection works best with sample '+\
            'rates above 44100 Hz. Current `sr` set at {}.'.format(sr)
        warnings.warn(msg)
    if isinstance(sound, np.ndarray):
        data = sound.copy()
    else:
        data, sr2 = sp.loadsound(sound, sr=sr, **kwargs)
        assert sr2 == sr
    frame_length = sp.dsp.calc_frame_length(win_size_ms, sr)
    num_overlap_samples = int(frame_length * percent_overlap)
    num_subframes = sp.dsp.calc_num_subframes(len(data),
                                                frame_length = frame_length,
                                                overlap_samples = num_overlap_samples,
                                                zeropad = zeropad)
    
    # set number of subframes for extending window
    extwin_num_samples = sp.dsp.calc_frame_length(extend_window_ms, sr)
    num_win_subframes = sp.dsp.calc_num_subframes(extwin_num_samples,
                                                    frame_length = frame_length,
                                                    overlap_samples = num_overlap_samples,
                                                    zeropad = zeropad)
    
    total_rows = fft_bins
    if len(data.shape) > 1 and data.shape[1] > 1:
        stereo = True
        stft_matrix = sp.dsp.create_empty_matrix(
            (num_subframes, total_rows, data.shape[1]),
            complex_vals = True)
        # stereo sound --> average out channels for measuring energy
        data_vad = sp.dsp.average_channels(data)
    else:
        stereo = False
        stft_matrix = sp.dsp.create_empty_matrix(
            (num_subframes, total_rows),
            complex_vals = True)
        data_vad = data
    
    vad_matrix, (sr, e, f, sfm) = sp.dsp.vad(data_vad, sr, 
                                               win_size_ms = win_size_ms,
                                               percent_overlap = percent_overlap, 
                                               use_beg_ms = use_beg_ms,
                                               energy_thresh = energy_thresh, 
                                               freq_thresh = freq_thresh, 
                                               sfm_thresh = sfm_thresh)
    vad_matrix_extwin = vad_matrix.copy()
    
    # extend VAD windows where VAD indicated
    if extend_window_ms > 0:
        for i, row in enumerate(vad_matrix):
            if row > 0:
                # label samples before VAD as VAD
                if i > num_win_subframes:
                    vad_matrix_extwin[i-num_win_subframes:i] = 1
                else:
                    vad_matrix_extwin[:i] = 1
                # label samples before VAD as VAD
                if i + num_win_subframes < len(vad_matrix):
                    vad_matrix_extwin[i:num_win_subframes+i] = 1
                else:
                    vad_matrix_extwin[i:] = 1
                    
    section_start = 0
    extra_rows = 0
    window_frame = sp.dsp.create_window(window, frame_length)
    row = 0
    for frame in range(num_subframes):
        vad = vad_matrix_extwin[frame]
        if vad > 0:
            section = data[section_start:section_start+frame_length]
            section = sp.dsp.apply_window(section, 
                                            window_frame, 
                                            zeropad = zeropad)
            section_fft = sp.dsp.calc_fft(section, 
                                            real_signal = real_signal,
                                            fft_bins = total_rows,
                                            )
            stft_matrix[row] = section_fft
            row += 1
        else:
            extra_rows += 1
        section_start += (frame_length - num_overlap_samples)
    stft_matrix = stft_matrix[:-extra_rows]
    return stft_matrix[:,:fft_bins//2+1], vad_matrix_extwin

def get_stft_clipped(samples, sr, win_size_ms = 50, percent_overlap = 0.5, 
                     extend_window_ms = 0,  window = 'hann', zeropad = True, **kwargs):
    '''Returns STFT matrix and VAD matrix with beginning and ending silence removed.
    
    Parameters
    ----------
    samples : str or numpy.ndarray [size=(num_samples,) or (num_samples, num_channels)]
        If str, wavfile (must be compatible with scipy.io.wavfile). Otherwise 
        the samples of the sound data. 
    
    sr : int, optional
        The sample rate of the sound data or the desired sample rate of
        the wavfile to be loaded. 
    
    win_size_ms : int or float
        Window length in milliseconds for Fourier transform to be applied
        (default 50)
    
    percent_overlap : int or float 
        Amount of overlap between processing windows. For example, if `percent_overlap`
        is set at 0.5, the overlap will be half that of `win_size_ms`. (default 0.5) 
        If an integer is provided, it will be converted to a float between 0 and 1. 
    
    extend_window_ms : int 
        The amount of time in milliseconds to pad or extend the identified VAD segments. This 
        may be useful to include more speech / sound, if desired.
        
    window : str 
        The window function to apply to each window segment. Options are 'hann' and 'hamming'.
        (default 'hann')
        
    zeropad : bool 
        If True, samples will be zeropadded to fill any partially filled window. If False, the 
        samples constituting the partially filled window will be cut off.
        
    **kwargs : additional keyword arguments 
        Keyword arguments for `soundpy.files.loadsound`.
        
    Returns 
    -------
    stft_speech : np.ndarry [size (num_frames_clipped, fft_bins//2+1)]
        The STFT of the `samples` with beginning and ending silences clipped.
    
    vad_matrix : np.ndarry [size (num_frames, )]
        A vector with zeros and ones indicating which indices of the full STFT that 
        have voice activity or not.
    '''
    stft = sp.feats.get_stft(samples, sr, 
                               win_size_ms = win_size_ms, 
                               percent_overlap = percent_overlap,
                               window = window, zeropad = zeropad)
    energy = sp.dsp.get_energy(stft)
    energy_mean = sp.dsp.get_energy_mean(energy)
    beg_index, beg_speech_found = sp.dsp.sound_index(
        energy,energy_mean,start=True)
    end_index, end_speech_found = sp.dsp.sound_index(
        energy,energy_mean,start=False)
    vad_matrix = np.zeros(len(stft))
    if beg_speech_found == False or end_speech_found == False:
        import warnings
        msg = '\nNo speech detected'
        warnings.warn(msg)
        return [], vad_matrix
    if beg_index < end_index:
        if extend_window_ms > 0:
            extra_samples = sp.dsp.calc_frame_length(extend_window_ms, sr)
            num_win_subframes = sp.dsp.calc_num_subframes(
                extra_samples,
                frame_length = frame_length,
                overlap_samples = num_overlap_samples,
                zeropad = zeropad)
            beg_index -= num_win_subframes
            if beg_index < 0:
                beg_index = 0
            end_index += num_win_subframes
            if end_index > len(vad_matrix):
                end_index = len(vad_matrix)
        stft_speech = stft[beg_index:end_index]
        vad_matrix[beg_index:end_index] = 1
        return stft_speech, vad_matrix
    return [], vad_matrix

def get_vad_samples(sound, sr=None, win_size_ms = 50, percent_overlap = 0.5,
                    use_beg_ms = 120, extend_window_ms = 0, energy_thresh = 40, 
                    freq_thresh = 185, sfm_thresh = 5, window = 'hann', zeropad = True,
                    **kwargs):
    '''Returns samples and VAD matrix. Only samples where with VAD are returned.
    
    Parameters
    ----------
    sound : str or numpy.ndarray [size=(num_samples,) or (num_samples, num_channels)]
        If str, wavfile (must be compatible with scipy.io.wavfile). Otherwise 
        the samples of the sound data. Note: in the latter case, `sr`
        must be declared.
    
    sr : int, optional
        The sample rate of the sound data or the desired sample rate of
        the wavfile to be loaded. (default None)
    
    win_size_ms : int or float
        Window length in milliseconds for Fourier transform to be applied
        (default 50)
    
    percent_overlap : int or float 
        Amount of overlap between processing windows. For example, if `percent_overlap`
        is set at 0.5, the overlap will be half that of `win_size_ms`. (default 0.5) 
        If an integer is provided, it will be converted to a float between 0 and 1. 
        
    use_beg_ms : int 
        The amount of time in milliseconds to use from beginning of signal to estimate background
        noise.
        
    extend_window_ms : int 
        The amount of time in milliseconds to pad or extend the identified VAD segments. This 
        may be useful to include more speech / sound, if desired.
        
    energy_thresh : int 
        The threshold to set for measuring energy for VAD in the signal. (default 40)
        
    freq_thresh : int 
        The threshold to set for measuring frequency for VAD in the signal. (default 185)
        
    sfm_thresh : int 
        The threshold to set for measuring spectral flatness for VAD in the signal. (default 5)
        
    window : str 
        The window function to apply to each window segment. Options are 'hann' and 'hamming'.
        (default 'hann')
        
    zeropad : bool 
        If True, samples will be zeropadded to fill any partially filled window. If False, the 
        samples constituting the partially filled window will be cut off.
    
    **kwargs : additional keyword arguments
        Keyword arguments for `soundpy.files.loadsound`
    
    Returns
    -------
    samples_matrix : np.ndarray [size = (num_samples_vad, )]
        The samples of where voice activity was detected.
    vad_matrix_extwin : np.ndarray [size = (num_frames, )]
        A vector of zeros and ones indicating the frames / windows of the samples that either
        had voice activity or not.
    '''
    # raise error if percent_overlap is not supported
    if percent_overlap != 0 and percent_overlap < 0.5:
        raise ValueError('For this VAD function, `percent_overlap` ' +\
            'set to {} is not currently supported.\n'.format(percent_overlap) +\
                'Suggested to set at either 0 or 0.5')
    if percent_overlap > 0.5:
        import warnings
        msg = '\nWarning: for this VAD function, parameter `percent_overlap` has most success '+\
            'when set at 0 or 0.5'
    # raise warnings if sample rate lower than 44100 Hz
    if sr < 44100:
        import warnings
        msg = '\nWarning: voice-activity-detection works best with sample '+\
            'rates above 44100 Hz. Current `sr` set at {}.'.format(sr)
        warnings.warn(msg)
    if isinstance(sound, np.ndarray):
        data = sound
    else:
        data, sr2 = sp.loadsound(sound, sr=sr, **kwargs)
        assert sr2 == sr
        
    frame_length = sp.dsp.calc_frame_length(win_size_ms, sr)
    num_overlap_samples = int(frame_length * percent_overlap)
    num_subframes = sp.dsp.calc_num_subframes(len(data),
                                                frame_length = frame_length,
                                                overlap_samples = num_overlap_samples,
                                                zeropad = zeropad)
    # set number of subframes for extending window
    extwin_num_samples = sp.dsp.calc_frame_length(extend_window_ms, sr)
    num_win_subframes = sp.dsp.calc_num_subframes(extwin_num_samples,
                                                    frame_length = frame_length,
                                                    overlap_samples = num_overlap_samples,
                                                    zeropad = zeropad)
    
    samples_matrix = sp.dsp.create_empty_matrix((len(data)),
                                                complex_vals = False)
    vad_matrix, (sr, e, f, sfm) = sp.dsp.vad(data, sr, 
                                               win_size_ms = win_size_ms,
                                               percent_overlap = percent_overlap, 
                                               use_beg_ms = use_beg_ms, 
                                               energy_thresh = energy_thresh, 
                                               freq_thresh = freq_thresh, 
                                               sfm_thresh = sfm_thresh)
    vad_matrix_extwin = vad_matrix.copy()
    # extend VAD windows with if VAD found
    if extend_window_ms > 0:
        for i, row in enumerate(vad_matrix):
            if row > 0:
                # label samples before VAD as VAD
                if i > num_win_subframes:
                    vad_matrix_extwin[i-num_win_subframes:i] = 1
                else:
                    vad_matrix_extwin[:i] = 1
                # label samples before VAD as VAD
                if i + num_win_subframes < len(vad_matrix):
                    vad_matrix_extwin[i:num_win_subframes+i] = 1
                else:
                    vad_matrix_extwin[i:] = 1
                    
    section_start = 0
    extra_rows = 0
    row = 0
    window_frame = sp.dsp.create_window(window, frame_length)
    for frame in range(num_subframes):
        vad = vad_matrix_extwin[frame]
        if vad > 0:
            section = data[section_start : section_start + frame_length]
            if percent_overlap > 0:
                # apply overlap add to signal
                section_windowed = sp.dsp.apply_window(section, window_frame, zeropad = zeropad)
                samples_matrix[row : row + frame_length] += section_windowed
            else:
                samples_matrix[row : row + frame_length] += section
            row += (frame_length - num_overlap_samples)
        else:
            extra_rows += frame_length - num_overlap_samples
        section_start += (frame_length - num_overlap_samples)
    samples_matrix = samples_matrix[:-extra_rows]
    return samples_matrix, vad_matrix_extwin

def get_samples_clipped(samples, sr, win_size_ms = 50, percent_overlap = 0.5,
                        extend_window_ms = 0, window = 'hann', zeropad = True, **kwargs):
    '''Returns samples and VAD matrix with beginning and ending silence removed.
    
    
    Parameters
    ----------
    samples : str or numpy.ndarray [size=(num_samples,) or (num_samples, num_channels)]
        If str, wavfile (must be compatible with scipy.io.wavfile). Otherwise 
        the samples of the sound data. 
    
    sr : int, optional
        The sample rate of the sound data or the desired sample rate of
        the wavfile to be loaded. 
    
    win_size_ms : int or float
        Window length in milliseconds for Fourier transform to be applied
        (default 50)
    
    percent_overlap : int or float 
        Amount of overlap between processing windows. For example, if `percent_overlap`
        is set at 0.5, the overlap will be half that of `win_size_ms`. (default 0.5) 
        If an integer is provided, it will be converted to a float between 0 and 1. 
    
    extend_window_ms : int 
        The amount of time in milliseconds to pad or extend the identified VAD segments. This 
        may be useful to include more speech / sound, if desired. (default 0)
        
    window : str 
        The window function to apply to each window segment. Options are 'hann' and 'hamming'.
        (default 'hann')
        
    zeropad : bool 
        If True, samples will be zeropadded to fill any partially filled window. If False, the 
        samples constituting the partially filled window will be cut off.
        
    **kwargs : additional keyword arguments 
        Keyword arguments for `soundpy.files.loadsound`.
        
        
    Returns 
    -------
    stft_speech : np.ndarry [size (num_frames_clipped, fft_bins//2+1)]
        The STFT of the `samples` with beginning and ending silences clipped.
    
    vad_matrix : np.ndarry [size (num_frames, )]
        A vector with zeros and ones indicating which indices of the full STFT that 
        have voice activity or not.
    '''
    if not isinstance(samples, np.ndarray):
        samples, sr = sp.loadsound(samples, sr=sr)
    stft = sp.feats.get_stft(samples,sr, 
                               win_size_ms = win_size_ms, 
                               percent_overlap = percent_overlap,
                               window = window, zeropad = zeropad)
    energy = sp.dsp.get_energy(stft)
    energy_mean = sp.dsp.get_energy_mean(energy)
    beg = sp.dsp.sound_index(energy,energy_mean,start=True)
    end = sp.dsp.sound_index(energy,energy_mean,start=False)
    vad_matrix = np.zeros(len(samples))
    if beg[1] == False or end[1] == False:
        import warnings
        msg = 'No speech detected'
        warnings.warn(msg)
        return [], vad_matrix
    
    perc_start = beg[0]/len(energy)
    perc_end = end[0]/len(energy)
    sample_start = int(perc_start*len(samples))
    sample_end = int(perc_end*len(samples))
    if sample_start < sample_end:
        if extend_window_ms > 0:
            extra_frames = sp.dsp.calc_frame_length(extend_window_ms, sr)
            sample_start -= extra_frames
            if sample_start < 0:
                sample_start = 0
            sample_end += extra_frames
            if sample_end > len(vad_matrix):
                sample_end = len(vad_matrix)
        samples_speech = samples[sample_start:sample_end]
        vad_matrix[sample_start:sample_end] = 1
            
        return samples_speech, vad_matrix

    import warnings
    msg = 'No speech detected'
    warnings.warn(msg)
    return [], vad_matrix

# have applied to stft matrix, looks good
def normalize(data, max_val=None, min_val=None):
    '''Normalizes data to be between 0 and 1. Should not be applied to raw sample data.
    
    This is useful if you have predetermined max and min values you want to normalize
    new data with. Is helpful in training models on sound features (not raw samples).
    
    Parameters
    ----------
    data : np.ndarray [size=(num_features,) or (num_frames,num_features)]
        Data to be normalized.
    
    max_val : int or float, optional
        Predetermined maximum value. If None, will use max value
        from `data`.
    
    min_val : int or float, optional
        Predetermined minimum value. If None, will use min value
        from `data`.
    
    
    Returns
    -------
    normed_data : np.ndarray [size = (num_features,) or (num_frames, num_features)]
    
    
    Examples
    --------
    >>> # using the min and max of a previous dataset:
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> input_samples = np.random.random_sample((5,))
    >>> input_samples
    array([0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ])
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
    if data.dtype == np.complex_:
        # take power of absoulte value of stft
        data = np.abs(data)**2
    # add epsilon to avoid division by zero error
    eps = 2**-52
    if max_val is None:
         normed_data = (data - np.min(data)) / (np.max(data) - np.min(data) + eps)
    else:
        if min_val is None:
            min_val = -max_val
        normed_data = (data - min_val) / (max_val - min_val + eps)
    return normed_data

# checked for stereo sound - works: plots each channel in separate plot
def plot_dom_freq(sound, energy_scale = 'power_to_db', title = None,
                  save_pic = False, name4pic = None, overwrite = False, **kwargs):
    '''Plots the approximate dominant frequency over a STFT plot of a signal.
    
    If `sound` has multiple channels, the VAD for each channel is plotted in its 
    own plot.
    
    Parameters
    ----------
    sound : np.ndarray [shape=(num_samples,) or (num_samples, num_channels)]
        The sound to plot the dominant frequency of.
    
    energy_scale : str 
        The scale of energy for the plot. If in frequency spectrum, likey in power and needs
        to be put into db. (default 'power_to_db')
    
    title : str 
        The title for the plot. (default None)
    
    **kwargs : additional keyword arguments 
        Keyword arguments used in both `soundpy.feats.get_stft` and `soundpy.dsp.get_pitch`.

    Returns 
    -------
    None
    '''
    import matplotlib.pyplot as plt
    # ensure numpy array 
    if not isinstance(sound, np.ndarray):
        raise TypeError('Function `soundpy.feats.plot_vad` expects a '+\
            'numpy.ndarray, not type {}.'.format(type(sound)))
    # ensure sample rate is provided
    if 'sr' not in kwargs:
        raise ValueError('Function `soundpy.feats.plot_vad` requires sample rate'+\
            ' of the provided audio samples. Please provide the sample rate '+\
                'under the parameter `sr`.')
    # set defaults if not provided
    if 'win_size_ms' not in kwargs:
        kwargs['win_size_ms'] = 20
    if 'percent_overlap' not in kwargs:
        kwargs['percent_overlap'] = 0.5
        

    y, sr = sound, kwargs['sr']
    if len(y.shape) == 1:
        # add channel column
        y = y.reshape(y.shape+(1,))
    elif y.shape[1] > 11:
        import warnings 
        msg = '\nWARNING: provided `sound` data could be in the wrong format. \n'+\
            'Function `soundpy.feats.plot_vad` expects raw sample data. Data '+\
                'provided could be a stft, fbank, mfcc matrix or some other data'+\
                    ' format. If plot results do not appear as expected, check data.'
        warnings.warn(msg)
        
    if 'mono' in kwargs and kwargs['mono'] is True:
        y = y[:,0]
        y = y.reshape(y.shape+(1,))
    
    for channel in range(y.shape[1]):
        stft_matrix = sp.feats.get_stft(y[:,channel], **kwargs)
        
        # remove complex info for plotting:
        power_matrix = sp.dsp.calc_power(stft_matrix)
        pitch = sp.dsp.get_pitch(sound, **kwargs)
        if energy_scale == 'power_to_db':
            db_matrix = librosa.power_to_db(power_matrix)
        plt.pcolormesh(db_matrix.T)
        # limit the y axis; otherwise y axis goes way too high
        axes = plt.gca()
        axes.set_ylim([0,db_matrix.shape[1]])
        color = 'yellow'
        linestyle = ':'
        plt.plot(pitch, 'ro', color=color)
        if not title:
            title = 'Appx Dominant Frequency'
        # adjust title if more than one channel
        if y.shape[1] > 1:
            if channel == 0:
                title += '\n(channel {})'.format(channel+1)
            else:
                title = title[:-2] + '{})'.format(channel+1)
        plt.title(title)
        if not save_pic:
            plt.show()
        # set up name for saving the plot, given channel number and if other files exist
        else:
            if name4pic is None:
                name4pic = 'dom_freq'
                if y.shape[1] > 1:
                    name4pic += '_channel{}'.format(channel+1)
                name4pic = sp.utils.string2pathlib(name4pic+'.png')
            else:
                name4pic = sp.utils.string2pathlib(name4pic)
                if y.shape[1] > 1:
                    if channel == 0:
                        name = name4pic.stem + '_channel{}'.format(channel+1)
                        name4pic = name4pic.parent.joinpath(name, name4pic.suffix)
                    else:
                        name = name4pic.stem[:-1] + str(channel+1)
                        name4pic = name4pic.parent.joinpath(name + name4pic.suffix)
                if not name4pic.suffix:
                    name4pic = name4pic.parent.joinpath(name4pic.stem+'.png')
            if not overwrite:
                if os.path.exists(name4pic):
                    final_name = name4pic.stem + '_' + sp.utils.get_date()
                    final_name = name4pic.parent.joinpath(final_name+name4pic.suffix)
                else:
                    final_name = name4pic
            else:
                final_name = name4pic
            plt.savefig(final_name)
    
# checked for stereo sound - works: plots each channel in separate plot
def plot_vad(sound, energy_scale = 'power_to_db', 
             title = 'Voice Activity', 
             use_beg_ms = 120, extend_window_ms=0, 
             beg_end_clipped = True, save_pic = False, name4pic = None, 
             overwrite = False, **kwargs):
    '''Plots where voice (sound) activity detected on power spectrum. 
    
    This either plots immediately or saves the plot at `name4pic`. If `sound` 
    has multiple channels, the VAD for each channel is plotted in its own plot.
    
    Parameters
    ----------
    sound : np.ndarray [shape=(num_samples,) or (num_samples, num_channels)]
        The sound to plot the VAD of.
    
    energy_scale : str 
        If plotting STFT or power spectrum, will plot it in decibels. 
        (default 'power_to_db')
        
    title : str 
        The title of the plot (default 'Voice Activity')
        
    use_beg_ms : int 
        The amount of noise to use at the beginning of the signal to measuer VAD. This
        is only applied if `beg_end_silence` is set to False.
        
    extend_window_ms : int 
        The number of milliseconds VAD should be padded. This is useful if one wants to 
        encompass more speech if the VAD is not including all the speech / desired sound. 
        However, this may capture more noise. (default 0)
        
    beg_end_silence : bool 
        If True, just the silences at the beginning and end of the sample will be cut off.
        If False, VAD will be checked throughout the sample, not just the beginning and 
        end. NOTE: Both options have strengths and weaknesses. Sometimes the VAD checking 
        the entire signal is unreliable (e.i. when `beg_end_silence is set to False`), 
        not recognizing speech in speech filled samples. And when set to True, some speech
        sounds tend to get ignored ('s', 'x' and other fricatives).
        
    save_pic : bool 
        If True, the plot will be saved rather than plotted immediately.
        
    name4pic : str 
        The full pathway and filename to save the picture (as .png file). A file
        extension is expected. (default None)
        
    overwrite : bool 
        If False, a date tag will be added to `name4pic` if `name4pic` already exists.
        (default False)
        
    **kwargs : keyword arguments
        Additional keyword arguments for `soundpy.feats.get_speech_stft` or 
        `soundpy.dsp.vad`.
        
    Returns
    -------
    None
    '''
    import matplotlib.pyplot as plt
    # ensure numpy array 
    if not isinstance(sound, np.ndarray):
        raise TypeError('Function `soundpy.feats.plot_vad` expects a '+\
            'numpy.ndarray, not type {}.'.format(type(sound)))
    # ensure sample rate is provided
    if 'sr' not in kwargs:
        raise ValueError('Function `soundpy.feats.plot_vad` requires sample rate'+\
            ' of the provided audio samples. Please provide the sample rate '+\
                'under the parameter `sr`.')
    else:
        # ensure sr is at least 44100; otherwise raise warning
        # vad does not work as well with lower sample rates
        if kwargs['sr'] < 44100:
            import warnings
            msg = '\nWarning: VAD works best with sample rates at or above '+\
                '44100 Hz. To supress this warning, resample the audio from'+\
                    ' {} Hz to at least 44100 Hz.'.format(kwargs['sr'])
            warnings.warn(msg)
    # set defaults if not in kwargs
    if 'win_size_ms' not in kwargs:
        kwargs['win_size_ms'] = 50
    if 'percent_overlap' not in kwargs:
        kwargs['percent_overlap'] = 0.5

    y, sr = sound, kwargs['sr']
    if len(y.shape) == 1:
        # add channel column
        y = y.reshape(y.shape+(1,))
    elif y.shape[1] > 11:
        import warnings 
        msg = '\nWARNING: provided `sound` data could be in the wrong format. \n'+\
            'Function `soundpy.feats.plot_vad` expects raw sample data. Data '+\
                'provided could be a stft, fbank, mfcc matrix or some other data'+\
                    ' format. If plot results do not appear as expected, check data.'
        warnings.warn(msg)
        
    if 'mono' in kwargs and kwargs['mono'] is True:
        y = y[:,0]
        y = y.reshape(y.shape+(1,))
    
    for channel in range(y.shape[1]):
        stft_matrix = sp.feats.get_stft(y[:,channel], **kwargs)
        
        if beg_end_clipped:
            stft_vad, vad_matrix = sp.feats.get_stft_clipped(y[:,channel],
                                                            **kwargs)
        else:
            #vad_matrix, __ = sp.dsp.vad(y[:,channel], use_beg_ms = use_beg_ms, **kwargs)
            stft_vad, vad_matrix = sp.feats.get_vad_stft(y[:,channel],
                                                        use_beg_ms = use_beg_ms,
                                                        **kwargs)
        
        # extend window of VAD if desired
        if extend_window_ms > 0:
            frame_length = sp.dsp.calc_frame_length(kwargs['win_size_ms'],
                                                    kwargs['sr'])
            num_overlap_samples = int(frame_length * kwargs['percent_overlap'])
            # set number of subframes for extending window
            extwin_num_samples = sp.dsp.calc_frame_length(extend_window_ms, kwargs['sr'])
            num_win_subframes = sp.dsp.calc_num_subframes(extwin_num_samples,
                                                            frame_length = frame_length,
                                                            overlap_samples = num_overlap_samples,
                                                            zeropad = True)
            vad_matrix_extwin = vad_matrix.copy()
            for i, row in enumerate(vad_matrix):
                if row > 0:
                    # label samples before VAD as VAD
                    if i > num_win_subframes:
                        vad_matrix_extwin[i-num_win_subframes:i] = 1
                    else:
                        vad_matrix_extwin[:i] = 1
                    # label samples before VAD as VAD
                    if i + num_win_subframes < len(vad_matrix):
                        vad_matrix_extwin[i:num_win_subframes+i] = 1
                    else:
                        vad_matrix_extwin[i:] = 1
            
            vad_matrix = vad_matrix_extwin
        # remove complex info for plotting:
        power_matrix = sp.dsp.calc_power(stft_matrix)
        db_matrix = librosa.power_to_db(power_matrix)
        y_axis = db_matrix.shape[1]
        if max(vad_matrix) > 0:
            vad_matrix = sp.dsp.scalesound(vad_matrix, max_val = y_axis, min_val = 0)
        plt.pcolormesh(db_matrix.T)
        # limit the y axis; otherwise y axis goes way too high
        axes = plt.gca()
        axes.set_ylim([0,db_matrix.shape[1]])
        color = 'yellow'
        linestyle = ':'
        plt.plot(vad_matrix, 'ro', color=color)
        if not title:
            title = 'Voice Activity in Signal'
        if beg_end_clipped and 'clipped' not in title:
            title += ' (clipped)'
        # adjust title if more than one channel
        if y.shape[1] > 1:
            if channel == 0:
                title += '\n(channel {})'.format(channel+1)
            else:
                title = title[:-2] + '{})'.format(channel+1)
        plt.title(title)
        if not save_pic:
            plt.show()
        # set up name for saving the plot, given channel number and if other files exist
        else:
            if name4pic is None:
                name4pic = 'vad'
                if beg_end_clipped:
                    name4pic += '_clipped'
                if y.shape[1] > 1:
                    name4pic += '_channel{}'.format(channel+1)
                name4pic = sp.utils.string2pathlib(name4pic+'.png')
            else:
                name4pic = sp.utils.string2pathlib(name4pic)
                if y.shape[1] > 1:
                    if channel == 0:
                        name = name4pic.stem + '_channel{}'.format(channel+1)
                        name4pic = name4pic.parent.joinpath(name, name4pic.suffix)
                    else:
                        name = name4pic.stem[:-1] + str(channel+1)
                        name4pic = name4pic.parent.joinpath(name + name4pic.suffix)
                if not name4pic.suffix:
                    name4pic = name4pic.parent.joinpath(name4pic.stem+'.png')
            if not overwrite:
                if os.path.exists(name4pic):
                    final_name = name4pic.stem + '_' + sp.utils.get_date()
                    final_name = name4pic.parent.joinpath(final_name+name4pic.suffix)
                else:
                    final_name = name4pic
            else:
                final_name = name4pic
            plt.savefig(final_name)

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
    '''Collects fbank or mfcc features via python-speech-features (rather than librosa).
    '''
    if samples.dtype == np.complex64 or samples.dtype == np.complex128:
        raise TypeError('Function `soundpy.feats.get_mfcc_fbank` only works'+\
            ' with raw signals, not complex data. Received input of type {}'.format(
                samples.dtype))
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
            samples = sp.dsp.zeropad_sound(samples, win_size_ms * sr / 1000, sr = sr)
        else:
            win_size_ms = len(samples)/sr*1000
    frame_length = sp.dsp.calc_frame_length(win_size_ms, sr)
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

def reduce_num_features(feats, desired_shape):
    '''Limits number features of a copy of feats.
    
    This is useful if you want the features to be a certain size, for 
    training models for example.
    '''
    fts = feats.copy()
    if feats.shape != desired_shape:
        empty_matrix = np.zeros(desired_shape, dtype = feats.dtype)
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
def adjust_shape(data, desired_shape, change_dims = None, complex_vals = None):
    try:
        if change_dims is not None:
            raise DeprecationWarning('\nWARNING: Function `soundpy.feats.adjust_shape` will not '+\
                'use the parameter `change_dims` in future versions. \nIf extra dimensions '+\
                    'of length 1 are to be added to the `data`, this will be completed. '+\
                        'However extra dims of greater length are not covered in this function.')
    except DeprecationWarning as e:
        print(e)
    try:
        if complex_vals is not None:
            raise DeprecationWarning('\nWARNING: Function `soundpy.feats.adjust_shape` will not '+\
                'use the parameter `complex_vals` in future versions. This will be '+\
                    'implicitly conducted within the function using `numpy.dtype`.')
    except DeprecationWarning as e:
        print(e)
    
    if len(data.shape) != len(desired_shape):
        data_shape_orig = data.shape
        if desired_shape[0] == 1:
            if data.shape[0] != 1:
                data = data.reshape((1,)+data.shape)
        if desired_shape[-1] == 1:
            if data.shape[-1] != 1:
                data = data.reshape(data.shape + (1,))
        if len(data.shape) != len(desired_shape):   
            raise ValueError('Currently cannot adjust data to a different number of '+\
                'dimensions.\nOriginal data shape: '+str(data_shape_orig)+ \
                    '\nDesired shape: '+str(desired_shape))
    
    # if complex values are in data, set complex_vals to True
    if data.dtype == np.complex_:
        complex_vals = True
    else:
        complex_vals = False
        
    # attempt to zeropad data:
    try:
        greater_items = [i for i, x in enumerate(data.shape) if x > desired_shape[i]]
        # all dimensions can be zeropadded or left alone
        if len(greater_items) == 0:
            data_prepped = sp.feats.zeropad_features(
                data, desired_shape = desired_shape, complex_vals = complex_vals)
        # not all dimensions can be zeropadded. Zeropad what can be zeropadded.
        # then reduce larger dimensions
        elif len(greater_items) == len(data.shape): 
            raise ValueError
            # get out of try statement and run `reduce_num_features` in except clause
        else:
            temp_shape = []
            for i, item in enumerate(data.shape):
                if item <= desired_shape[i]:
                    temp_shape.append(desired_shape[i])
                else:
                    temp_shape.append(item)
            temp_shape = tuple(temp_shape)
            # first zeropad the dimensions that are too small
            data_prepped = sp.feats.zeropad_features(
                data, desired_shape = temp_shape, complex_vals = complex_vals)
            # then clip the dimensions that are too big
            data_prepped = sp.feats.reduce_num_features(
                data_prepped, desired_shape = desired_shape)
    # if zeropadding is smaller than data.shape/features:
    except ValueError:
        # remove extra data/columns to match desired_shape:
        data_prepped = sp.feats.reduce_num_features(data, 
                                                 desired_shape = desired_shape)
    return data_prepped


def reduce_dim(matrix, axis=0):
    import math
    import numpy as np
    if axis < 0:
        axis = len(matrix.shape) + axis
    if axis == 0:
        new_matrix = np.zeros((math.ceil(matrix.shape[0]/2),)+matrix.shape[1:])
        row = 0
        for i in np.arange(0, matrix.shape[0], 2):
            if i < matrix.shape[0] - 2:
                new_matrix[row] = (matrix[i] + matrix[i+1]) / 2
                row += 1
            else:
                new_matrix[row] = matrix[i]
    elif axis == 1:
        new_matrix = np.zeros(matrix.shape[:1] + (math.ceil(matrix.shape[1]/2),))
        col = 0
        for i in np.arange(0, matrix.shape[1], 2):
            if i < matrix.shape[1] - 2:
                new_matrix[:, col] = (matrix[:, i] + matrix[:, i+1]) / 2
                col += 1
            else:
                new_matrix[:, col] = matrix[:, i]
    else:
        raise ValueError('Function `reduce_dim` only accepts 2D data. Axis {}'.format(axis),
                         ' is out of bounds.')
    return new_matrix


def featshape_new_subframe(feature_matrix_shape, new_frame_size, 
                               zeropad = True, axis=0, include_dim_size_1=False):
    '''Subdivides features from (num_frames, num_feats) to (new_frame_size, num_frames, num_feats)
    
    Parameters
    ----------
    feature_matrix_shape : tuple [size=(num_frames, num_features)]
        Feature matrix shape to be subdivided. Can be multidimensional.
        
    new_frame_size : int 
        The number of subframes to section axis into.
    
    zeropad : bool 
        If True, frames that don't completely fill a `new_frame_size` will be 
        zeropadded. Otherwise, those frames will be discarded. (default True)
        
    axis : int 
        The axis where the `new_frame_size` should be applied. (default 0)
        
    Returns
    -------
    new_shape : tuple [size=(num_subframes, new_frame_size, num_feats)]
    
    '''
    if axis < 0:
        # get the axis number if using -1 or -2, etc.
        axis = len(feature_matrix_shape) + axis
    original_dim_length = feature_matrix_shape[axis]
    if zeropad is True:
        subsection_frames = math.ceil(original_dim_length / new_frame_size)
    else:
        subsection_frames = original_dim_length // new_frame_size
    new_shape = []
    for i, ax in enumerate(feature_matrix_shape):
        if i == axis:
            if subsection_frames == 1 and include_dim_size_1 is False:
                # don't include extra dimension if length 1
                new_shape.append(new_frame_size)
            else:
                new_shape.append(new_frame_size) 
                new_shape.append(subsection_frames)
        else:
            new_shape.append(ax)
    new_shape = tuple(new_shape)
    return new_shape


def apply_new_subframe(feature_matrix, new_frame_size, zeropad=True, axis=0):
    '''Reshapes `feature_matrix` to allow for `new_frame_size`. 
    
    Note: Dimensions of `feature_matrix` must be at least 2 and can be up to 5, 
    returning a matrix with one additional dimension. 
    
    Parameters
    ----------
    feature_matrix : np.ndarray [size(num_frames, num_features) ]
        Expects minimum 2D, maximum 5D matrix.
        
    new_frame_size : int 
        The number of subframes to section axis into.
        
    axis : int 
        The axis to apply the `new_frame_size`. (default 0)

    zeropad : bool 
        If True, the feature_matrix will be zeropadded to include frames that do not 
        fill entire frame_size, given the `new_frame_size`. If False, feature_matrix
        will not include the last zeropadded frame. (default True)
        
    Returns
    -------
    feats_reshaped : np.ndarray [size(num_subframes, new_frame_size, num_features)]
        The `feature_matrix` returned with `axis` subdivided into 2 dimensions, the number of subframes and the other length `new_frame_size`. 
        
    Raises
    ------
    ValueError if number of dimensions of `feature_matrix` is below 2 or exceeds 5.
    
    Examples
    --------
    >>> import numpy as np
    >>> matrix = np.arange(24).reshape(3,4,2)
    >>> # apply new_frame_size to dimension of length 4 (i.e. axis 1)
    >>> matrix_zp = apply_new_subframe(matrix, new_frame_size = 3, axis = 1)
    >>> matrix_zp.shape
    (3, 2, 3, 2)
    >>> matrix_zp
    array([[[[ 0,  1],
            [ 2,  3],
            [ 4,  5]],

            [[ 6,  7],
            [ 0,  0],
            [ 0,  0]]],


        [[[ 8,  9],
            [10, 11],
            [12, 13]],

            [[14, 15],
            [ 0,  0],
            [ 0,  0]]],


        [[[16, 17],
            [18, 19],
            [20, 21]],

            [[22, 23],
            [ 0,  0],
            [ 0,  0]]]])
    >>> matrix_nozp = apply_new_subframe(matrix, new_frame_size = 3, axis = 1,
    ...                                    zeropad=False)
    >>> matrix_nozp.shape
    (3, 1, 3, 2)
    >>> matrix_nozp
    array([[[[ 0,  1],
            [ 2,  3],
            [ 4,  5]]],


        [[[ 8,  9],
            [10, 11],
            [12, 13]]],


        [[[16, 17],
            [18, 19],
            [20, 21]]]])

    '''
    if len(feature_matrix.shape) < 2 or len(feature_matrix.shape) > 5:
        raise ValueError('Function `soundpy.feats.apply_new_subframe` '+\
            'can only be applied to matrices between 2 and 5 dimensions.')
    
    datatype = feature_matrix.dtype
    if axis < 0:
        # get the axis number if using -1 or -2, etc.
        axis = len(feature_matrix.shape) + axis
    new_shape = featshape_new_subframe(feature_matrix.shape,
                                           new_frame_size = new_frame_size,
                                           axis = axis,
                                           zeropad = zeropad)
    total_new_samples = np.prod(new_shape)
    current_samples = np.prod(feature_matrix.shape)
    
    # zeropad or reduce feature_matrix to match number of current samples
    diff = total_new_samples - current_samples

    for i, item in enumerate(feature_matrix.shape):
        if i != axis:
            diff /= item
    if zeropad is True:
        if diff >= 0:
            diff = math.ceil(diff)
        else:
            diff = int(diff)
    else:
        if diff >= 0:
            diff = int(diff)
        else:
            diff = math.ceil(diff)
    if axis == 0:
        feature_matrix = sp.feats.adjust_shape(
            feature_matrix,
            ((feature_matrix.shape[0] + diff,) + feature_matrix.shape[1:]))
    elif axis > 0:
        feature_matrix = sp.feats.adjust_shape(
            feature_matrix,
            (feature_matrix.shape[:axis] + (feature_matrix.shape[axis] + diff, ) + \
                feature_matrix.shape[axis+1:]))
    
    feats_reshaped = feature_matrix.reshape(new_shape)
    feats_reshaped = feats_reshaped.astype(datatype)
    return feats_reshaped


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
    X, y = sp.feats.separate_dependent_var(matrix)
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
        X[:, :, j] = preprocessing.normalize(X[:, :, j])
    # Keras needs an extra dimension as a tensor / holder of data
    X = sp.feats.add_tensor(X)
    y = sp.feats.add_tensor(y)
    return X, y, scalars

# TODO test for all these features:
def list_available_features():
    return ['stft', 'powspec', 'fbank', 'mfcc', 'signal']

# TODO REMOVE context_window for next release.
# don't apply context window and such during feature extraction phase
# TODO check if `real_signal` influences change of shape or not
def get_feature_matrix_shape(sr = None, dur_sec = None, feature_type = None,
                             win_size_ms = None, percent_overlap = None,
                             fft_bins = None, num_mfcc = None, num_filters = None,
                             rate_of_change = False, rate_of_acceleration = False,
                             context_window = None, frames_per_sample = None, zeropad = True, labeled_data = False, remove_first_coefficient = False, real_signal = False, **kwargs):
    '''Returns expected shapes of feature matrix depending on several parameters.
     
    Parameters
    ----------
    sr : int 
        Sample rate of the audio to be extracted.
        
    dur_sec : int, float 
        The number of seconds of audio feature extraction will be applied to.
        
    feature_type : str 
        Accepted features include 'signal', 'stft', 'powspec', 'fbank', 'mfcc'. Which
        `feature_type` applied will influence the resulting shape of the feature matrix
        shape.
        
    win_size_ms : int or float
        The size of the window the audio signal should be broken into. If `feature_type` 
        is set to 'signal', this is irrelevant. Otherwise will raise TypeError if set to None.
        
    percent_overlap : float 
        The amount of overlap between windows. If set to 0.5, the number of overlapping
        samples will be half the number of samples that make up `win_size_ms`.
        
    fft_bins : int 
        The number of frequency bins to use when calculating the fast Fourier transform.
        If None, the calculated `frame_length` will be used. 
        
    num_mfcc : int 
        If extracting 'mfcc' features, the total number of coefficients expected.
        
    num_filters : int 
        If extracting 'fbank' features, the total number of mel-filters to be applied.
        
    rate_of_change : bool 
        If True, the first delta will be concatenated to features extracted.
        
    rate_of_acceleration : bool 
        If True, the second delta will be concatenated to features extracted.
        
    context_window : int
        The size of `context_window` or number of samples padding a central frame.
        This may be useful for models training on small changes occuring in the signal, e.g. to break up the image of sound into smaller parts. 
        
    frames_per_sample : int
        The previous keyword argument for sugementing audio into smaller parts.
        Will be removed in future versions and available in generator functions as 
        `context_window`. `frames_per_sample` equals 2 * `context_window` + 1. See 
        `soundpy.models.dataprep.Generator`
        
    zeropad : bool 
        If True, windows and frames will be zeropadded to avoid losing any sample data.
        
    labeled_data : bool 
        If True, a label will be added to the output shape of features. 
        
    remove_first_coefficient : bool 
        If True, the first mfcc coefficient will not be included in feature
        matrix.
        
    **kwargs : additional keyword arguments
        Keyword arguments for `soundpy.feats.get_feats`. These may not be used in this
        function as they may not influence the size of the feature matrix.
        
    Returns
    -------
    feature_matrix_base : tuple
        The base shape of the feature matrix. This is the shape that should result from 
        extracting the features for each audio file 
        
    feature_matrix_model : tuple 
        The shape relevant to training models. For example, one including space for a
        context window and label. 
    '''
    if sr is None:
        raise TypeError('Function `soundpy.feats.get_feature_matrix_shape` expects'+\
            ' parameter `sr` to be of type `int`, not type None.')
    if dur_sec is None:
        raise TypeError('Function `soundpy.feats.get_feature_matrix_shape` expects'+\
            ' parameter `dur_sec` to be of type `int` or `float`, not type None.')
    if win_size_ms is None:
        raise TypeError('Function `soundpy.feats.get_feature_matrix_shape` expects'+\
            ' parameter `win_size_ms` to be of type `int` or `float`, not type None.')
    if feature_type is None:
        raise TypeError('Function `soundpy.feats.get_feature_matrix_shape` expected'+\
            ' parameter `feature_type` to be one of the following: '+\
                ','.join(sp.feats.list_available_features())+\
                    '\nInstead got None.')
    total_samples = sp.dsp.calc_frame_length(dur_sec*1000, sr=sr)
    frame_length = sp.dsp.calc_frame_length(win_size_ms, sr)
    # all we need to know if signal is feature
    if 'signal' in feature_type:
        total_rows_per_wav = total_samples // frame_length
        num_feats = frame_length
        feature_matrix_model = (
            total_rows_per_wav,
            num_feats)
        feature_matrix_base = (
            total_samples,) # currently only single channels
    else:
        if win_size_ms is None or percent_overlap is None:
            raise TypeError('`win_size_ms` or `percent_overlap` cannot be type '+\
                'None. Please set these values, e.g. `win_size_ms` = 20, `percent_overlap` = 0.5')
        win_shift_ms = win_size_ms - (win_size_ms * percent_overlap)
        hop_length = int(win_shift_ms * 0.001 * sr)
        if fft_bins is None:
            fft_bins = int(win_size_ms * sr // 1000)
        # https://librosa.org/doc/latest/generated/librosa.util.frame.html#librosa.util.frame
        total_rows_per_wav = int(1 + (total_samples - fft_bins)//hop_length)
        if 'mfcc' in feature_type:
            if num_mfcc is None:
                num_feats = 40
            else:
                num_feats = num_mfcc
            if remove_first_coefficient is True:
                num_feats -= 1
        elif 'fbank' in feature_type:
            if num_filters is None:
                num_feats = 40
            else:
                num_feats = num_filters
        elif 'stft' in feature_type or 'powspec' in feature_type:
            num_feats = fft_bins//2 + 1
        else:
            raise ValueError('Feature type "{}" '.format(feature_type)+\
                'not understood.\nMust include one of the following: \n'+\
                    ', '.join(list_available_features()))
        if rate_of_change is True and rate_of_acceleration is True:
            num_feats += 2 * num_feats
        elif rate_of_change is True or rate_of_acceleration is True:
            num_feats += num_feats
        try:
            if frames_per_sample is not None or context_window is not None:
                raise DeprecationWarning('\nWARNING: In future versions, the `frames_per_sample` and '+\
                    '`context_window` parameters will be no longer used in feature extraction.\n'+\
                        ' Instead features can be segmented in generator functions using the '+\
                            'parameter `context_window`: `soundpy.models.dataprep.Generator`.')
        except DeprecationWarning as e:
            print(e)
        if context_window or frames_per_sample:
            if context_window:
                subframes = context_window * 2 + 1
            else:
                subframes = frames_per_sample
            batches = math.ceil(total_rows_per_wav/subframes)
            feature_matrix_model = (
                batches,
                subframes,
                num_feats)
            feature_matrix_base = (
                batches * subframes,
                num_feats)
        else:
            feature_matrix_model = (
                total_rows_per_wav,
                num_feats)
            feature_matrix_base = (
                total_rows_per_wav,
                num_feats)
    if labeled_data is True:
        feature_matrix_model = feature_matrix_model[:-1] + (feature_matrix_model[-1] + 1,)
    return feature_matrix_base, feature_matrix_model
        
        
def visualize_feat_extraction(feats, iteration = None, dataset=None, label=None,
                              datadir = None, subsections = False, **kwargs):
    '''Saves plots of features during feature extraction or training of models.
    
    Parameters
    ----------
    feats : np.ndarray [shape=(num_samples,) or (num_samples, num_frames) or \
    (num_frames, num_features) or (num_subsections, num_frames, num_features)]
        The extracted features can be raw signal data, stft, fbank, powspec, mfcc
        data, either as a single plot or subsectioned into batches / subframes.
        
    iteration : int, optional
        The iteration of the audio getting extracted; e.g. the 10th training item.
        
    dataset : str, optional
        The identifying string (for example 'train' , 'val', or 'test', but this can 
        be anything).
        
    label : str, int, optional
        The label of the audio file. Used in titles and filenames.
        
    datadir : str, pathlib.PosixPath, optional
        The directory where related data is located. An 'image' directory will be 
        created within this `datadir` where the saved plots will be stored. If 
        None, will be created in current working directory.
        
    subsections : bool, optional 
        To subsection raw 'signal' data into frames. For other features, this is 
        easier to identify via the shape of `feats`.
        
    **kwargs : additional keyword arguments
        Keyword arguments for `soundpy.feats.get_feats`
        
    Returns
    -------
    None
    '''
    # visualize features:
    if datadir is None:
        datadir = './'
    if not isinstance(datadir, pathlib.PosixPath):
        datadir = sp.utils.string2pathlib(datadir)
    if dataset is not None and iteration is not None:
        save_pic_path = datadir.joinpath(
            'images',dataset,'{}_sample{}_{}'.format(
                kwargs['feature_type'], iteration, label))
        title = '{} {} features: label {}'.format(
                        dataset, kwargs['feature_type'].upper(),
                        label)
    else: 
        save_pic_path = datadir.joinpath(
            'images', '{}_{}'.format(kwargs['feature_type'],
                                     sp.utils.get_date()))
        title = '{} features: label {}'.format(kwargs['feature_type'].upper(),
                        label)
    # make sure this directory exists
    save_pic_dir = sp.utils.check_dir(save_pic_path.parent, make=True)
    # if in batches, save the features in each batch 
    
    # first non raw signal data (e.g. stft, powspec, fbank, mfcc)
    if 'signal' not in kwargs['feature_type'] and \
        len(feats.shape) > 2:
        if len(feats.shape) == 4:
            if feats.shape[-1] == 1:
                feats_temp = feats[:,:,:,0]
            else:
                raise ValueError('Cannot visualize greater than 3D data.')
        elif len(feats.shape) > 4:
            raise ValueError('Cannot visualize greater than 3D data.')
        else:
            feats_temp = feats
        orig_name = save_pic_path.stem
        for i, feat_section in enumerate(feats_temp):
            new_name = orig_name + '_frame_{}'.format(i)
            save_pic_path = save_pic_path.parent.joinpath(new_name)
            sp.feats.plot(feature_matrix = feat_section, 
                            feature_type = kwargs['feature_type'],
                            win_size_ms = kwargs['win_size_ms'],
                            percent_overlap = kwargs['percent_overlap'],
                            title = title + ' frame {}'.format(i),
                            name4pic = save_pic_path,
                            save_pic = True,
                            subprocess = True)
        return None
    
    # then raw signal data; needs parameter `subsections` set to True
    # can only be 2D or 3D (if last dimension is 1)
    if subsections is True and 'signal' in kwargs['feature_type']:
        if len(feats.shape) == 3 and feats.shape[-1] > 1:
            raise ValueError('Cannot visualize raw signal greater than 2D.')
        elif len(feats.shape) == 3 and feats.shape[-1] == 1:
            feats_temp = feats[:,:,0]
        elif len(feats.shape) == 2:
            feats_temp = feats
        elif len(feats.shape) == 1:
            feats_temp = None
        else:
            raise ValueError('Cannot visualize raw signal greater than 2D.')
        if feats_temp is not None:
            orig_name = save_pic_path.stem
            for i, feat_section in enumerate(feats_temp):
                new_name = orig_name + '_frame_{}'.format(i)
                save_pic_path = save_pic_path.parent.joinpath(new_name)
                sp.feats.plot(feature_matrix = feat_section, 
                                    feature_type = kwargs['feature_type'],
                                    win_size_ms = kwargs['win_size_ms'],
                                    percent_overlap = kwargs['percent_overlap'],
                                    title = title + ' frame {}'.format(i),
                                    save_pic = True,
                                    name4pic = save_pic_path.joinpath('frame {}'.format(i)),
                                    subprocess = True)
            return None
    
    # otherwise save features in a single plot
    sp.feats.plot(feature_matrix = feats, 
                    feature_type = kwargs['feature_type'],
                    win_size_ms = kwargs['win_size_ms'],
                    percent_overlap = kwargs['percent_overlap'],
                    title = title,
                    save_pic = True,
                    name4pic = save_pic_path,
                    subprocess = True)
    return None
    
def save_features_datasets(datasets_dict, datasets_path2save_dict, 
                            context_window=None, frames_per_sample = None, labeled_data=False, 
                            subsection_data=False, divide_factor=None,
                            visualize=False, vis_every_n_frames=50, 
                            log_settings=True, decode_dict = None, 
                            random_seed = None, **kwargs):
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
    
    context_window : int
        The size of `context_window` or number of samples padding a central frame.
        This may be useful for models training on small changes occuring in the signal, e.g. to break up the image of sound into smaller parts, to feed 
        to a long short-term memory network (LSTM), for example.
        (Can avoid this by simply reshaping data later). 
        
    frames_per_sample : int
        The previous keyword argument for sugementing audio into smaller parts.
        Will be removed in future versions. This equals 2 * `context_window` + 1
    
    labeled_data : bool 
        If True, expects each audiofile to be accompanied by an integer label. See example 
        given for `datasets_dict`.
    
    subsection_data : bool 
        If you have a large dataset, you may want to divide it into subsections. See 
        soundpy.datasets.subsection_data. If datasets are large enough to raise a MemoryError, 
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
    
    log_settings : bool
        If True, a .csv file will be saved in the feature extraction directory with 
        most of the feature settings saved. (default True)
    
    decode_dict : dict, optional
        The dictionary to get the label given the encoded label. This is for plotting 
        purposes. (default None)
    
    **kwargs : additional keyword arguments
        Keyword arguments for `soundpy.feats.get_feats`.
    
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
    soundpy.feats.get_feats
        Extract features from audio file or audio data.
    '''
    # if dataset is large, may want to divide it into sections
    if divide_factor is None:
        divide_factor = 5
    if subsection_data:
        datasets_dict, datasets_path2save_dict = sp.datasets.section_data(
            datasets_dict,
            datasets_path2save_dict,
            divide_factor=divide_factor)
    # save where data was extracted from:
    dataset_dirs = []
    try:
        # sr must be set. Set to default value.
        if not 'sr' in kwargs or kwargs['sr'] is None:
            import warnings
            msg = '\nWARNING: sample rate was not set. Setting it at 22050 Hz.'
            warnings.warn(msg)
            kwargs['sr'] = 22050
            
        # win_size_ms must be set. Set to default value.
        if not 'win_size_ms' in kwargs or kwargs['win_size_ms'] is None:
            import warnings
            msg = '\nWARNING: `win_size_ms` was not set. Setting it to 20 ms'
            warnings.warn(msg)
            kwargs['win_size_ms'] = 20
            
        # percent_overlap must be set. Set to default value.
        if not 'percent_overlap' in kwargs or kwargs['percent_overlap'] is None:
            import warnings
            msg = '\nWARNING: `percent_overlap` was not set. Setting it to 0.5'
            warnings.warn(msg)
            kwargs['percent_overlap'] = 0.5
        
            
        feat_base_shape, feat_model_shape = sp.feats.get_feature_matrix_shape(
            context_window = context_window,
            frames_per_sample = frames_per_sample,
            labeled_data = labeled_data,
            **kwargs)
    
        # set whether or not features will include complex values:
        if 'stft' in kwargs['feature_type']:
            complex_vals = True
        else:
            complex_vals = False
            
        total_audiofiles = 0

        for key, value in datasets_dict.items():
            # get parent directory of where data should be saved (i.e. for saving pics)
            datapath = datasets_path2save_dict[key]
            if not isinstance(datapath, pathlib.PosixPath):
                datapath = pathlib.Path(datapath)
            datadir = datapath.parent
            # when loading a dictionary, the value is a string
            if isinstance(value, str):
                value = sp.utils.restore_dictvalue(value)
            # len(vale) is the total number of audio files
            feats4model_shape = (len(value),) + feat_model_shape
            feats_matrix = sp.dsp.create_empty_matrix(
                feats4model_shape, 
                complex_vals=complex_vals)
            
            audio_list = value.copy()
            total_audiofiles += len(audio_list)
            # shuffle audiofiles:
            if random_seed is not None:
                random.seed(random_seed)
            random.shuffle(audio_list)
            for j, audiofile in enumerate(audio_list):
                if labeled_data:
                    label, audiofile = int(audiofile[0]), audiofile[1]
                else:
                    label = None
                if isinstance(audiofile, str):
                    audiofile = pathlib.PosixPath(audiofile)
                if j == 0:
                    dataset_dirs.append(audiofile.parent)
                feats = sp.feats.get_feats(audiofile,
                                            **kwargs)

                # zeropad or clip feats if too short or long:
                feats = sp.feats.adjust_shape(
                    feats, 
                    desired_shape = feat_base_shape)
                
                # add label column to feature matrix
                if labeled_data:
                    # create label column
                    label_col = np.zeros((len(feats),1)) + label
                    feats = np.concatenate([feats,label_col], axis=1)
                    
                feats = feats.reshape(feats4model_shape[1:])
                
                #visualize features only every n num frames
                if visualize and j % vis_every_n_frames == 0:
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
                
                    sp.feats.visualize_feat_extraction(
                        feats,
                        iteration = j,
                        dataset = key,
                        label = label_plot,
                        datadir = datadir,
                        subsections = True, # prepping feats 4 model results in subsections 
                        **kwargs)
                
                # fill in empty matrix with features from each audiofile
                feats_matrix[j] = feats
                sp.utils.print_progress(iteration = j, 
                                        total_iterations = len(value),
                                        task = '{} {} feature extraction'.format(
                                            key, kwargs['feature_type']))
            # save data:
            np.save(datasets_path2save_dict[key], feats_matrix)
            print('\nFeatures saved at {}\n'.format(datasets_path2save_dict[key]))
        if log_settings:
            log_filename = datadir.joinpath('log_extraction_settings.csv')
            feat_settings = dict(
                dataset_dirs = dataset_dirs,
                feat_base_shape = feat_base_shape,
                feat_model_shape = feat_model_shape,
                complex_vals = complex_vals,
                context_window = context_window,
                frames_per_sample = frames_per_sample,
                labeled_data = labeled_data,
                decode_dict = decode_dict,
                visualize = visualize,
                vis_every_n_frames = vis_every_n_frames,
                subsection_data = subsection_data,
                divide_factor = divide_factor,
                total_audiofiles = total_audiofiles,
                kwargs = kwargs
                )
            feat_settings_path = sp.utils.save_dict(
                dict2save = feat_settings,
                filename = log_filename,
                overwrite=True)
    except MemoryError as e:
        print('MemoryError: ',e)
        print('\nSectioning data and trying again.\n')
        datasets_dict, datasets_path2save_dict = sp.datasets.section_data(
            datasets_dict, datasets_path2save_dict, divide_factor=divide_factor)
        datasets_dict, datasets_path2save_dict = save_features_datasets(
            datasets_dict = datasets_dict, 
            datasets_path2save_dict = datasets_path2save_dict,
            context_window = context_window,
            frames_per_sample = frames_per_sample,
            labeled_data = labeled_data,
            subsection_data = subsection_data,
            divide_factor = divide_factor,
            visualize = visualize,
            vis_every_n_frames = vis_every_n_frames,
            log_settings = log_settings,
            decode_dict = decode_dict,
            **kwargs)
    return datasets_dict, datasets_path2save_dict

# TODO: update / consolidate
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
        soundpy.datasets.subsection_data. If datasets are large enough to raise a MemoryError, 
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
        Keyword arguments for `soundpy.feats.get_feats`.
    
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
    soundpy.feats.get_feats
        Extract features from audio file or audio data.
    '''
    # if dataset is large, may want to divide it into sections
    if divide_factor is None:
        divide_factor = 5
    if subsection_data:
        datasets_dict, datasets_path2save_dict = sp.datasets.section_data(
            datasets_dict,
            datasets_path2save_dict,
            divide_factor=divide_factor)
    try:
        # depending on which packages one uses, shape of data changes.
        # for example, Librosa centers/zeropads data automatically
        # TODO see which shapes result from python_speech_features
        total_samples = sp.dsp.calc_frame_length(dur_sec*1000, sr=sr)
        # if using Librosa:
        if use_librosa:
            frame_length = sp.dsp.calc_frame_length(win_size_ms, sr)
            win_shift_ms = win_size_ms - (win_size_ms * percent_overlap)
            hop_length = int(win_shift_ms*0.001*sr)
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
                    value = sp.utils.restore_dictvalue(value)
                extraction_shape = (len(value) * audiofile_lim,) + input_shape
                feats_matrix = sp.dsp.create_empty_matrix(
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
                    extract_dir = sp.utils.check_dir(extract_dir, make=True)
                    sp.files.extract(zipfile, extract_path = extract_dir)
                    audiolist = sp.files.collect_audiofiles(extract_dir,
                                                              recursive = True)
                    if audiofile_lim is not None:
                        for i in range(audiofile_lim):
                            if i == len(audiolist) and i < audiofile_lim:
                                print('Short number files: ', audiofile_lim - i)
                                empty_rows += audiofile_lim - i
                                break
                            feats = sp.feats.get_feats(audiolist[i],
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
                                    save_pic_dir = sp.utils.check_dir(save_pic_path.parent, make=True)
                                    sp.feats.plot(feats, 
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
                            
                            feats = sp.feats.zeropad_features(
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
                    sp.files.delete_dir_contents(extract_dir, remove_dir = False)
                    sp.utils.print_progress(iteration = j, 
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
                feat_settings_path = sp.utils.save_dict(
                    dict2save = feat_settings,
                    filename = log_filename,
                    overwrite=True)
        else:
            raise ValueError('Sorry, this functionality is not yet supported. '+\
                'Set `use_librosa` to True.')
    except MemoryError as e:
        print('MemoryError: ',e)
        print('\nSectioning data and trying again.\n')
        datasets_dict, datasets_path2save_dict = sp.datasets.section_data(
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
    feats_reshaped = sp.feats.adjust_shape(feats, desired_shape)
    # reshape to input shape with a necessary "tensor" dimension
    feats_reshaped = feats_reshaped.reshape(input_shape)
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
    # while soundpy works with data in shape (num_frames, num_feats)
    if phase is not None:
        try:
            assert feats.shape == phase.shape
        except AssertionError:
            raise ValueError('Expected `feats` (shape {})'.format(feats.shape)+\
                ' and `phase` (shape {}) '.format(phase.shape) +\
                    'to have the same shape: (num_frames, num_features)')
    win_shift_ms = win_size_ms - (win_size_ms * percent_overlap)
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
            hop_length=int(win_shift_ms*0.001*sr))
    elif 'mfcc' in feature_type:
        feats = feats[:14,:]
        y = librosa.feature.inverse.mfcc_to_audio(
            feats, 
            sr=sr, 
            n_fft = int(win_size_ms*0.001*sr), 
            hop_length=int(win_shift_ms*0.001*sr),
            n_mels=13)
    elif 'stft' in feature_type or 'powspec' in feature_type:
        # can use istft with phase information applied
        if phase is not None:
            feats = feats * phase
            y = librosa.istft(
                feats,
                hop_length=int(win_shift_ms*0.001*sr),
                win_length = int(win_size_ms*0.001*sr))
        # if no phase information available:
        else:
            y = librosa.griffinlim(
                feats,
                hop_length=int(win_shift_ms*0.001*sr),
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
    image_zeropadded = sp.feats.zeropad_features(image_matrix, expected_shape)
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
