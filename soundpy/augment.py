'''The augment module includes functions related to augmenting audio data.
These functions pull from implementations performed in research. 

Other resources for augmentation (not included in soundpy functionality):

Ma, E. (2019). NLP Augmentation. https://github.com/makcedward/nlpaug

Park, D. S., Chan, W., Zhang, Y., Chiu, C., Zoph, B., Cubuk, E. D., & Le, Q. V.
(2019). Google Brain. arxiv.org/pdf/1904.08779.pdf


Nanni, L., Maguolo, G., & Paci, M. (2020). Data augmentation approaches for 
improving animal audio classification. Ecological Informatics, 57, 101084. 
https://doi.org/https://doi.org/10.1016/j.ecoinf.2020.101084:

1.Signal speed scaling by a random number in[0.8,1.2](SpeedupFactoryRange).
2.Pitch shift by a random number in [−2,2]semitones(SemitoneShiftRange).
3.Volume increase/decrease by a random number in [−3,3]dB(VolumeGainRange).
4.Addition of random noise in the range [0,10]dB(SNR).
5.Time shift in the range [−0.005,0.005]seconds(TimeShiftRange).

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
import pathlib
import soundpy as sp

def speed_increase(sound, sr, perc=0.15, **kwargs):
    '''Acoustic augmentation of speech.
    
    References
    ----------
    Nanni, L., Maguolo, G., & Paci, M. (2020). Data augmentation approaches for 
    improving animal audio classification. Ecological Informatics, 57, 101084. 
    https://doi.org/https://doi.org/10.1016/j.ecoinf.2020.101084
    
    Ko, T., Peddinti, V., Povey, D., & Khudanpur (2015). Audio Augmentation for 
    Speech Recognition. Interspeech. 
    
    W. Verhelst and M. Roelands, “An overlap-add technique based on
    waveform similarity (wsola) for high quality time-scale modifica-
    tion of speech,” in Proceedings of the International Conference on
    Acoustics, Speech and Signal Processing (ICASSP), vol. 2, April
    1993, pp. 554–557 vol.2.
    '''
    if isinstance(sound, np.ndarray):
        data = sound
    else:
        data, sr2 = sp.loadsound(sound, sr=sr, **kwargs)
        assert sr2 == sr
    # if entered 50 instead of .50, turns 50 into .50
    if perc > 1:
        while perc > 1:
            perc *= .01
            if perc <= 1:
                break
    rate = 1. + perc
    y_fast = librosa.effects.time_stretch(data, rate)
    return y_fast

def speed_decrease(sound, sr, perc=0.15, **kwargs):
    '''Acoustic augmentation of speech. 
    
    References
    ----------
    Nanni, L., Maguolo, G., & Paci, M. (2020). Data augmentation approaches for 
    improving animal audio classification. Ecological Informatics, 57, 101084. 
    https://doi.org/https://doi.org/10.1016/j.ecoinf.2020.101084
    '''
    if isinstance(sound, np.ndarray):
        data = sound
    else:
        data, sr2 = sp.loadsound(sound, sr=sr, **kwargs)
        assert sr2 == sr
    # if entered 50 instead of .50, turns 50 into .50
    if perc > 1:
        while perc > 1:
            perc *= .01
            if perc <= 1:
                break
    rate = 1. - perc
    y_slow = librosa.effects.time_stretch(data, rate)
    return y_slow


def time_shift(sound, sr, random_seed = None, **kwargs):
    '''Acoustic augmentation of sound (probably not for speech).
    
    Applies random shift of sound by dividing sound into 2 sections and 
    switching them.
    
    Nanni, L., Maguolo, G., & Paci, M. (2020). Data augmentation approaches for 
    improving animal audio classification. Ecological Informatics, 57, 101084. 
    https://doi.org/https://doi.org/10.1016/j.ecoinf.2020.101084
    '''
    if isinstance(sound, np.ndarray):
        data = sound
    else:
        data, sr2 = sp.loadsound(sound, sr=sr, **kwargs)
        assert sr2 == sr
    switched = sp.augment.shufflesound(data, sr=sr, 
                                          num_subsections = 2, 
                                          random_seed = random_seed)
    return switched
    

def shufflesound(sound, sr, num_subsections = 2, random_seed = None, **kwargs):
    '''Acoustic augmentation of noise or background sounds.
    
    This separates the sound into `num_subsections` and pseudorandomizes
    the order.
    
    References
    ----------
    Inoue, T., Vinayavekhin, P., Wang, S., Wood, D., Munawar, A., Ko, B. J.,
    Greco, N., & Tachibana, R. (2019). Shuffling and mixing data augmentation 
    for environmental sound classification. Detection and Classification of 
    Acoustic Scenes and Events 2019. 25-26 October 2019, New York, NY, USA
    '''
    if isinstance(sound, np.ndarray):
        data = sound
    else:
        data, sr2 = sp.loadsound(sound, sr=sr, **kwargs)
        assert sr2 == sr
    subsection_length = len(data) // num_subsections
    order = np.arange(num_subsections)
    if random_seed is not None:
        np.random.seed(random_seed)
    np.random.shuffle(order)
    section_dict = {}
    sample = 0
    for i in range(num_subsections):
        if i == num_subsections-1:
            section = data[sample:]
        else:
            section = data[sample:sample+subsection_length]
        section_dict[i] = section
        sample += subsection_length
    # combine samples in new order:
    samples_shuffled = np.array([])
    for i in order:
        samples_shuffled = np.concatenate((samples_shuffled, section_dict[i]),axis=0)
    return samples_shuffled

def add_white_noise(sound, sr, noise_level=0.01, snr=10, random_seed=None, **kwargs):
    '''
    References
    ----------
    Nanni, L., Maguolo, G., & Paci, M. (2020). Data augmentation approaches for 
    improving animal audio classification. Ecological Informatics, 57, 101084. 
    https://doi.org/https://doi.org/10.1016/j.ecoinf.2020.101084
    '''
    if isinstance(sound, np.ndarray):
        data = sound
    else:
        data, sr2 = sp.loadsound(sound, sr=sr)
        assert sr2 == sr
    n = sp.dsp.generate_noise(num_samples = len(data), 
                                amplitude=noise_level, 
                                random_seed=random_seed)
    if isinstance(snr, list):
        snr = np.random.choice(snr)
    sound_n, snr = sp.dsp.add_backgroundsound(data, n, sr = sr, snr=snr, **kwargs)
    return sound_n

def harmonic_distortion(sound, sr, **kwargs):
    '''Applies sin function five times.
    
    References
    ----------
    Nanni, L., Maguolo, G., & Paci, M. (2020). Data augmentation approaches for 
    improving animal audio classification. Ecological Informatics, 57, 101084. 
    https://doi.org/https://doi.org/10.1016/j.ecoinf.2020.101084
    '''
    if isinstance(sound, np.ndarray):
        data = sound
    else:
        data, sr2 = sp.loadsound(sound, sr=sr, **kwargs)
        assert sr2 == sr
    data = 2*np.pi*data
    count = 0
    while count < 5:
        data = np.sin(data)
        count += 1
    return data
    
def pitch_increase(sound, sr, num_semitones = 2, **kwargs):
    '''
    
    References
    ----------
    Nanni, L., Maguolo, G., & Paci, M. (2020). Data augmentation approaches for 
    improving animal audio classification. Ecological Informatics, 57, 101084. 
    https://doi.org/https://doi.org/10.1016/j.ecoinf.2020.101084
    '''
    if isinstance(sound, np.ndarray):
        data = sound
    else:
        data, sr2 = sp.loadsound(sound, sr=sr, **kwargs)
        assert sr2 == sr
    y_i = librosa.effects.pitch_shift(data, sr=sr, n_steps = num_semitones)
    return y_i

def pitch_decrease(sound, sr, num_semitones = 2, **kwargs):
    '''
    
    References
    ----------
    Nanni, L., Maguolo, G., & Paci, M. (2020). Data augmentation approaches for 
    improving animal audio classification. Ecological Informatics, 57, 101084. 
    https://doi.org/https://doi.org/10.1016/j.ecoinf.2020.101084
    '''
    if isinstance(sound, np.ndarray):
        data = sound
    else:
        data, sr2 = sp.loadsound(sound, sr=sr, **kwargs)
        assert sr2 == sr
    y_d = librosa.effects.pitch_shift(data, sr=sr, n_steps = -num_semitones)
    return y_d
      
# TODO how to control output size without losing frequency data?
# basically how to scale down dimension of frequencies after warping?
# https://docs.scipy.org/doc/scipy/reference/tutorial/ndimage.html#interpolation-functions
# scikit-image resize (only powerspectrum)
# https://stackoverflow.com/questions/23918036/interpolate-whole-arrays-of-complex-numbers
def vtlp(sound, sr, a = (0.8,1.2), random_seed = None,
         oversize_factor = 16, win_size_ms = 50, percent_overlap = 0.5,
         bilinear_warp = True, real_signal = True, fft_bins = 1024, window = 'hann',
         zeropad = True, expected_shape = None, visualize = False):
    '''Applies vocal tract length perturbations directly to dft (oversized) windows.
    
    References
    ----------
    Kim, C., Shin, M., Garg, A., & Gowda, D. (2019). Improved vocal tract length perturbation 
    for a state-of-the-art end-to-end speech recognition system. Interspeech. September 15-19, 
    Graz, Austria.
    
    Nanni, L., Maguolo, G., & Paci, M. (2020). Data augmentation approaches for 
    improving animal audio classification. Ecological Informatics, 57, 101084. 
    https://doi.org/https://doi.org/10.1016/j.ecoinf.2020.101084
    '''
    if isinstance(sound, np.ndarray):
        data = sound
    else:
        data, sr2 = sp.loadsound(sound, sr=sr)
        assert sr2 == sr
    if random_seed is not None:
        np.random.seed(random_seed)
    if isinstance(a, tuple) or isinstance(a, list):
        vtlp_a = np.random.choice(np.arange(min(a), max(a)+.1, 0.1)  )
    elif isinstance(a, int) or isinstance(a, float):
        vtlp_a = a
    else:
        vtlp_a = None
    if isinstance(vtlp_a, int) or isinstance(vtlp_a, float) or isinstance(vtlp_a, np.int_) \
        or isinstance(vtlp_a, np.float_):
            pass
    else:
        raise TypeError('Function `soundpy.augment.vtlp` expected a to be an int or float, or'+\
            ' a list / tuple of ints, or floats; not of type {}'.format(type(a)))
    frame_length = sp.dsp.calc_frame_length(win_size_ms, sr)
    num_overlap_samples = int(frame_length * percent_overlap)
    num_subframes = sp.dsp.calc_num_subframes(len(data),
                                                frame_length = frame_length,
                                                overlap_samples = num_overlap_samples,
                                                zeropad = zeropad)
    max_freq = sr/2.
    if expected_shape is not None:
        # expects last column to represent the number of relevant frequency bins
        #fft_bins = expected_shape[-1]
        fft_bins = (expected_shape[-1]-1) * 2 
    if fft_bins is None:
        fft_bins = int(win_size_ms * sr // 1000)
    total_rows = fft_bins * oversize_factor
    # initialize empty matrix to fill dft values into
    stft_matrix = sp.dsp.create_empty_matrix(
        (num_subframes,total_rows), complex_vals = True)
    
    section_start = 0
    window_frame = sp.dsp.create_window(window, frame_length)
    for frame in range(num_subframes):
        section = data[section_start:section_start+frame_length]
        section = sp.dsp.apply_window(section, window_frame, zeropad = zeropad)
        # apply dft to large window - increase frequency resolution during warping
        section_fft = sp.dsp.calc_fft(section, 
                                        real_signal = real_signal,
                                        fft_bins = total_rows,
                                        )
        if bilinear_warp:
            section_warped = sp.dsp.bilinear_warp(section_fft, vtlp_a)
        else:
            section_warped = sp.dsp.piecewise_linear_warp(section_fft, vtlp_a,
                                                                max_freq = max_freq)
        if real_signal:
            section_warped = section_warped[:len(section_warped)]
        else:
            section_warped = section_warped[:len(section_warped)//2 + 1]
        stft_matrix[frame][:len(section_warped)] = section_warped
        section_start += (frame_length - num_overlap_samples)
    if expected_shape is not None:
        stft_matrix = stft_matrix[:,:len(section_warped)]
        # TODO: find out how to reduce resolution of frequency
        # this technically works but is 1) slow and 2) loses lots of info
        if oversize_factor > 1:
            import skimage
            from skimage.transform import resize
            power_matrix = sp.dsp.calc_power(stft_matrix)
            stft_matrix = resize(power_matrix, expected_shape)
            import warnings
            msg = '\nWARNING: Only the power spectrum of the VTLP augmented signal'+\
                ' can be returned due to resizing the augmentation from '+\
                    '{} to {}'.format(power_matrix.shape, expected_shape)
            warnings.warn(msg)
            #for i in np.arange(0, int(np.sqrt(oversize_factor))):
                #stft_matrix = sp.feats.reduce_dim(stft_matrix, axis=1)
        # ensures matches expected_shape
        stft_matrix = sp.feats.adjust_shape(stft_matrix, expected_shape)
    else:
        stft_matrix = stft_matrix[:,:len(section_warped)]
    if visualize:
        sp.feats.plot(stft_matrix, feature_type = 'stft', sub_process=True, 
                    name4pic = 'vtlp_{}.png'.format(sp.utils.get_date()),
                    title = 'size: {}'.format(stft_matrix.shape),
                    save_pic=True)
    return stft_matrix, vtlp_a

def get_augmentation_dict():
    '''Returns dictionary with augmentation options as keys and values set to False.
    
    Examples
    --------
    >>> import soundpy as sp
    >>> ad = sp.augment.get_augmentation_dict()
    >>> ad
    {'speed_increase': False,
    'speed_decrease': False,
    'time_shift': False,
    'shufflesound': False,
    'add_white_noise': False,
    'harmonic_distortion': False,
    'pitch_increase': False,
    'pitch_decrease': False,
    'vtlp': False}
    >>> # to set augmentation to True:
    >>> ad['add_white_noise'] = True
    >>> ad
    {'speed_increase': False,
    'speed_decrease': False,
    'time_shift': False,
    'shufflesound': False,
    'add_white_noise': True,
    'harmonic_distortion': False,
    'pitch_increase': False,
    'pitch_decrease': False,
    'vtlp': False}
    '''
    base_dict = dict([('speed_increase', False),
                      ('speed_decrease', False),
                      ('time_shift', False),
                      ('shufflesound', False),
                      ('add_white_noise', False),
                      ('harmonic_distortion', False),
                      ('pitch_increase', False),
                      ('pitch_decrease', False),
                      ('vtlp', False),
                      ])
    return base_dict

def list_augmentations():
    '''Lists available augmentations.
    
    Examples
    --------
    >>> import soundpy as sp
    >>> print(sp.augment.list_augmentations())
    Available augmentations:
            speed_increase
            speed_decrease
            time_shift
            shufflesound
            add_white_noise
            harmonic_distortion
            pitch_increase
            pitch_decrease
            vtlp
    '''
    augmentation_dict = sp.augment.get_augmentation_dict()
    aug_list = '\t'+'\n\t'.join(str(x) for x in augmentation_dict.keys())
    augmentations = 'Available augmentations:\n '+ aug_list
    return augmentations
    
# TODO test to see if list can be applied to all augmentations, not just 'add_white_noise'
def get_augmentation_settings_dict(augmentation):
    '''Returns default settings of base function for augmentation.
    
    Parameters
    ----------
    augmentation : str 
        The augmentation of interest.
    
    Returns
    -------
    aug_defaults : dict 
        A dictionary with the base augmentation function parameters as keys  
        and default values as values. 
    
    Examples
    --------
    >>> import soundpy as sp
    >>> d = sp.augment.get_augmentation_settings_dict('speed_decrease')
    >>> d
    {'perc': 0.15}
    >>> # can use this dictionary to apply different values for augmentation
    >>> d['perc'] = 0.1
    >>> d
    {'perc': 0.1}
    >>> # to build a dictionary with several settings:
    >>> many_settings_dict = {}
    >>> many_settings_dict['add_white_noise'] = sp.augment.get_augmentation_settings_dict('add_white_noise')
    >>> many_settings_dict['pitch_increase'] = sp.augment.get_augmentation_settings_dict('pitch_increase')
    >>> many_settings_dict
    {'add_white_noise': {'noise_level': 0.01, 'snr': 10, 'random_seed': None},
    'pitch_increase': {'num_semitones': 2}}
    >>> # change 'snr' default values to list of several values
    >>> # this would apply white noise at either 10, 15, or 20 SNR, at random
    >>> many_settings_dict['add_white_noise']['snr'] = [10, 15, 20]
    >>> # change number of semitones pitch increase is applied
    >>> many_settings_dict['pitch_increase']['num_semitones'] = 1
    >>> many_settings_dict
    {'add_white_noise': {'noise_level': 0.01,
    'snr': [10, 15, 20],
    'random_seed': None},
    'pitch_increase': {'num_semitones': 1}}

    Raises
    ------
    ValueError if `augmentation` does not match available augmentations.
    
    See Also
    --------
    soundpy.models.dataprep.augment_features
        The above dictionary example `many_settings_dict` can be applied under the
        parameter `augment_settings_dict` to apply augmentation settings when 
        augmenting data, for example, within a generator function. See `soundpy.models.dataprep.GeneratorFeatExtraction`.
    '''
    if augmentation == 'speed_increase':
        aug_defaults = sp.utils.get_default_args(sp.augment.speed_increase)
    elif augmentation == 'speed_decrease':
        aug_defaults = sp.utils.get_default_args(sp.augment.speed_decrease)        
    elif augmentation == 'time_shift':
        aug_defaults = sp.utils.get_default_args(sp.augment.time_shift)
    elif augmentation == 'shufflesound':
        aug_defaults = sp.utils.get_default_args(sp.augment.shufflesound)
    elif augmentation == 'add_white_noise':
        aug_defaults = sp.utils.get_default_args(sp.augment.add_white_noise)
    elif augmentation == 'harmonic_distortion':
        aug_defaults = sp.utils.get_default_args(sp.augment.harmonic_distortion)
    elif augmentation == 'pitch_increase':
        aug_defaults = sp.utils.get_default_args(sp.augment.pitch_increase)
    elif augmentation == 'pitch_decrease':
        aug_defaults = sp.utils.get_default_args(sp.augment.pitch_decrease)
    elif augmentation == 'vtlp':
        aug_defaults = sp.utils.get_default_args(sp.augment.vtlp)
    else:
        raise ValueError('Receieved `augmentation` "{}"'.format(augmentation)+\
            ' which is not included in available augmentations:\n{}'.format(
                sp.augment.list_augmentations()))
    return aug_defaults
    
