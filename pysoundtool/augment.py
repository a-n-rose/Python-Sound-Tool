'''The augment module includes functions related to augmenting audio data.
These functions pull from implementations performed in research. 

Other resources for augmentation (not included in PySoundTool functionality):

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
import pysoundtool as pyst

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
        data, sr2 = pyst.loadsound(sound, sr=sr, **kwargs)
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
        data, sr2 = pyst.loadsound(sound, sr=sr, **kwargs)
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
        data, sr2 = pyst.loadsound(sound, sr=sr, **kwargs)
        assert sr2 == sr
    switched = pyst.augment.shufflesound(data, sr=sr, 
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
        data, sr2 = pyst.loadsound(sound, sr=sr, **kwargs)
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
        data, sr2 = pyst.loadsound(sound, sr=sr)
        assert sr2 == sr
    n = pyst.dsp.generate_noise(num_samples = len(data), 
                                amplitude=noise_level, 
                                random_seed=random_seed)
    sound_n, snr = pyst.dsp.add_backgroundsound(data, n, sr = sr, snr=snr, **kwargs)
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
        data, sr2 = pyst.loadsound(sound, sr=sr, **kwargs)
        assert sr2 == sr
    data = 2*np.pi*data
    count = 0
    while count < 5:
        data = np.sin(data)
        count += 1
    return data
    
def pitch_increase(sound, sr = 16000, num_semitones = 2, **kwargs):
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
        data, sr2 = pyst.loadsound(sound, sr=sr, **kwargs)
        assert sr2 == sr
    y_i = librosa.effects.pitch_shift(data, sr=sr, n_steps = num_semitones)
    return y_i

def pitch_decrease(sound, sr = 16000, num_semitones = 2, **kwargs):
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
        data, sr2 = pyst.loadsound(sound, sr=sr, **kwargs)
        assert sr2 == sr
    y_d = librosa.effects.pitch_shift(data, sr=sr, n_steps = -num_semitones)
    return y_d
      
def vtlp(sound, sr = 16000, a = (0.8,1.2), random_seed = None,
         oversize_factor = 16, win_size_ms = 50, percent_overlap = 0.5,
         bilinear_warp = True, real_signal = False, fft_bins = 1024, window = 'hann'):
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
        data, sr2 = pyst.loadsound(sound, sr=sr)
        assert sr2 == sr
    if random_seed is not None:
        np.random.seed(random_seed)
    vtlp_a = np.random.choice(np.arange(min(a), max(a)+.1, 0.1)  )
    
    frame_length = pyst.dsp.calc_frame_length(win_size_ms, sr)
    num_overlap_samples = int(frame_length * percent_overlap)
    num_subframes = pyst.dsp.calc_num_subframes(len(data),
                                                frame_length = frame_length,
                                                overlap_samples = num_overlap_samples,
                                                zeropad = True)
    max_freq = sr//2
    # ensure even
    if not max_freq % 2 == 0:
        max_freq += 1
    total_rows = fft_bins * oversize_factor
    # initialize empty matrix to fill dft values into
    stft_matrix = pyst.dsp.create_empty_matrix((num_subframes,total_rows), complex_vals = True)
    
    section_start = 0
    window_frame = pyst.dsp.create_window(window, frame_length)
    for frame in range(num_subframes):
        section = data[section_start:section_start+frame_length]
        section = pyst.dsp.apply_window(section, window_frame, zeropad = True)
        # apply dft to large window - increase frequency resolution during warping
        section_fft = pyst.dsp.calc_fft(section, 
                                        real_signal = real_signal,
                                        fft_bins = total_rows,
                                        )
        if bilinear_warp:
            section_warped = pyst.dsp.bilinear_warp(section_fft, vtlp_a)
        else:
            section_warped = pyst.dsp.piecewise_linear_warp(section_fft, vtlp_a,
                                                                max_freq = max_freq)
        section_warped = section_warped[:total_rows]
        stft_matrix[frame] = section_warped
        section_start += (frame_length - num_overlap_samples)
    stft_matrix = stft_matrix[:,:max_freq]
    return stft_matrix, vtlp_a
