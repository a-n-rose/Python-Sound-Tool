'''The augment module includes functions related to augmenting audio data.
These functions pull from implementations performed in research. 

Other resources for augmentation (not included in PySoundTool functionality):

Ma, E. (2019). NLP Augmentation. https://github.com/makcedward/nlpaug

Park, D. S., Chan, W., Zhang, Y., Chiu, C., Zoph, B., Cubuk, E. D., & Le, Q. V.
(2019). Google Brain. arxiv.org/pdf/1904.08779.pdf

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

def speed():
    '''Acoustic augmentation of speech.
    
    References
    ----------
    Ko, T., Peddinti, V., Povey, D., & Khudanpur (2015). Audio Augmentation for 
    Speech Recognition. Interspeech. 
    '''
    pass

def shufflesound(sound, sr, num_subsections = 2, random_seed = 40, **kwargs):
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

def mix2sounds(sound1, sound2):
    '''Acoustic augmentation of noise or background sounds.
    
    Two sounds are mixed together. However, currently out of the scope of PySoundTool
    functionality.
    
    References
    ----------
    Inoue, T., Vinayavekhin, P., Wang, S., Wood, D., Munawar, A., Ko, B. J.,
    Greco, N., & Tachibana, R. (2019). Shuffling and mixing data augmentation 
    for environmental sound classification. Detection and Classification of 
    Acoustic Scenes and Events 2019. 25-26 October 2019, New York, NY, USA
    '''
    pass


def vtlp_stft(sound, sr = 16000, a = (0.8,1.2), random_seed = 40,
         oversize_factor = 1, win_size_ms = 16, percent_overlap = 0.5):
    '''
    
    References
    ----------
    Kim, C., Shin, M., Garg, A., & Gowda, D. (2019). Improved vocal tract length perturbation 
    for a state-of-the-art end-to-end speech recognition system. Interspeech. September 15-19, 
    Graz, Austria.
    '''
    if isinstance(sound, np.ndarray):
        data = sound
    else:
        data, sr2 = pyst.loadsound(sound, sr=sr)
        assert sr2 == sr
    vtlp_a = np.random.choice(np.arange(min(a), max(a)+.1, 0.1)  )
    # can put in a stft or complex matrix. If so, skip this step
    if data.dtype != np.complex64 and data.dtype != np.complex128:
        stft = pyst.feats.get_feats(sound, sr=sr, feature_type = 'stft', 
                                    win_size_ms = win_size_ms * oversize_factor, 
                                    percent_overlap = 0.5)
    stft = data
    # bi-linear transformation
    nominator = (1-vtlp_a) * np.sin(stft)
    denominator = 1 - (1-vtlp_a) * np.cos(stft)
    stft_t = stft + 2 * np.arctan(nominator/(denominator + 1e-6) + 1e-6)

    return stft_t, vtlp_a
      
      
def vtlp_dft(sound, sr = 16000, a = (0.8,1.2), random_seed = 40,
         oversize_factor = 16, win_size_ms = 50, percent_overlap = 0.5,
         bilinear_warp = True, real_signal = True, fft_bins = 1024, window = 'hann'):
    '''Applies vocal tract length perturbations directly to dft (oversized) windows.
    
    References
    ----------
    Kim, C., Shin, M., Garg, A., & Gowda, D. (2019). Improved vocal tract length perturbation 
    for a state-of-the-art end-to-end speech recognition system. Interspeech. September 15-19, 
    Graz, Austria.
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
            section_warped = pyst.augment.bilinear_warp(section_fft, vtlp_a)
        else:
            section_warped = pyst.augment.piecewise_linear_warp(section_fft, vtlp_a)
        section_warped = section_warped[:total_rows]
        stft_matrix[frame][:len(section_warped)] = section_warped
        section_start += num_overlap_samples
    return stft_matrix, vtlp_a
      
def bilinear_warp(fft_value, alpha):
    nominator = (1-alpha) * np.sin(fft_value)
    denominator = 1 - (1-alpha) * np.cos(fft_value)
    fft_warped = fft_value + 2 * np.arctan(nominator/(denominator + 1e-6) + 1e-6)
    return fft_warped

def piecewise_linear_warp(fft_value, alpha):
    if fft_value.all() <= (fft_value * (min(alpha, 1)/ (alpha + 1e-6))).all():
        fft_warped = fft_value * alpha
    else:
        nominator = np.pi - fft_value * (min(alpha, 1))
        denominator = np.pi - fft_value * (min(alpha, 1) / (alpha + 1e-6))
        fft_warped = np.pi - (nominator / denominator + 1e-6) * (np.pi - fft_value)
    return fft_warped

def bilinear_warp_stft(stft_matrix, alpha):
    nominator = (1-alpha) * np.sin(stft_matrix)
    denominator = 1 - (1-alpha) * np.cos(stft_matrix)
    stft_t = stft_matrix + 2 * np.arctan(nominator/(denominator + 1e-6) + 1e-6)
    return stft_t
    
def jitter():
    '''
    References
    ----------
    Navdeep Jaitly and G. E. Hinton.  Vocal tract length perturbation (vtlp) improves speech 
    recognition. In Proceedings of the International Conference on Machine Learning (ICML) 
    2013 Workshop on DeepLearning for Audio, Speech and Language Processing, Atlanta, Georgia, 
    USA, June 16-21, 2013, 2013.
    '''
    pass

def stochastic_feature_mapping():
    '''
    References
    ----------
    Xiaodong Cui, Vaibhava Goel, and Brian Kingsbury. Data augmentation for deep convolutional
    neuralnetwork acoustic modeling. In 2015 IEEE International Conference on Acoustics, Speech 
    and SignalProcessing, ICASSP 2015, South Brisbane, Queensland, Australia, April 19-24, 2015, 
    pages 4545–4549,2015.
    '''
    pass
