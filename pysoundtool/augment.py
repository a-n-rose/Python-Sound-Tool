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

def shuffle(sound):
    '''Acoustic augmentation of noise or background sounds.
    
    References
    ----------
    Inoue, T., Vinayavekhin, P., Wang, S., Wood, D., Munawar, A., Ko, B. J., 
    Greco, N., & Tachibana, R. (2019). Detection and Classification of 
    Acoustic Scenes and Events 2019. 25-26 October 2019, New York, NY, USA
    '''
    pass

def mix2sounds(sound1, sound2):
    '''Acoustic augmentation of noise or background sounds.
    
    References
    ----------
    Inoue, T., Vinayavekhin, P., Wang, S., Wood, D., Munawar, A., Ko, B. J., 
    Greco, N., & Tachibana, R. (2019). Detection and Classification of 
    Acoustic Scenes and Events 2019. 25-26 October 2019, New York, NY, USA
    '''
    pass


def vtlp(sound, sr = 16000, a = (0.8,1.2), random_seed = 40,
         oversize_factor = 1, win_size_ms = 16, percent_overlap = 0.5,
         bilinear = True):
    '''
    # TODO work out how to apply oversize factor. Perhaps works better
    # if calculate dft per frame rather than stft with librosa.
    # TODO add piecewise linear rule option (not just bi-linear rule)
    
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
    Xiaodong Cui, Vaibhava Goel, and Brian Kingsbury. Dataaugmentation for deep convolutional
    neuralnetwork acoustic modeling. In2015 IEEE International Conference on Acoustics, Speech 
    and SignalProcessing, ICASSP 2015, South Brisbane, Queensland, Australia, April 19-24, 2015, 
    pages 4545–4549,2015.
    '''
    pass
