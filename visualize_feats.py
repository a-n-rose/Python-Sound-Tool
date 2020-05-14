import librosa
import numpy as np
import matplotlib.pyplot as plt

def visualize_feats(feature_matrix, feature_type, 
                    save_pic=False, name4pic=None, scale='power_to_db'):
    '''Visualize feature extraction; frames on x axis, features on y axis. 
    
    Parameters
    ----------
    feature_matrix : numpy.ndarray
        Matrix of feeatures.
    feature_type : str
        Either 'mfcc' or 'fbank' features. MFCC: mel frequency cepstral
        coefficients; FBANK: mel-log filterbank energies (default 'fbank')
    scale : str, optional
        If features need to be adjusted, e.g. from power to decibels. 
        Default is 'power_to_db', as the visuals look best when working 
        with features extracted via librosa package.
    '''
    if 'fbank' in feature_type:
        axis_feature_label = 'Num Mel Filters'
    elif 'mfcc' in feature_type:
        axis_feature_label = 'Num Mel Freq Cepstral Coefficients'
    elif 'stft' in feature_type:
        axis_feature_label = 'Short-Time Fourier Transform'
    elif 'signal' in feature_type:
        axis_feature_label = 'Amplitude'
    elif 'band_spec' in feature_type:
        axis_feature_label = 'Bands'
    if scale is None:
        pass
    elif scale == 'power_to_db':
        feature_matrix = librosa.power_to_db(feature_matrix)
    elif scale == 'db_to_power':
        feature_matrix = librosa.db_to_power(feature_matrix)
    elif scale == 'amplitude_to_db':
        feature_matrix = librosa.amplitude_to_db(feature_matrix)
    elif scale == 'db_to_amplitude':
        feature_matrix = librosa.db_to_amplitude(feature_matrix)
    plt.clf()
    if feature_type == 'signal':
        plt.plot(feature_matrix)
    else:
        plt.pcolormesh(feature_matrix.T)
    plt.xlabel('Frames')
    plt.ylabel(axis_feature_label)
    plt.title('{} Features'.format(feature_type.upper()))
    if save_pic:
        outputname = name4pic or 'visualize{}feats'.format(feature_type.upper())
        plt.savefig('{}.png'.format(outputname))
    else:
        plt.show()

target_wav = '/home/airos/Projects/Data/denoising_sample_data/noisyspeech_uwnu_limit100_test/CHM02_56-08__noise_keyboard_scale0.3.wav'
y, sr = librosa.load(target_wav)

#librosa.feature.spectral_bandwidth(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='reflect', freq=None, centroid=None, norm=True, p=2)


stft = np.abs(librosa.stft(y))
band_feats = librosa.feature.spectral_bandwidth(S = stft)
visualize_feats(band_feats, feature_type = 'band_spec')
