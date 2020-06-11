
import pysoundtool as pyst
import numpy as np
import math


# 1) specify clean and noisy audio directories, and ensure they exist
audio_clean_path = pyst.utils.check_dir(\
    './audiodata/minidatasets/denoise/cleanspeech_IEEE_small/', 
                                         make=False)
audio_noisy_path = pyst.utils.check_dir(\
    './audiodata/minidatasets/denoise/noisyspeech_IEEE_small/',
                                         make=False)

# 2) create paths for what we need to save:
denoise_data_path = pyst.utils.check_dir('./audiodata/test_denoiser/', make=True)
# train, val, and test data
# which features will we extract?
feature_type = 'mfcc'  # 'fbank', 'stft', 'mfcc'
data_train_noisy_path = denoise_data_path.joinpath('{}_data_{}_{}.npy'.format('train',
                                                                        'noisy',
                                                                      feature_type))
data_val_noisy_path = denoise_data_path.joinpath('{}_data_{}_{}.npy'.format('val',
                                                                      'noisy',
                                                                      feature_type))
data_test_noisy_path = denoise_data_path.joinpath('{}_data_{}_{}.npy'.format('test',
                                                                       'noisy',
                                                                      feature_type))
data_train_clean_path = denoise_data_path.joinpath('{}_data_{}_{}.npy'.format('train',
                                                                        'clean',
                                                                      feature_type))
data_val_clean_path = denoise_data_path.joinpath('{}_data_{}_{}.npy'.format('val',
                                                                      'clean',
                                                                      feature_type))
data_test_clean_path = denoise_data_path.joinpath('{}_data_{}_{}.npy'.format('test',
                                                                       'clean',
                                                                      feature_type))

# TODO
# log feature settings as well!
feature_settings_path = denoise_data_path.joinpath('feature_settings.csv')


# 3) collect audiofiles and divide them into train, val, and test datasets
# can do this different ways.. this is simply one way I've done it.

noisyaudio = sorted(pyst.utils.collect_audiofiles(audio_noisy_path, 
                                                  hidden_files = False,
                                                  wav_only = False,
                                                  recursive = False))
cleanaudio = sorted(pyst.utils.collect_audiofiles(audio_clean_path, 
                                                  hidden_files = False,
                                                  wav_only = False,
                                                  recursive = False))

# check if they match up:
for i, audiofile in enumerate(noisyaudio):
    if not pyst.utils.check_noisy_clean_match(audiofile, cleanaudio[i]):
        raise ValueError('The noisy and clean audio datasets do not appear to match.')

# save to csv:
noisy_audio_dict = dict([('noisy', noisyaudio)])
clean_audio_dict = dict([('clean', cleanaudio)])

noisy_audio_dict_path = denoise_data_path.joinpath('noisy_audio.csv')
noisy_audio_dict_path = pyst.utils.save_dict(noisy_audio_dict, noisy_audio_dict_path,
                                             overwrite=True)

clean_audio_dict_path = denoise_data_path.joinpath('clean_audio.csv')
clean_audio_dict_path = pyst.utils.save_dict(clean_audio_dict, clean_audio_dict_path,
                                             overwrite=True)

train_noisy, val_noisy, test_noisy = pyst.data.audio2datasets(noisy_audio_dict_path,
                                                              seed=40)
train_clean, val_clean, test_clean = pyst.data.audio2datasets(clean_audio_dict_path,
                                                              seed=40)

# save train, val, test datasets:

dataset_dict_noisy = dict([('train', train_noisy),('val', val_noisy),('test', test_noisy)])
dataset_dict_clean = dict([('train', train_clean),('val', val_clean),('test', test_clean)])

# keep track of paths to save data
dataset_paths_noisy_dict = dict([('train',data_train_noisy_path),
                                 ('val', data_val_noisy_path),
                                 ('test',data_test_noisy_path)])
dataset_paths_clean_dict = dict([('train',data_train_clean_path),
                                 ('val', data_val_clean_path),
                                 ('test',data_test_clean_path)])

path2_noisy_dataset_waves = denoise_data_path.joinpath('audiofiles_datasets_noisy.csv')
path2_clean_dataset_waves = denoise_data_path.joinpath('audiofiles_datasets_clean.csv')

path2_noisy_dataset_waves = pyst.utils.save_dict(dataset_dict_noisy,
                                                 path2_noisy_dataset_waves,
                                                 overwrite=True)
path2_clean_dataset_waves = pyst.utils.save_dict(dataset_dict_clean,
                                                 path2_clean_dataset_waves,
                                                 overwrite=True)


# 5) extract features

'''When extracting features, need to first create empty matrix to fill.
This means we must know the final shape of all features put together:
'''

# decide settings, which will influence the size of data:
sr = 22050
win_size_ms = 25
percent_overlap = 0.5
dur_sec = 3
frames_per_sample = 11
# decide number of features:
# for mfcc, this is the number of mel cepstral coefficients (common: 13, 40)
# for bank, the number of filters applied (common: 22, 24, 29, 40, 128 (etc.) )
num_feats = 40 
# for mfcc, fbank, and stft features, Fast Fourier Transform is applied (fft)
# must determin how many fft bins (frequency bins) will be used
# default number of fft bins is the frame length (num samples in each processing window)
n_fft = win_size_ms * sr // 1000
# same as:
frame_length = pyst.dsp.calc_frame_length(win_size_ms, sr)

# number of features for stft data may be different if not using Librosa.
if 'stft' in feature_type:
    # using Librosa for extracing stft features:
    # according to Librosa 
    # num features is the number of fft bins divided by 2 + 1
    num_feats = int(1+n_fft/2)
    complex_vals = True
else:
    complex_vals = False
    
# depending on which packages one uses, shape of data changes.
# for example, Librosa centers/zeropads data automatically
# TODO see which shapes result from python_speech_features
total_samples = pyst.dsp.calc_frame_length(dur_sec*1000, sr=sr)
use_librosa= True
# if using Librosa:
if use_librosa:
    hop_length = int(win_size_ms*percent_overlap*0.001*sr)
    center=True
    mode='reflect'
    # librosa centers samples by default, sligthly adjusting total 
    # number of samples
    if center:
        y_zeros = np.zeros((total_samples,))
        y_centered = np.pad(y_zeros, int(n_fft // 2), mode=mode)
        total_samples = len(y_centered)
    # each audio file 
    total_rows_per_wav = int(1 + (total_samples - n_fft)//hop_length)
    
    
    # adjust shape for autoencoder
    # want smaller windows
    batch_size = math.ceil(total_rows_per_wav/frames_per_sample)

    for key, value in dataset_dict_clean.items():
        extraction_shape = (len(value),
            batch_size, frames_per_sample,
            num_feats)
        
        feats_matrix = pyst.dsp.create_empty_matrix(
            extraction_shape, 
            complex_vals=complex_vals)

        for j, audiofile in enumerate(value):
            if not pyst.utils.check_noisy_clean_match(dataset_dict_noisy[key][j],
                                                    audiofile):
                raise ValueError('There is a mismatch between noisy and clean audio. '+\
                    '\nThe noisy file:\n{}'.format(dataset_dict_noisy[key][i])+\
                        '\ndoes not seem to match the clean file:\n{}'.format(audiofile))

            feats = pyst.feats.get_feats(audiofile,
                                        sr=sr,
                                        features=feature_type,
                                        win_size_ms=win_size_ms,
                                        percent_overlap=percent_overlap,
                                        window='hann',
                                        num_filters=num_feats,
                                        num_mfcc=num_feats,
                                        duration=dur_sec)
  
            # zeropad feats if too short:
            feats = pyst.data.zeropad_features(feats, 
                                               desired_shape = (
                                                   extraction_shape[1]*extraction_shape[2],
                                                   extraction_shape[3]),
                                               complex_vals = complex_vals)
            ## visualize features:
            #if 'mfcc' in feature_type:
                #scale = None
            #else:
                #scale = 'power_to_db'
            ##visualize features only every n num frames
            #every_n_frames = 20
            #if j % every_n_frames == 0:
                #pyst.feats.plot(feats, feature_type = feature_type, scale=scale,
                                #title='{} {} clean features'.format(key, feature_type.upper()))
            feats = feats.reshape(extraction_shape[1:])
            # fill in empty matrix with features from each audiofile

            feats_matrix[j] = feats
        ## must be 2 D to visualize
        #pyst.feats.plot(feats_matrix.reshape((feats_matrix.shape[0] * feats_matrix.shape[1] * feats_matrix.shape[2], feats_matrix.shape[3])), feature_type=feature_type,
                        #scale=scale, x_label='number of audio files',
                        #title='{} {} clean features'.format(key, feature_type.upper()))
        # save data:
        np.save(dataset_paths_clean_dict[key], feats_matrix)






    for key, value in dataset_dict_noisy.items():
        extraction_shape = (len(value),
            batch_size, frames_per_sample,
            num_feats)
        
        feats_matrix = pyst.dsp.create_empty_matrix(
            extraction_shape, 
            complex_vals=complex_vals)

        for j, audiofile in enumerate(value):
            if not pyst.utils.check_noisy_clean_match(dataset_dict_noisy[key][j],
                                                    audiofile):
                raise ValueError('There is a mismatch between noisy and clean audio. '+\
                    '\nThe noisy file:\n{}'.format(dataset_dict_noisy[key][i])+\
                        '\ndoes not seem to match the clean file:\n{}'.format(audiofile))

            feats = pyst.feats.get_feats(audiofile,
                                        sr=sr,
                                        features=feature_type,
                                        win_size_ms=win_size_ms,
                                        percent_overlap=percent_overlap,
                                        window='hann',
                                        num_filters=num_feats,
                                        num_mfcc=num_feats,
                                        duration=dur_sec)
  
            # zeropad feats if too short:
            feats = pyst.data.zeropad_features(feats, 
                                               desired_shape = (
                                                   extraction_shape[1]*extraction_shape[2],
                                                   extraction_shape[3]),
                                               complex_vals = complex_vals)
            ## visualize features:
            #if 'mfcc' in feature_type:
                #scale = None
            #else:
                #scale = 'power_to_db'
            ##visualize features only every n num frames
            #every_n_frames = 20
            #if j % every_n_frames == 0:
                #pyst.feats.plot(feats, feature_type = feature_type, scale=scale,
                                #title='{} {} noisy features'.format(key, feature_type.upper()))
            feats = feats.reshape(extraction_shape[1:])
            # fill in empty matrix with features from each audiofile

            feats_matrix[j] = feats
        ## must be 2 D to visualize
        #pyst.feats.plot(feats_matrix.reshape((feats_matrix.shape[0] * feats_matrix.shape[1] * feats_matrix.shape[2], feats_matrix.shape[3])), feature_type=feature_type,
                        #scale=scale, x_label='number of audio files',
                        #title='{} {} noisy features'.format(key, feature_type.upper()))
        # save data:
        np.save(dataset_paths_noisy_dict[key], feats_matrix)

        

