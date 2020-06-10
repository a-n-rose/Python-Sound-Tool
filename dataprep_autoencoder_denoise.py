
import pysoundtool as pyst
import numpy as np


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
data_train_path = denoise_data_path.joinpath('{}_data_{}.npy'.format('train',
                                                                      feature_type))
data_val_path = denoise_data_path.joinpath('{}_data_{}.npy'.format('val',
                                                                      feature_type))
data_test_path = denoise_data_path.joinpath('{}_data_{}.npy'.format('test',
                                                                      feature_type))
# TODO
# log feature settings as well!
feature_settings_path = denoise_data_path.joinpath('feature_settings.csv')


# 3) collect audiofiles and divide them into train, val, and test datasets
# can do this different ways.. this is simply one way I've done it.

noiseaudio = sorted(pyst.utils.collect_audiofiles(audio_noisy_path, 
                                                  hidden_files = False,
                                                  wav_only = False,
                                                  recursive = False))
cleanaudio = sorted(pyst.utils.collect_audiofiles(audio_clean_path, 
                                                  hidden_files = False,
                                                  wav_only = False,
                                                  recursive = False))

# check if they match up:
for i, audiofile in enumerate(noiseaudio):
    if not pyst.utils.check_noisy_clean_match(audiofile, cleanaudio[i]):
        raise ValueError('The noisy and clean audio datasets do not appear to match.')


