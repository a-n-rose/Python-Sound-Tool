
import pysoundtool as pyst
import numpy as np


# 1) specify clean and noisy audio directories, and ensure they exist
audio_clean_path = pyst.ustils.check_dir(\
    './audiodata/minidatasets/denoise/cleanspeech_IEEE_small/', 
                                         make=False)
audio_noisy_path = pyst.ustils.check_dir(\
    '/.audiodata/minidatasets/denoise/noisyspeech_IEEE_small/',
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
feature_settings_path = classify_data_path.joinpath('feature_settings.csv')


# 3) collect audiofiles and divide them into train, val, and test datasets
