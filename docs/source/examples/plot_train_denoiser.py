# coding: utf-8
"""
=============================
Train a Denoising Autoencoder
=============================

Train a denoising autoencoder with clean and noisy acoustic features.

To see how PySoundTool implements this, see `pysoundtool.builtin.denoiser_train`, 
`pysoundtool.builtin.denoiser_feats` and `pysoundtool.builtin.create_denoise_data`.
"""


###############################################################################################
# 


#####################################################################
# Let's import pysoundtool
import pysoundtool as pyso


######################################################
# Prepare for Training: Data Organization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##########################################################
# Designate path relevant for accessing audiodata
pyso_dir = '../../../'


######################################################
# I will load previously extracted features (sample data), see `pysoundtool.feats.save_features_datasets` or `pysoundtool.builtin.denoiser_feats`
feature_extraction_dir = '{}audiodata2/example_feats_models/'.format(pyso_dir)+\
    'denoiser/example_feats_fbank/'

#########################################################
# What is in this folder?
feature_extraction_dir = pyso.utils.check_dir(feature_extraction_dir)
files = list(feature_extraction_dir.glob('*.*'))
for f in files:
    print(f.name)
  
#########################################################
# The .npy files contain the features themselves, in train, validation, and
# test datasets:
files = list(feature_extraction_dir.glob('*.npy'))
for f in files:
    print(f.name)
  
#########################################################
# The .csv files contain information about how the features were extracted
files = list(feature_extraction_dir.glob('*.csv'))
for f in files:
    print(f.name)

#########################################################
# We'll have a look at which features were extracted and other settings:
feat_settings = pyso.utils.load_dict(
    feature_extraction_dir.joinpath('log_extraction_settings.csv'))
for key, value in feat_settings.items():
    print(key, ' --> ', value)
    
#########################################################
# For more about these settings, see `pysoundtool.feats.save_features_datasets`.
    
#########################################################
# We'll have a look at the audio files that were assigned 
# to the train, val, and test datasets.
audio_datasets = pyso.utils.load_dict(
    feature_extraction_dir.joinpath('audiofiles_datasets_clean.csv'))
count = 0
for key, value in audio_datasets.items():
    print(key, ' --> ', value)
    count += 1
    if count > 5:
        break

#############################################################
# Built-In Functionality: PySoundTool does everything for you
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For more about this, see `pysoundtool.builtin.denoiser_train`.

#############################################################
model_dir, history = pyso.denoiser_train(feature_extraction_dir = feature_extraction_dir,
                                         epochs = 100)

#########################################################


#############################################################
# Where the model and logs are located:
model_dir


#############################################################
# Let's plot how the model performed (on this mini dataset)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()
