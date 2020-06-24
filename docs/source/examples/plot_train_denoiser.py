# coding: utf-8
"""
=============================
Train a Denoising Autoencoder
=============================

Use PySoundTool to train a denoising autoencoder with clean and noisy acoustic features.
"""


###############################################################################################
# 

##########################################################
# Ignore this snippet of code: it is only for this example
import os
package_dir = '../../../'
os.chdir(package_dir)

#####################################################################
# Let's import pysoundtool, assuming it is in your working directory:
import pysoundtool as pyst


######################################################
# Prepare for Training: Data Organization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

######################################################
# PySoundTool offers pre-extracted features. Let's use them.
feature_extraction_dir = '{}audiodata/example_feats_models/'.format(package_dir)+\
    'denoiser/features_fbank_6m20d0h7m17s996ms/'

#########################################################
# What is in this folder?
feature_extraction_dir = pyst.utils.check_dir(feature_extraction_dir)
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
# The .csv files contain information about where the features came from.
files = list(feature_extraction_dir.glob('*.csv'))
for f in files:
    print(f.name)

#########################################################
# We'll have a look at which features were extracted and other settings:
feat_settings = pyst.utils.load_dict(
    feature_extraction_dir.joinpath('log_extraction_settings.csv'))
for key, value in feat_settings.items():
    print(key, ' --> ', value)
    
#########################################################
# For more about these settings, see `pysoundtool.feats.get_feats`.
    
#########################################################
# We'll have a look at the audio files that were assigned 
# to the train, val, and test datasets.
audio_datasets = pyst.utils.load_dict(
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

#############################################################
model_dir, history = pyst.denoiser_train(feature_extraction_dir = feature_extraction_dir,
                                         epochs = 10)

#########################################################
# For more about this function, see `pysoundtool.train_models.denoiser_train`.

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
