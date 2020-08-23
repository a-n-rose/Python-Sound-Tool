# coding: utf-8
"""
=======================================================
Feature Extraction for Denoising: Clean and Noisy Audio
=======================================================

Extract acoustic features from clean and noisy datasets for 
training a denoising model, e.g. a denoising autoencoder.

To see how soundpy implements this, see `soundpy.builtin.denoiser_feats`.
"""


###############################################################################################
# 

#####################################################################
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parparentdir = os.path.dirname(parentdir)
packagedir = os.path.dirname(parparentdir)
sys.path.insert(0, packagedir)

import soundpy as sp 
import IPython.display as ipd
package_dir = '../../../'
os.chdir(package_dir)
sp_dir = package_dir

######################################################
# Prepare for Extraction: Data Organization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

######################################################
# I will use a mini denoising dataset as an example

# Example noisy data:
data_noisy_dir = '{}../mini-audio-datasets/denoise/noisy'.format(sp_dir)
# Example clean data:
data_clean_dir = '{}../mini-audio-datasets/denoise/clean'.format(sp_dir)
# Where to save extracted features:
data_features_dir = './audiodata/example_feats_models/denoiser/'

######################################################
# Choose Feature Type 
# ~~~~~~~~~~~~~~~~~~~
# We can extract 'mfcc', 'fbank', 'powspec', and 'stft'.
# if you are working with speech, I suggest 'fbank', 'powspec', or 'stft'.

feature_type = 'stft'
sr = 22050

######################################################
# Set Duration of Audio 
# ~~~~~~~~~~~~~~~~~~~~~
# How much audio in seconds used from each audio file.
# the speech samples are about 3 seconds long.
dur_sec = 3

#######################################################################
# Option 1: Built-In Functionality: soundpy does everything for you
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

############################################################
# Define which data to use and which features to extract. 
# NOTE: beacuse of the very small dataset, will set 
# `perc_train` to a lower level than 0.8. (Otherwise, will raise error)
# Everything else is based on defaults. A feature folder with
# the feature data will be created in the current working directory.
# (Although, you can set this under the parameter `data_features_dir`)
# `visualize` saves periodic images of the features extracted.
# This is useful if you want to know what's going on during the process.
perc_train = 0.6 # with larger datasets this would be around 0.8
extraction_dir = sp.denoiser_feats(
    data_clean_dir = data_clean_dir, 
    data_noisy_dir = data_noisy_dir,
    sr = sr,
    feature_type = feature_type, 
    dur_sec = dur_sec,
    perc_train = perc_train,
    visualize=True);
extraction_dir

################################################################
# The extracted features, extraction settings applied, and 
# which audio files were assigned to which datasets
# will be saved in the `extraction_dir` directory


############################################################
# Logged Information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let's have a look at the files in the extraction_dir. The files ending 
# with .npy extension contain the feature data; the .csv files contain 
# logged information. 
featfiles = list(extraction_dir.glob('*.*'))
for f in featfiles:
    print(f.name)
  
############################################################
# Feature Settings
# ~~~~~~~~~~~~~~~~~~
# Since much was conducted behind the scenes, it's nice to know how the features
# were extracted, for example, the sample rate and number of frequency bins applied, etc.
feat_settings = sp.utils.load_dict(
    extraction_dir.joinpath('log_extraction_settings.csv'))
for key, value in feat_settings.items():
    print(key, ' --> ', value)
    
