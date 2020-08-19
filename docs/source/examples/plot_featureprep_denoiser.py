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
import soundpy as sp
import IPython.display as ipd

######################################################
# Prepare for Extraction: Data Organization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

######################################################
# I will use a mini denoising dataset as an example

# Example noisy data:
data_noisy_dir = '/home/airos/Projects/Data/denoising/uwnu/noisy'
# Example clean data:
data_clean_dir = '/home/airos/Projects/Data/denoising/uwnu/clean/'
# Where to save extracted features:
data_features_dir = './audiodata/example_feats_models/denoiser/'

######################################################
# Choose Feature Type 
# ~~~~~~~~~~~~~~~~~~~
# We can extract 'mfcc', 'fbank', 'powspec', and 'stft'.
# if you are working with speech, I suggest 'fbank', 'powspec', or 'stft'.

feature_type = 'fbank'
sr = 22050

######################################################
# Set Duration of Audio 
# ~~~~~~~~~~~~~~~~~~~~~
# How much audio in seconds used from each audio file.
# the speech samples are about 3 seconds long.
dur_sec = 3

######################################################
# Set Context Window / Number of Frames
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# How many sections should each sample be broken into? (optional)
# Some research papers include a 'context window' or the like, 
# which this refers to. This will result in subframe sizes:
# context_window * 2 + 1. (The context windows surrounding a central frame)
context_window = 5

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
    context_window = context_window,
    perc_train = perc_train,
    limit = 200,
    visualize=True);
extraction_dir

################################################################
# The extracted features, extraction settings applied, and 
# which audio files were assigned to which datasets
# will be saved in the `extraction_dir` directory


############################################################
# And that's it!
