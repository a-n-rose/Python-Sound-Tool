# coding: utf-8
"""
=====================================
Feature Extraction for Classification
=====================================

Extract acoustic features from labeled data for 
training an environment or speech classifier.

To see how soundpy implements this, see `soundpy.builtin.envclassifier_feats`.
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
# -----------------------------------------

######################################################
# I will use a sample speech commands data set:

##########################################################
# Designate path relevant for accessing audiodata
data_dir = '{}../mini-audio-datasets/speech_commands/'.format(sp_dir)

######################################################
# Choose Feature Type 
# ~~~~~~~~~~~~~~~~~~~
# We can extract 'mfcc', 'fbank', 'powspec', and 'stft'.
# if you are working with speech, I suggest 'fbank', 'powspec', or 'stft'.

feature_type = 'fbank'

######################################################
# Set Duration of Audio 
# ~~~~~~~~~~~~~~~~~~~~~
# How much audio in seconds used from each audio file.
# The example noise and speech files are only 1 second long
dur_sec = 1


#############################################################
# Built-In Functionality - soundpy extracts the features for you
# ---------------------------------------------------------------

############################################################
# Define which data to use and which features to extract
# Everything else is based on defaults. A feature folder with
# the feature data will be created in the current working directory.
# (Although, you can set this under the parameter `data_features_dir`)
# `visualize` saves periodic images of the features extracted.
# This is useful if you want to know what's going on during the process.
extraction_dir = sp.envclassifier_feats(data_dir, 
                                          feature_type=feature_type, 
                                          dur_sec=dur_sec,
                                          visualize=True);

################################################################
# The extracted features, extraction settings applied, and 
# which audio files were assigned to which datasets
# will be saved in the following directory:
extraction_dir

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
    
    
############################################################
# Labeled Data
# ~~~~~~~~~~~~~~~~~~
# These are the labels and their encoded values:
encode_dict = sp.utils.load_dict(
    extraction_dir.joinpath('dict_encode.csv'))
for key, value in encode_dict.items():
    print(key, ' --> ', value)
