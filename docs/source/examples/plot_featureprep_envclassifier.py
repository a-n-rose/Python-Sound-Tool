# coding: utf-8
"""
=====================================
Feature Extraction for Classification
=====================================

Extract acoustic features from labeled data for 
training an environment or speech classifier.

To see how PySoundTool implements this, see `pysoundtool.builtin.envclassifier_feats`.
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

import matplotlib.pyplot as plt
import pysoundtool as pyso 
import IPython.display as ipd
package_dir = '../../../'
os.chdir(package_dir)
pyso_dir = package_dir
######################################################
# Prepare for Extraction: Data Organization
# -----------------------------------------

######################################################
# I will use a sample speech commands data set:

##########################################################
# Designate path relevant for accessing audiodata
data_dir = '/home/airos/Projects/Data/sound/speech_commands_small_section/'

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
# Built-In Functionality - PySoundTool extracts the features for you
# ----------------------------------------------------------------------------

############################################################
# Define which data to use and which features to extract
# Everything else is based on defaults. A feature folder with
# the feature data will be created in the current working directory.
# (Although, you can set this under the parameter `data_features_dir`)
# `visualize` saves periodic images of the features extracted.
# This is useful if you want to know what's going on during the process.
extraction_dir = pyso.envclassifier_feats(data_dir, 
                                          feature_type=feature_type, 
                                          dur_sec=dur_sec,
                                          visualize=True);

################################################################
# The extracted features, extraction settings applied, and 
# which audio files were assigned to which datasets
# will be saved in the following directory:
extraction_dir

############################################################
# And that's it!
