# coding: utf-8
"""
=================================
Implement a Denoising Autoencoder
=================================

Implement denoising autoencoder to denoise a noisy speech signal.

To see how soundpy implements this, see `soundpy.builtin.denoiser_run`.
"""


############################################################################################
# 

#####################################################################
# Let's import soundpy and other packages
import soundpy as sp
import numpy as np
# for playing audio in this notebook:
import IPython.display as ipd

#####################################################################
# As well as the deep learning component of soundpy
from soundpy import models as spdl

######################################################
# Prepare for Implementation: Data Organization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##########################################################
# Set path relevant for audio data for this example
sp_dir = '../../../'

######################################################
# Set model pathway
# ~~~~~~~~~~~~~~~~~
# Currently, this expects a model saved with weights, with a .h5 extension.
# (See `model` below)

######################################################
# The soundpy repo offers a pre-trained denoiser, which we'll use.
model = '{}audiodata/models/'.format(sp_dir)+\
    'denoiser/example_denoiser_stft.h5'
# ensure is a pathlib.PosixPath object
print(model)
model = sp.utils.string2pathlib(model)
model_dir = model.parent

#########################################################
# What is in this folder?
files = list(model_dir.glob('*.*'))
for f in files:
    print(f.name)
  
######################################################
# Provide dictionary with feature extraction settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#########################################################
# If soundpy extracts features for you, a 'log_extraction_settings.csv' 
# file will be saved, which includes relevant feature settings for implementing 
# the model; see `soundpy.feats.save_features_datasets`
feat_settings = sp.utils.load_dict(
    model_dir.joinpath('log_extraction_settings.csv'))
for key, value in feat_settings.items():
    print(key, ' --> ', value)
    # change objects that were string to original format
    import ast
    try:
        feat_settings[key] = ast.literal_eval(value)
    except ValueError:
        pass
    except SyntaxError:
        pass

#########################################################
# For the purposes of plotting, let's use some of the settings defined:
feature_type = feat_settings['feature_type']
sr = feat_settings['sr']

######################################################
# Provide new audio for the denoiser to denoise!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#########################################################
# We'll use sample speech from the soundpy repo:
speech = sp.string2pathlib('{}audiodata/python.wav'.format(sp_dir))
s, sr = sp.loadsound(speech, sr=sr)

#########################################################
# Let's add some white noise (10 SNR)
s_n = sp.augment.add_white_noise(s, sr=sr, snr=10)

##############################################################
# What does the noisy audio sound like?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ipd.Audio(s_n,rate=sr)

##############################################################
# What does the noisy audio look like?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sp.plotsound(s_n, sr = sr, feature_type='signal')

##############################################################
# What does the clean audio sound like?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ipd.Audio(s,rate=sr)

##############################################################
# What does the clean audio look like?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sp.plotsound(s, sr = sr, feature_type='signal')

#########################################################################
# Built-In Denoiser Functionality
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##############################################################
# We just need to feed the model path, the noisy sample path, and 
# the feature settings dictionary we looked at above.
y, sr = spdl.denoiser_run(model, s_n, feat_settings)

##########################################################
# How does the output sound?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
ipd.Audio(y,rate=sr)

##########################################################
# How does is the output look? 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sp.plotsound(y, sr=sr, feature_type = 'signal')

##########################################################
# How do the features compare?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##########################################################
# STFT features of the noisy input speech:
sp.plotsound(s_n, sr=sr, feature_type = 'stft', energy_scale = 'power_to_db',
               title = 'Noisy input: STFT features')

##########################################################
# STFT features of the output
sp.plotsound(y, sr=sr, feature_type = 'stft', energy_scale = 'power_to_db',
               title = 'Denoiser Output: STFT features')

##########################################################
# STFT features of the clean version of the audio:
sp.plotsound(s, sr=sr, feature_type = 'stft', energy_scale = 'power_to_db',
               title = 'Clean "target" audio: STFT features')


##########################################################
# It's not perfect but for a pretty simple implementation, the noise is gone
# and you can hear the person speaking. Pretty cool! 
