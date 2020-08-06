# coding: utf-8
"""
=================================
Implement a Denoising Autoencoder
=================================

Implement denoising autoencoder to denoise a noisy speech signal.

To see how PySoundTool implements this, see `pysoundtool.builtin.denoiser_run`.
"""


############################################################################################
# 

#####################################################################
# Let's import pysoundtool and other packages
import pysoundtool as pyso
import numpy as np
# for playing audio in this notebook:
import IPython.display as ipd

######################################################
# Prepare for Implementation: Data Organization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##########################################################
# Set path relevant for audio data for this example
pyso_dir = '../../../'

######################################################
# Set model pathway
# ~~~~~~~~~~~~~~~~~
# Currently, this expects a model saved with weights, with a .h5 extension.
# (See `model` below)

######################################################
# The PySoundTool repo offers a pre-trained denoiser, which we'll use.
model = '{}audiodata/models/'.format(pyso_dir)+\
    'denoiser/example_denoiser_stft.h5'
# ensure is a pathlib.PosixPath object
print(model)
model = pyso.utils.string2pathlib(model)
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
# If PySoundTool extracts features for you, a 'log_extraction_settings.csv' 
# file will be saved, which includes relevant feature settings for implementing 
# the model; see `pysoundtool.feats.save_features_datasets`
feat_settings = pyso.utils.load_dict(
    model_dir.joinpath('log_extraction_settings.csv'))
for key, value in feat_settings.items():
    print(key, ' ---> ', value)
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
# We'll use sample speech from the PySoundTool repo:
speech = pyso.string2pathlib('{}audiodata/python.wav'.format(pyso_dir))
s, sr = pyso.loadsound(speech, sr=sr)

#########################################################
# Let's add some white noise (10 SNR)
s_n = pyso.augment.add_white_noise(s, sr=sr, snr=10)

##############################################################
# What does the noisy audio sound like?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ipd.Audio(s_n,rate=sr)

##############################################################
# What does the noisy audio look like?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyso.plotsound(s_n, sr = sr, feature_type='signal')

##############################################################
# What does the clean audio sound like?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ipd.Audio(s,rate=sr)

##############################################################
# What does the clean audio look like?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyso.plotsound(s, sr = sr, feature_type='signal')

#########################################################################
# Built-In Denoiser Functionality
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##############################################################
# We just need to feed the model path, the noisy sample path, and 
# the feature settings dictionary we looked at above.
y, sr = pyso.denoiser_run(model, s_n, feat_settings)

##########################################################
# How does the output sound?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
ipd.Audio(y,rate=sr)

##########################################################
# How does is the output look? 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyso.plotsound(y, sr=sr, feature_type = 'signal')

##########################################################
# How do the features compare?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##########################################################
# STFT features of the noisy input speech:
pyso.plotsound(s_n, sr=sr, feature_type = 'stft', energy_scale = 'power_to_db',
               title = 'Noisy input: STFT features')

##########################################################
# STFT features of the output
pyso.plotsound(y, sr=sr, feature_type = 'stft', energy_scale = 'power_to_db',
               title = 'Denoiser Output: STFT features')

##########################################################
# STFT features of the clean version of the audio:
pyso.plotsound(s, sr=sr, feature_type = 'stft', energy_scale = 'power_to_db',
               title = 'Clean "target" audio: STFT features')


##########################################################
# It's not perfect but for a pretty simple implementation, the noise is gone
# and you can hear the person speaking. Pretty cool! 
