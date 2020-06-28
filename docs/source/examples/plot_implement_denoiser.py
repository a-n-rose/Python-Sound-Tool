# coding: utf-8
"""
=================================
Implement a Denoising Autoencoder
=================================

Use PySoundTool to train a denoising autoencoder with clean and noisy acoustic features.

To see how PySoundTool implements this, see `pysoundtool.builtin.denoiser_run`.
"""


############################################################################################
# 

##########################################################
# Ignore this snippet of code: it is only for this example
import os
package_dir = '../../../'
os.chdir(package_dir)

#####################################################################
# Let's import pysoundtool, assuming it is in your working directory:
import pysoundtool as pyst;
# for playing audio in this notebook:
import IPython.display as ipd
import keras


######################################################
# Prepare for Implementation: Data Organization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

######################################################
# Set model pathway
# ~~~~~~~~~~~~~~~~~
# Currently, this expects a model saved with weights, with a .h5 extension.
# (See `model` below)

######################################################
# PySoundTool offers pre-trained models. Let's use one.
model = '{}audiodata/example_feats_models/'.format(package_dir)+\
    'denoiser/example_model/example_denoiser_stft.h5'
# ensure is a pathlib.PosixPath object
print(model)
model = pyst.utils.string2pathlib(model)
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
# the model. See pysoundtool.feats.save_features_datasets
feat_settings = pyst.utils.load_dict(
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
# Provide a new audiofile for the denoiser to denoise!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#########################################################
# To load a new audiofile the denoiser hasn't seen yet, we'll use the 
# dataset assignment files PySoundTool generated during feature extraction:
clean_audio_datasets = pyst.utils.load_dict(
    model_dir.joinpath('audiofiles_datasets_clean.csv'))
noisy_audio_datasets = pyst.utils.load_dict(
    model_dir.joinpath('audiofiles_datasets_noisy.csv'))

#########################################################
# Let's use the first two from the test dataset. The denoiser has never seen these.
clean_audio_test = pyst.utils.restore_dictvalue(clean_audio_datasets['test'])
noisy_audio_test = pyst.utils.restore_dictvalue(noisy_audio_datasets['test'])
clean_sample = clean_audio_test[0]
noisy_sample = noisy_audio_test[0]

##############################################################
# What does the noisy audio sound like?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
samps_noisy, sr = pyst.loadsound(noisy_sample, sr=sr)
ipd.Audio(samps_noisy,rate=sr)

##############################################################
# What does the noisy audio look like?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyst.plotsound(samps_noisy, sr = sr, feature_type='signal')

##############################################################
# What does the clean audio sound like?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
samps_clean, sr = pyst.loadsound(clean_sample, sr = sr)
ipd.Audio(samps_clean,rate=sr)

##############################################################
# What does the clean audio look like?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyst.plotsound(samps_clean, sr = sr, feature_type='signal')

#########################################################################
# Built-In Denoiser Functionality
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##############################################################
# We just need to feed the model path, the noisy sample path, and 
# the feature settings dictionary we looked at above.
y, sr = pyst.denoiser_run(model, noisy_sample, feat_settings)

##########################################################
# How does the output sound?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
ipd.Audio(y,rate=sr)

##########################################################
# How does is the output look? 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyst.plotsound(y, sr=sr, feature_type = 'signal')

##########################################################
# How do they all compare?
# ~~~~~~~~~~~~~~~~~~~~~~~~

##########################################################
# STFT features of the noisy version of the audio:
pyst.plotsound(samps_noisy, sr=sr, feature_type = 'stft', power_scale = 'power_to_db',
               title = 'Noisy input audiofile: STFT features')

##########################################################
# STFT features of the output
pyst.plotsound(y, sr=sr, feature_type = 'stft', power_scale = 'power_to_db',
               title = 'Denoiser Output: STFT features')

##########################################################
# STFT features of the clean version of the audio:
pyst.plotsound(samps_clean, sr=sr, feature_type = 'stft', power_scale = 'power_to_db',
               title = 'Clean "target" audiofile: STFT features')


##########################################################
# It's not perfect but for a very simple implementation, the noise is gone
# and you can hear the person speaking. Pretty cool! 
