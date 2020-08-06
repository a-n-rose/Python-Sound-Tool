# coding: utf-8
"""
=====================================================
Extract and Augment Features During Training: ConvNet
=====================================================

Extract acoustic features and augment them as you train a model.

To see how PySoundTool implements this, see `pysoundtool.builtin.envclassifier_extract_train` and 
`pysoundtool.models.dataprep`
"""


###############################################################################################
# 


#######################################################################

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

#########################################################
# Prepare for Extraction: Set Feature Extraction Settings
# =======================================================

#########################################################
# Choose Feature Type, etc. 
# -------------------------
# As of now, this functionality unfortunately only works for 'stft'.
feature_type = 'stft'
sr = 44100 # higher sr rates seem to do better with augmentation techniques
num_filters = 40 # number of fbTypeank filters, in this case
dur_sec = 1 # how much sound data from each audio file to use 
win_size_ms = 20 # size of processing window for applying the fourier transform
percent_overlap = 0.5 # the amount of overlap between processing windows

# set these variables into dict:
get_feats_kwargs = dict(feature_type = feature_type,
                        num_filters = num_filters,
                        dur_sec = dur_sec,
                        win_size_ms = win_size_ms,
                        percent_overlap = percent_overlap,
                        sr = sr)

#########################################################
# Choose Augmentations
# --------------------
# Augmenting the audio data we have allows for us to feed the model with more data
# than we actually have, hopefully resulting in a more robust model

##########################################################
# Find out which augmentations can be made:
print(pyso.augment.list_augmentations())

##########################################################
# get a dict with the available augmentations:
aug_dict = pyso.augment.get_augmentation_dict()

##########################################################
# see the default values for these augmentations
augment_settings_dict = {}
for key in aug_dict.keys():
    augment_settings_dict[key] = pyso.augment.get_augmentation_settings_dict(key)
for key, value in augment_settings_dict.items():
    print(key, ' : ', value)

##########################################################
# Alrighty, let's decide on some augmentations:
# If we want several different ones, we can set them up 
# by creating a dictionary for each set of augmentations

##########################################################
# No Augmentation
# ~~~~~~~~~~~~~~~
# Just an empty dict or `aug_dict` with all keys set to False 
augmentation_none = dict()

##########################################################
# Add White Noise
# ~~~~~~~~~~~~~~~
augmentation_noise = dict(aug_dict)
augmentation_noise['add_white_noise'] = True
for key, value in augmentation_noise.items():
    print(key, ' : ', value)

##########################################################
# Decrease the Pitch
# ~~~~~~~~~~~~~~~~~~
augmentation_pitch_decrease = dict(aug_dict)
augmentation_pitch_decrease['pitch_decrease'] = True
for key, value in augmentation_pitch_decrease.items():
    print(key, ' : ', value)

##########################################################
# Increase the Pitch
# ~~~~~~~~~~~~~~~~~~
augmentation_pitch_increase = dict(aug_dict)
augmentation_pitch_increase['pitch_increase'] = True
for key, value in augmentation_pitch_increase.items():
    print(key, ' : ', value)


##########################################################
# Decrease the Speed
# ~~~~~~~~~~~~~~~~~~
augmentation_speed_decrease = dict(aug_dict)
augmentation_speed_decrease['speed_decrease'] = True
for key, value in augmentation_speed_decrease.items():
    print(key, ' : ', value)


##########################################################
# Increase the Speed
# ~~~~~~~~~~~~~~~~~~
augmentation_speed_increase = dict(aug_dict)
augmentation_speed_increase['speed_increase'] = True
for key, value in augmentation_speed_increase.items():
    print(key, ' : ', value)
    
#########################################################
# Combine Augmentations
# ~~~~~~~~~~~~~~~~~~~~~
# As an example, we will use one of many combinations of these augmentations:

augmentation_speed_increase_pitch_decrease_white_noise = dict(aug_dict)
augmentation_speed_increase_pitch_decrease_white_noise['add_white_noise'] = True
augmentation_speed_increase_pitch_decrease_white_noise['pitch_decrease'] = True
augmentation_speed_increase_pitch_decrease_white_noise['speed_increase'] = True
for key, value in augmentation_speed_increase_pitch_decrease_white_noise.items():
    print(key, ' : ', value)

#########################################################
# Listen to Augmentations (Default Settings)
# ------------------------------------------
# If you are able to hear the augmentations, this may give you an idea
# the settings you want to apply
# For this we'll load the example 'python' speech, available from the 
# PySoundTool repo.

##########################################################
# No Augmentation
# ~~~~~~~~~~~~~~~
y, sr = pyso.loadsound('{}audiodata/python.wav'.format(package_dir), sr = sr)
pyso.plotsound(y, sr=sr, feature_type = feature_type)
ipd.Audio(y,rate=sr)

##########################################################
# White Noise
# ~~~~~~~~~~~
y_noise = pyso.augment.add_white_noise(y, sr=sr)
pyso.plotsound(y_noise, sr=sr, feature_type = feature_type)
ipd.Audio(y_noise,rate=sr)


##########################################################
# Pitch Decrease
# ~~~~~~~~~~~~~~
y_pd = pyso.augment.pitch_decrease(y, sr=sr)
pyso.plotsound(y_pd, sr=sr, feature_type = feature_type)
ipd.Audio(y_pd,rate=sr)

##########################################################
# Pitch Increase
# ~~~~~~~~~~~~~~
y_pi = pyso.augment.pitch_increase(y, sr=sr)
pyso.plotsound(y_pi, sr=sr, feature_type = feature_type)
ipd.Audio(y_pi,rate=sr)

##########################################################
# Speed Decrease
# ~~~~~~~~~~~~~~
y_sd = pyso.augment.speed_decrease(y, sr=sr)
pyso.plotsound(y_sd, sr=sr, feature_type = feature_type)
ipd.Audio(y_sd,rate=sr)

##########################################################
# Speed Increase
# ~~~~~~~~~~~~~~
y_si = pyso.augment.speed_increase(y, sr=sr)
pyso.plotsound(y_si, sr=sr, feature_type = feature_type)
ipd.Audio(y_si,rate=sr)

##########################################################
# Combination
# ~~~~~~~~~~~
y_comb = pyso.augment.add_white_noise(y, sr=sr)
y_comb = pyso.augment.pitch_decrease(y_comb, sr=sr)
y_comb = pyso.augment.speed_increase(y_comb, sr=sr)
pyso.plotsound(y_comb, sr=sr, feature_type = feature_type)
ipd.Audio(y_comb,rate=sr)


#########################################################
# Adjust Augmentation Default Settings
# ------------------------------------

##########################################################
# Adjust Add White Noise
# ~~~~~~~~~~~~~~~~~~~~~~
# I want the SNR of the white noise to vary between several: 
# SNR 10, 15, and 20. 
augment_settings_dict['add_white_noise']['snr'] = [10,15,20]

##########################################################
# Adjust Pitch Decrease
# ~~~~~~~~~~~~~~~~~~~~~
# I found the pitch changes too exaggerated, so I will 
# set those to 1 instead of 2 semitones.  
augment_settings_dict['pitch_decrease']['num_semitones'] = 1 

##########################################################
# Adjust Pitch Increase
# ~~~~~~~~~~~~~~~~~~~~~
augment_settings_dict['pitch_increase']['num_semitones'] = 1 

##########################################################
# Adjust Speed Decrease
# ~~~~~~~~~~~~~~~~~~~~~
# Just for fun, we can see how it is if we change the speed 
# from 15% to 10% adjustment.
augment_settings_dict['speed_decrease']['perc'] = 0.1 

##########################################################
# Adjust Speed Increase
# ~~~~~~~~~~~~~~~~~~~~~
augment_settings_dict['speed_increase']['perc'] = 0.1 


##########################################################
# Listen with Settings Adjusted 
# -----------------------------

##########################################################
# White Noise Adjusted
# ~~~~~~~~~~~~~~~~~~~~
y_noise = pyso.augment.add_white_noise(y, sr=sr,
                                      **augment_settings_dict['add_white_noise'])
pyso.plotsound(y_noise, sr=sr, feature_type = feature_type)
ipd.Audio(y_noise,rate=sr)

##########################################################
# Pitch Decrease Adjusted
# ~~~~~~~~~~~~~~~~~~~~~~~
y_pd = pyso.augment.pitch_decrease(y, sr=sr,
                                      **augment_settings_dict['pitch_decrease'])
pyso.plotsound(y_pd, sr=sr, feature_type = feature_type)
ipd.Audio(y_pd,rate=sr)

##########################################################
# Pitch Increase Adjusted
# ~~~~~~~~~~~~~~~~~~~~~~~
y_pi = pyso.augment.pitch_increase(y, sr=sr,
                                      **augment_settings_dict['pitch_increase'])
pyso.plotsound(y_pi, sr=sr, feature_type = feature_type)
ipd.Audio(y_pi,rate=sr)

##########################################################
# Speed Decrease Adjusted
# ~~~~~~~~~~~~~~~~~~~~~~~
y_sd = pyso.augment.speed_decrease(y, sr=sr,
                                      **augment_settings_dict['speed_decrease'])
pyso.plotsound(y_sd, sr=sr, feature_type = feature_type)
ipd.Audio(y_sd,rate=sr)

##########################################################
# Speed Increase Adjusted
# ~~~~~~~~~~~~~~~~~~~~~~~
y_si = pyso.augment.speed_increase(y, sr=sr,
                                      **augment_settings_dict['speed_increase'])
pyso.plotsound(y_si, sr=sr, feature_type = feature_type)
ipd.Audio(y_si,rate=sr)

##########################################################
# combination Adjusted
# ~~~~~~~~~~~~~~~~~~~~
y_comb = pyso.augment.add_white_noise(y, sr=sr,
                                      **augment_settings_dict['add_white_noise'])
y_comb = pyso.augment.pitch_decrease(y_comb, sr=sr,
                                      **augment_settings_dict['pitch_decrease'])
y_comb = pyso.augment.speed_increase(y_comb, sr=sr,
                                      **augment_settings_dict['speed_increase'])
pyso.plotsound(y_comb, sr=sr, feature_type = feature_type)
ipd.Audio(y_comb,rate=sr)


#######################################################################
# Prepare for PySoundTool
# =======================
# PySoundTool works a lot with dictionaries. We will organize the augmentations 
# and settings in a list of dictionaries for PySoundTool to access during training.

#########################################################
# Update to Adjustments to Augmentation Dicts
# -------------------------------------------
# I will simply provide the same `augment_settings_dict` to each of the
# augmentation dicts as I don't want to change the settings for each 
# augmentation. One can easily do that though (e.g. if you want to 
# do multiple versions of speed increase at .1, .15, .2 or 10%, 15%, 20%
# respectively. You would just create an augmentation dict for each instance
# with a corresponding augmentation settings dict.

##########################################################
# Update Add White Noise
# ~~~~~~~~~~~~~~~~~~~~~~
augmentation_noise.update(
    dict(augment_settings_dict=augment_settings_dict))
for key, value in augmentation_noise.items():
    if isinstance(value, bool):
        print(key, ' : ', value)
    elif isinstance(value, dict):
        print(key, ':')
        for k, v in value.items():
            print('\t ',k,' : ',v)


##########################################################
# Update Pitch Decrease
# ~~~~~~~~~~~~~~~~~~~~~
augmentation_pitch_decrease.update(
    dict(augment_settings_dict=augment_settings_dict))
for key, value in augmentation_pitch_decrease.items():
    if isinstance(value, bool):
        print(key, ' : ', value)
    elif isinstance(value, dict):
        print(key, ':')
        for k, v in value.items():
            print('\t ',k,' : ',v)


##########################################################
# Update Pitch Increase
# ~~~~~~~~~~~~~~~~~~~~~
augmentation_pitch_increase.update(
    dict(augment_settings_dict=augment_settings_dict))
for key, value in augmentation_pitch_increase.items():
    if isinstance(value, bool):
        print(key, ' : ', value)
    elif isinstance(value, dict):
        print(key, ':')
        for k, v in value.items():
            print('\t ',k,' : ',v)


##########################################################
# Update Speed Decrease
# ~~~~~~~~~~~~~~~~~~~~~
augmentation_speed_decrease.update(
    dict(augment_settings_dict=augment_settings_dict))
for key, value in augmentation_speed_decrease.items():
    if isinstance(value, bool):
        print(key, ' : ', value)
    elif isinstance(value, dict):
        print(key, ':')
        for k, v in value.items():
            print('\t ',k,' : ',v)


##########################################################
# Update Speed Increase
# ~~~~~~~~~~~~~~~~~~~~~
augmentation_speed_increase.update(
    dict(augment_settings_dict=augment_settings_dict))
for key, value in augmentation_speed_increase.items():
    if isinstance(value, bool):
        print(key, ' : ', value)
    elif isinstance(value, dict):
        print(key, ':')
        for k, v in value.items():
            print('\t ',k,' : ',v)


##########################################################
# Update Combinations
# ~~~~~~~~~~~~~~~~~~~
augmentation_speed_increase_pitch_decrease_white_noise.update(
    dict(augment_settings_dict = augment_settings_dict))
for key, value in augmentation_speed_increase_pitch_decrease_white_noise.items():
    if isinstance(value, bool):
        print(key, ' : ', value)
    elif isinstance(value, dict):
        print(key, ':')
        for k, v in value.items():
            print('\t ',k,' : ',v)


#######################################################################
# Put All Augmentation Setting Dicts Into List
# ---------------------------------------------------------------------
augment_dict_list = [
    augmentation_none,
    augmentation_noise,
    augmentation_pitch_decrease,
    augmentation_pitch_increase,
    augmentation_speed_decrease,
    augmentation_speed_increase,
    augmentation_speed_increase_pitch_decrease_white_noise]


#########################################################
# Set Model Parameters
# --------------------
# This includes if it should expect labeled data (in this case yes), number of 
# epochs etc.
model_name = 'docs_augment_speech_commands'
labeled_data = True 
batch_size = 1

frames_per_sample = None   
# `frames_per_sample` is if you want to break up each audio file into several sections.
# This is somewhat related to 'context window' from research.

epochs = 15
# Note: for **each** set of augmentations, this number of epochs will be attempted.
# With our sets of augmentations, with 15 epochs, this totals to potentially 105 epochs.

patience = 10
# This sets when to stop training / apply early stopping:
# if there is no improvement in validation loss after so many epochs.

# Designate path relevant for accessing audiodata
data_dir = '../../../../mini-audio-datasets/speech_commands/'

#########################################################
# Get the Training Started!
# =========================

feat_model_dir, history = pyso.envclassifier_extract_train(
    model_name = model_name,
    audiodata_path = data_dir,
    augment_dict_list = augment_dict_list,
    labeled_data = labeled_data,
    batch_size = batch_size,
    frames_per_sample = frames_per_sample,
    epochs = epochs, 
    patience = patience,
    **get_feats_kwargs)

#########################################################
# Plot Training and Validation Accuracy 
# -------------------------------------
# load the logging settings of all training sessions:
import csv
model_log_path = feat_model_dir.joinpath('log.csv')
total_epochs = []
train_accuracy = []
train_loss = []
val_accuracy = []
val_loss = []
with open(model_log_path) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    counter = 0
    for row in csvReader:
        if counter == 0:
            # gets headers, unnecessary for plotting
            counter += 1
            continue
        total_epochs.append(counter)
        train_accuracy.append(round(float(row[1]),3))
        train_loss.append(round(float(row[2]),3))
        val_accuracy.append(round(float(row[3]),3))
        val_loss.append(round(float(row[4]),3))
        counter += 1

plt.plot(total_epochs, train_accuracy)
plt.plot(total_epochs, val_accuracy)
plt.title('model accuracy across {} training sessions'.format(
    len(augment_dict_list)))
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('{}_accuracy_{}.png'.format(model_name, pyso.utils.get_date()))


#########################################################
# Plot Training and Validation Loss 
# ---------------------------------
# Can you tell where each of the training sessions started?

plt.clf()
plt.plot(total_epochs, train_loss)
plt.plot(total_epochs, val_loss)
plt.title('model loss across {} training sessions'.format(
    len(augment_dict_list)))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('{}_loss_{}.png'.format(model_name, pyso.utils.get_date()))
