# coding: utf-8
"""
==================================================
Extract, Augment, and Train an Acoustic Classifier
==================================================

Extract and augment features as an acoustic classifier is trained on speech.

To see how soundpy implements this, see `soundpy.models.builtin.envclassifier_extract_train`.
"""

###############################################################################################
#

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parparentdir = os.path.dirname(parentdir)
packagedir = os.path.dirname(parparentdir)
sys.path.insert(0, packagedir)

import matplotlib.pyplot as plt
import IPython.display as ipd
package_dir = '../../../'
os.chdir(package_dir)
sp_dir = package_dir


#####################################################################
# Let's import soundpy for handling sound
import soundpy as sp
#####################################################################
# As well as the deep learning component of soundpy
from soundpy import models as spdl


######################################################
# Prepare for Training: Data Organization
# =======================================

######################################################
# I will use a sample speech commands data set:

##########################################################
# Designate path relevant for accessing audiodata
data_dir = '{}../mini-audio-datasets/speech_commands/'.format(sp_dir)


######################################################
# Setup a Feature Settings Dictionary
# -----------------------------------


feature_type = 'fbank'
num_filters = 40
rate_of_change = False
rate_of_acceleration = False
dur_sec = 1
win_size_ms = 25
percent_overlap = 0.5
sr = 22050
fft_bins = None
num_mfcc = None
real_signal = True

get_feats_kwargs = dict(feature_type = feature_type,
                        sr = sr,
                        dur_sec = dur_sec,
                        win_size_ms = win_size_ms,
                        percent_overlap = percent_overlap,
                        fft_bins = fft_bins,
                        num_filters = num_filters,
                        num_mfcc = num_mfcc,
                        rate_of_change = rate_of_change,
                        rate_of_acceleration = rate_of_acceleration,
                        real_signal = real_signal)

######################################################
# Setup an Augmentation Dictionary
# --------------------------------
# This will apply augmentations at random at each epoch.
augmentation_all = dict([('add_white_noise',True),
                        ('speed_decrease', True),
                        ('speed_increase', True),
                        ('pitch_decrease', True),
                        ('pitch_increase', True),
                        ('harmonic_distortion', True),
                        ('vtlp', True)
                        ])

##########################################################
# see the default values for these augmentations
augment_settings_dict = {}
for key in augmentation_all.keys():
    augment_settings_dict[key] = sp.augment.get_augmentation_settings_dict(key)
for key, value in augment_settings_dict.items():
    print(key, ' : ', value)
    
##########################################################
# Adjust Augmentation Defaults
# ----------------------------


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
augment_settings_dict['speed_decrease']['perc'] = 0.1 

##########################################################
# Adjust Speed Increase
# ~~~~~~~~~~~~~~~~~~~~~
augment_settings_dict['speed_increase']['perc'] = 0.1 


######################################################
# Update an Augmentation Dictionary
# ---------------------------------
# We'll include in the dictionary the settings we want for augmentations:
augmentation_all.update(
    dict(augment_settings_dict = augment_settings_dict))


######################################################
# Train the Model
# ===============
# Note: disregard the warning:
# WARNING: Only the power spectrum of the VTLP augmented signal can be returned due to resizing the augmentation from (56, 4401) to (79, 276)
# 
# This is due to the hyper frequency resolution applied to the audio during 
# vocal-tract length perturbation, and then deresolution to bring to correct size.
# The current implementation applies the deresolution to the power spectrum rather than
# directly to the STFT. 
model_dir, history = spdl.envclassifier_extract_train(
    model_name = 'augment_builtin',
    audiodata_path = data_dir,
    augment_dict = augmentation_all,
    labeled_data = True,
    batch_size = 1,
    epochs = 10, 
    patience = 5,
    visualize = True,
    vis_every_n_items = 1,
    **get_feats_kwargs)

#############################################################
# Let's plot how the model performed (on this small dataset)
plt.clf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('accuracy.png')
