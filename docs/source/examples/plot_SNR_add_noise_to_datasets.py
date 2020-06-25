
# coding: utf-8
"""
==========================================
Add Noise to Speech at Specific SNR Levels
==========================================

Use PySoundTool to add noise at specific SNR levels to speech signals.
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
import pysoundtool as pyst;
import IPython.display as ipd


######################################################
# Define the speech and noise data samples
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

######################################################
# PySoundTool offers example datasets. Let's use them.

##########################################################
# Speech sample:
speech_sample = '{}audiodata/minidatasets/speech_commands/zero/c7aaad67_nohash_0.wav'.format(package_dir)
speech_sample = pyst.utils.string2pathlib(speech_sample)
print(speech_sample)
# as pathlib object, can do the following: 
word = speech_sample.parent.stem
word

##########################################################
# noise sample:
noise_sample = '{}audiodata/minidatasets/background_samples/cafe.wav'.format(package_dir)
print(noise_sample)
noise_sample = pyst.utils.string2pathlib(noise_sample)
# as pathlib object, can do the following: 
noise = noise_sample.stem
noise

##########################################################
# Hear and see what the speech looks like with SNR level of 20
noisyspeech_20snr, sr20, snr20 = pyst.dsp.add_backgroundsound(speech_sample,
                                           noise_sample,
                                           snr = 20)
ipd.Audio(noisyspeech_20snr,rate=sr20)

##########################################################
# Hear and see what the speech looks like with SNR level of 5
noisyspeech_5snr, sr5, snr5 = pyst.dsp.add_backgroundsound(speech_sample,
                                           noise_sample,
                                           snr = 5)
ipd.Audio(noisyspeech_5snr,rate=sr5)

######################################################################
# Visualize the Audio Samples
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

######################################################################
# Speech audio sample in its raw signal
pyst.plotsound(speech_sample, feature_type='signal', 
               title = 'Speech: ' + word.upper())

######################################################################
# Noise audio sample in its raw signal
pyst.plotsound(noise_sample, feature_type='signal',
               title = 'Noise: ' + noise.upper())

##########################################################
# The sounds added together at SNR level 20
pyst.plotsound(noisyspeech_20snr, sr = sr20, feature_type = 'signal',
               title = '"{}" with {} noise at SNR 20'.format(word.upper(), noise.upper()))

##########################################################
# The sounds added together at SNR level 5
pyst.plotsound(noisyspeech_5snr, sr = sr5, feature_type = 'signal',
               title = '"{}" with {} noise at SNR 5'.format(word.upper(), noise.upper()))

##########################################################
# Let's visualize the power spectrum: pure speech
pyst.plotsound(speech_sample, sr = sr20, feature_type = 'powspec',
               title = 'Clean speech: {}'.format(word),
               power_scale = 'power_to_db')


##########################################################
# Let's visualize the power spectrum: SNR 20
pyst.plotsound(noisyspeech_20snr, sr = sr20, feature_type = 'powspec',
               title = '"{}" with {} noise at SNR 20'.format(word.upper(), noise.upper()),
               power_scale = 'power_to_db')


##########################################################
# Let's visualize the power spectrum: SNR 5
pyst.plotsound(noisyspeech_5snr,sr = sr5, feature_type = 'powspec',
               title = '"{}" with {} noise at SNR 5'.format(word.upper(), noise.upper()),
               power_scale = 'power_to_db')

######################################################################
# More Functionality
# ^^^^^^^^^^^^^^^^^^

######################################################################
# Make longer and shorter
# ~~~~~~~~~~~~~~~~~~~~~~~

##########################################################
# Hear and see what the speech looks like:

##########################################################
# Delay the speech and lengthen the total signal
noisyspeech_20snr, sr20, snr20 = pyst.dsp.add_backgroundsound(speech_sample,
                                           noise_sample,
                                           snr = 20,
                                           delay_mainsound_sec = 1,
                                           total_len_sec = 4)

##########################################################
ipd.Audio(noisyspeech_20snr,rate=sr20)

##########################################################
pyst.plotsound(noisyspeech_20snr, sr = sr20, feature_type = 'signal',
               title = '"{}" with {} noise at SNR 20'.format(word.upper(), noise.upper()))


##########################################################
# Shorten the total signal
noisyspeech_20snr, sr20, snr20 = pyst.dsp.add_backgroundsound(speech_sample,
                                           noise_sample,
                                           snr = 20,
                                           total_len_sec = 0.5)

##########################################################
ipd.Audio(noisyspeech_20snr,rate=sr20)

##########################################################
pyst.plotsound(noisyspeech_20snr, sr = sr20, feature_type = 'signal',
               title = '"{}" with {} noise at SNR 20'.format(word.upper(), noise.upper()))
