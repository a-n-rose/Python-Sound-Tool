
# coding: utf-8
"""
==========================================
Add Noise to Speech at Specific SNR Levels
==========================================

Use PySoundTool to add noise at specific SNR levels to speech signals.

To see how PySoundTool implements this, see `pysoundtool.dsp.add_backgroundsound`.
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
import pysoundtool as pyso;
import IPython.display as ipd


######################################################
# Define the speech and noise data samples
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

######################################################
# I will use speech data from a mini dataset on my computer.

##########################################################
# Speech sample:
speech_sample = '{}../mini-audio-datasets/speech_commands/zero/c7aaad67_nohash_0.wav'.format(package_dir)
speech_sample = pyso.utils.string2pathlib(speech_sample)
print(speech_sample)
# as pathlib object, can do the following: 
word = speech_sample.parent.stem
word

##########################################################
# noise sample: (this comes with PySoundTool)
noise_sample = '{}audiodata/background_samples/cafe.wav'.format(package_dir)
print(noise_sample)
noise_sample = pyso.utils.string2pathlib(noise_sample)
# as pathlib object, can do the following: 
noise = noise_sample.stem
noise

##########################################################
# Hear and see what the speech looks like with SNR level of 20
sr = 16000
noisyspeech_20snr, snr20 = pyso.dsp.add_backgroundsound(speech_sample,
                                           noise_sample,
                                           sr = sr,
                                           snr = 20)
ipd.Audio(noisyspeech_20snr,rate=sr)

##########################################################
# Hear and see what the speech looks like with SNR level of 5
noisyspeech_5snr, snr5 = pyso.dsp.add_backgroundsound(speech_sample,
                                           noise_sample,
                                           sr = sr,
                                           snr = 5)
ipd.Audio(noisyspeech_5snr,rate=sr)

######################################################################
# Visualize the Audio Samples
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

######################################################################
# Speech audio sample in its raw signal
pyso.plotsound(speech_sample, feature_type='signal', 
               title = 'Speech: ' + word.upper())

######################################################################
# Noise audio sample in its raw signal
pyso.plotsound(noise_sample, feature_type='signal',
               title = 'Noise: ' + noise.upper())

##########################################################
# The sounds added together at SNR level 20
pyso.plotsound(noisyspeech_20snr, sr = sr, feature_type = 'signal',
               title = '"{}" with {} noise at SNR 20'.format(word.upper(), noise.upper()))

##########################################################
# The sounds added together at SNR level 5
pyso.plotsound(noisyspeech_5snr, sr = sr, feature_type = 'signal',
               title = '"{}" with {} noise at SNR 5'.format(word.upper(), noise.upper()))

##########################################################
# Let's visualize the power spectrum: pure speech
pyso.plotsound(speech_sample, sr = sr, feature_type = 'powspec',
               title = 'Clean speech: {}'.format(word),
               energy_scale = 'power_to_db')


##########################################################
# Let's visualize the power spectrum: SNR 20
pyso.plotsound(noisyspeech_20snr, sr = sr, feature_type = 'powspec',
               title = '"{}" with {} noise at SNR 20'.format(word.upper(), noise.upper()),
               energy_scale = 'power_to_db')


##########################################################
# Let's visualize the power spectrum: SNR 5
pyso.plotsound(noisyspeech_5snr,sr = sr, feature_type = 'powspec',
               title = '"{}" with {} noise at SNR 5'.format(word.upper(), noise.upper()),
               energy_scale = 'power_to_db')

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
noisyspeech_20snr, snr20 = pyso.dsp.add_backgroundsound(speech_sample,
                                           noise_sample,
                                           sr = sr,
                                           snr = 20,
                                           delay_mainsound_sec = 1,
                                           total_len_sec = 4)

##########################################################
ipd.Audio(noisyspeech_20snr,rate=sr)

##########################################################
pyso.plotsound(noisyspeech_20snr, sr = sr, feature_type = 'signal',
               title = '"{}" with {} noise at SNR 20'.format(word.upper(), noise.upper()))


##########################################################
# Shorten the total signal
noisyspeech_20snr, snr20 = pyso.dsp.add_backgroundsound(speech_sample,
                                           noise_sample,
                                           sr = sr,
                                           snr = 20,
                                           total_len_sec = 0.5)

##########################################################
ipd.Audio(noisyspeech_20snr,rate=sr)

##########################################################
pyso.plotsound(noisyspeech_20snr, sr = sr, feature_type = 'signal',
               title = '"{}" with {} noise at SNR 20'.format(word.upper(), noise.upper()))
