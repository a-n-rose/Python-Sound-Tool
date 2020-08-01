
# coding: utf-8
"""
==========================================
Add Noise to Speech at Specific SNR Levels
==========================================

Add noise to speech at specific signal-to-noise ratio levels.

To see how PySoundTool implements this, see `pysoundtool.dsp.add_backgroundsound`.
"""


###############################################################################################
# 


#####################################################################
# Let's import pysoundtool, and ipd for playing audio data
import pysoundtool as pyso
import IPython.display as ipd


######################################################
# Define the speech and noise data samples
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

######################################################
# I will use speech and noise data from the PySoundTool repo.

##########################################################
# Designate path relevant for accessing audiodata
pyso_dir = '../../../'

##########################################################
# Speech sample:
speech_sample = '{}audiodata/python.wav'.format(pyso_dir)
speech_sample = pyso.utils.string2pathlib(speech_sample)
# as pathlib object, can do the following: 
word = speech_sample.stem
word

##########################################################
# Noise sample:
noise_sample = '{}audiodata/background_samples/cafe.wav'.format(pyso_dir)
noise_sample = pyso.utils.string2pathlib(noise_sample)
# as pathlib object, can do the following: 
noise = noise_sample.stem
noise


##########################################################
# Hear Clean Speech
# ~~~~~~~~~~~~~~~~~
# I'm using a higher sample rate here as calculating SNR 
# performs best upwards of 44100 Hz.
sr = 48000
s, sr = pyso.loadsound(speech_sample, sr = sr)
ipd.Audio(s,rate=sr)


##########################################################
# Hear Noise
# ~~~~~~~~~~
n, sr = pyso.loadsound(noise_sample, sr = sr)
ipd.Audio(n,rate=sr)


##########################################################
# Hear Signal-to-Noise Ratio 20
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
noisyspeech_20snr, snr20 = pyso.dsp.add_backgroundsound(
    speech_sample,
    noise_sample,
    sr = sr,
    snr = 20)
ipd.Audio(noisyspeech_20snr,rate=sr)

########################################################
# `snr20` is simply the measured SNR post adjustment fo the noise signal.
# This is useful to check that the indicated snr is at least close
# to the resulting snr.
snr20

##########################################################
# Hear Signal-to-Noise Ratio 5
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
noisyspeech_5snr, snr5 = pyso.dsp.add_backgroundsound(
    speech_sample,
    noise_sample,
    sr = sr,
    snr = 5)
ipd.Audio(noisyspeech_5snr,rate=sr)

#########################################################
snr5

######################################################################
# Visualize the Audio Samples
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

######################################################################
# See Clean Speech (raw signal)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyso.plotsound(speech_sample, feature_type='signal', 
               sr = sr, title = 'Speech: ' + word.upper())

######################################################################
# See Clean Speech (stft)
# ~~~~~~~~~~~~~~~~~~~~~~~
pyso.plotsound(speech_sample, feature_type='stft', 
               sr = sr, title = 'Speech: ' + word.upper())

###################################################################### See Noise (raw signal)
# ~~~~~~~~~~~~~~~~~~~~~~
pyso.plotsound(noise_sample, feature_type='signal',
               title = 'Noise: ' + noise.upper())

###################################################################### See Noise (stft)
# ~~~~~~~~~~~~~~~~
pyso.plotsound(noise_sample, feature_type='stft',
               title = 'Noise: ' + noise.upper())

######################################################################
# See Noisy Speech: SNR 20 (raw signal)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyso.plotsound(noisyspeech_20snr, sr = sr, feature_type = 'signal',
               title = '"{}" with {} noise at SNR 20'.format(word.upper(), noise.upper()))

######################################################################
# See Noisy Speech: SNR 20 (stft)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyso.plotsound(noisyspeech_20snr, sr = sr, feature_type = 'stft',
               title = '"{}" with {} noise at SNR 20'.format(word.upper(), noise.upper()))

######################################################################
# See Noisy Speech: SNR 5 (raw signal)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyso.plotsound(noisyspeech_5snr, sr = sr, feature_type = 'signal',
               title = '"{}" with {} noise at SNR 5'.format(word.upper(), noise.upper()))

######################################################################
# See Noisy Speech: SNR 5 (stft)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyso.plotsound(noisyspeech_20snr, sr = sr, feature_type = 'stft',
               title = '"{}" with {} noise at SNR 5'.format(word.upper(), noise.upper()))

######################################################################
# Make Combined Sound Longer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

##########################################################
# Pad Speech and Set Total Length 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
noisyspeech_20snr, snr20 = pyso.dsp.add_backgroundsound(
    speech_sample,
    noise_sample,
    sr = sr,
    snr = 20,
    pad_mainsound_sec = 1,
    total_len_sec = 4)

##########################################################
ipd.Audio(noisyspeech_20snr,rate=sr)

##########################################################
pyso.plotsound(noisyspeech_20snr, sr = sr, feature_type = 'signal',
               title = '"{}" with {} noise at SNR 20'.format(word.upper(), noise.upper()))


######################################################################
# Make Combined Sound Shorter
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

##########################################################
# Set Total Length
# ~~~~~~~~~~~~~~~~
noisyspeech_20snr, snr20 = pyso.dsp.add_backgroundsound(
    speech_sample,
    noise_sample,
    sr = sr,
    snr = 20,
    total_len_sec = 0.5)

##########################################################
ipd.Audio(noisyspeech_20snr,rate=sr)

##########################################################
pyso.plotsound(noisyspeech_20snr, sr = sr, feature_type = 'signal',
               title = '"{}" with {} noise at SNR 20'.format(word.upper(), noise.upper()))

######################################################################
# Wrap the Background Sound
# ^^^^^^^^^^^^^^^^^^^^^^^^^
noisyspeech_20snr, snr20 = pyso.dsp.add_backgroundsound(
    speech_sample,
    noise_sample,
    sr = sr,
    snr = 20,
    wrap = True,
    pad_mainsound_sec = 2,
    total_len_sec = 5)

##########################################################
ipd.Audio(noisyspeech_20snr,rate=sr)

##########################################################
pyso.plotsound(noisyspeech_20snr, sr = sr, feature_type = 'signal',
               title = '"{}" with {} noise at SNR 20'.format(word.upper(), noise.upper()))

