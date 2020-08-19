# coding: utf-8
"""
=======================
Create and Plot Signals
=======================

Create and plot signals / noise; combine them at a specific SNR.

To see how soundpy implements this, see `soundpy.dsp.generate_sound`, 
`soundpy.dsp.generate_noise` and `soundpy.dsp.add_backgroundsound`.
"""


###############################################################################################
#  

#####################################################################
# Let's import soundpy
import soundpy as sp

###########################################################################
# Create a Signal
# ^^^^^^^^^^^^^^^

########################################################################
# First let's set what sample rate we want to use
sr = 44100


#########################################################################
# Let's create a signal of 10 Hz 
sig1_hz = 10
sig1, sr = sp.generate_sound(freq=sig1_hz, amplitude = 0.4, sr=sr, dur_sec=1)
sp.plotsound(sig1, sr=sr, feature_type = 'signal',
               title = 'Signal: {} Hz'.format(sig1_hz))


#########################################################################
# Let's create a signal of 20 Hz
sig2_hz = 20 
sig2, sr = sp.generate_sound(freq=sig2_hz, amplitude= 0.4, sr=sr, dur_sec=1)
sp.plotsound(sig2, sr=sr, feature_type = 'signal',
               title = 'Signal: {} Hz'.format(sig2_hz))

###########################################################################
# Combine Signals 
# ^^^^^^^^^^^^^^^


#########################################################################
# Add them together and see what they look like:
sig3 = sig1 + sig2
sp.plotsound(sig3, sr=sr, feature_type = 'signal', 
               title='Mixed Signals: {} Hz + {} Hz'.format(sig1_hz, sig2_hz))


##########################################################################
# Generate Pseudo-Random Noise
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#########################################################################
# Create noise to add to the signal:
noise = sp.generate_noise(len(sig3), amplitude=0.02, random_seed=40)
sp.plotsound(noise, sr=sr, feature_type = 'signal',
               title='Random Noise')

###########################################################################
# Control SNR: Adding a Background Sound
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#########################################################################
# Add noise at signal-to-noise ratio of 40
sig_noisy, snr = sp.dsp.add_backgroundsound(
    audio_main = sig3, 
    audio_background = noise, 
    sr = sr,
    snr = 40,
    clip_at_zero = False)

# keep energy between 1 and -1 
sig_noisy = sp.dsp.scalesound(sig_noisy, max_val=1)
sp.plotsound(sig_noisy, sr=sr, feature_type = 'signal', title='Signal + Noise: 40 SNR')

#########################################################################
# Add noise at signal-to-noise ratio of 20
sig_noisy, snr = sp.dsp.add_backgroundsound(
    audio_main = sig3, 
    audio_background = noise,
    sr = sr,
    snr = 20)
# keep energy between 1 and -1 
sig_noisy = sp.dsp.scalesound(sig_noisy, max_val=1)
sp.plotsound(sig_noisy, sr=sr, feature_type = 'signal', title='Signal + Noise: 20 SNR')

#########################################################################
# Add noise at signal-to-noise ratio of 10
sig_noisy, snr = sp.dsp.add_backgroundsound(
    audio_main = sig3, 
    audio_background = noise,
    sr = sr,
    snr = 10)
# keep energy between 1 and -1 
sig_noisy = sp.dsp.scalesound(sig_noisy, max_val=1)
sp.plotsound(sig_noisy, sr=sr, feature_type = 'signal', title='Signal + Noise: 10 SNR')

#########################################################################
# Add noise at signal-to-noise ratio of 0
sig_noisy, snr = sp.dsp.add_backgroundsound(
    audio_main = sig3,
    audio_background = noise,
    sr = sr,
    snr = 0)
# keep energy between 1 and -1 
sig_noisy = sp.dsp.scalesound(sig_noisy, max_val=1)
sp.plotsound(sig_noisy, sr=sr, feature_type = 'signal', title='Signal + Noise: 0 SNR')


#########################################################################
# Add noise at signal-to-noise ratio of -10
sig_noisy, snr = sp.dsp.add_backgroundsound(
    audio_main = sig3, 
    audio_background = noise,
    sr = sr,
    snr = -10)
# keep energy between 1 and -1 
sig_noisy = sp.dsp.scalesound(sig_noisy, max_val=1)
sp.plotsound(sig_noisy, sr=sr, feature_type = 'signal', title='Signal + Noise: -10 SNR')
