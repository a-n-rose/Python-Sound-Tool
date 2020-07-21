
# coding: utf-8
"""
========================
Augment Speech and Sound
========================

Use PySoundTool to augment audio signals. 

To see how PySoundTool implements this, see `pysoundtool.augment`.
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


#############################################
# Let's work with speech and sound (car horn)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##########################################################
# Speech sample:
speech = '{}audiodata/python.wav'.format(package_dir)
speech = pyso.utils.string2pathlib(speech)
# Car horn sample:
honk = '{}audiodata/car_horn.wav'.format(package_dir)
honk = pyso.utils.string2pathlib(honk)


################################################################
# Hear and see speech (later we'll examine the non-speech sound)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sr = 16000
f, sr = pyso.loadsound(speech, sr=sr)
ipd.Audio(f,rate=sr)

##########################################################
pyso.plotsound(f, sr=sr, feature_type='stft', title='Female Speech')


##########################################################
# Augmentation appropriate for speech signals 
# -------------------------------------------

##########################################################
# Change Speed
# ~~~~~~~~~~~~

##########################################################
# Let's increase the speed by 15%:

fast = pyso.augment.speed_increase(f, sr=sr, perc = 0.15) 

##########################################################
ipd.Audio(fast,rate=sr)

##########################################################
pyso.plotsound(fast, sr=sr, feature_type='stft', 
               title='Female speech: 15%  faster')

##########################################################
# Let's decrease the speed by 15%:

slow = pyso.augment.speed_decrease(f, sr=sr, perc = 0.15) 

##########################################################
ipd.Audio(slow,rate=sr)

##########################################################
pyso.plotsound(slow, sr=sr, feature_type='stft', 
               title='Speech: 15%  slower')


##########################################################
# Add Noise
# ~~~~~~~~~

##########################################################
# Add white noise: 10 SNR

noisy = pyso.augment.add_white_noise(f, sr=sr, snr = 10) 

##########################################################
ipd.Audio(noisy,rate=sr)

##########################################################
pyso.plotsound(noisy, sr=sr, feature_type='stft', 
               title='Speech with white noise: 10 SNR')


##########################################################
# Harmonic Distortion
# ~~~~~~~~~~~~~~~~~~~

hd = pyso.augment.harmonic_distortion(f, sr=sr) 

##########################################################
ipd.Audio(hd,rate=sr)

##########################################################
pyso.plotsound(hd, sr=sr, feature_type='stft', 
               title='Speech with harmonic distortion')


##########################################################
# Pitch Shift
# ~~~~~~~~~~~

##########################################################
# Pitch shift increase

psi = pyso.augment.pitch_increase(f, sr=sr, num_semitones = 2) 

##########################################################
ipd.Audio(psi,rate=sr)

##########################################################
pyso.plotsound(psi, sr=sr, feature_type='stft', 
               title='Speech with pitch shift increase')

##########################################################
# Pitch shift decrease

psd = pyso.augment.pitch_decrease(f, sr=sr, num_semitones = 2) 

##########################################################
ipd.Audio(psd,rate=sr)

##########################################################
pyso.plotsound(psd, sr=sr, feature_type='stft', 
               title='Speech with pitch shift decrease')



##########################################################
# Vocal Tract Length Perturbation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##########################################################
# Vocal tract length perturbation (by factor 0.8 to 1.2)

vtlp_stft, a = pyso.augment.vtlp(f, sr=sr, win_size_ms = 50,
                                 percent_overlap = 0.5,
                                 random_seed = 41) 

##########################################################
# In order to listen to this, we need to turn the stft into 
# samples:
vtlp_y = pyso.feats.feats2audio(vtlp_stft, sr = sr,
                                feature_type = 'stft',
                                win_size_ms = 50,
                                percent_overlap = 0.5)
ipd.Audio(vtlp_y,rate=sr)

##########################################################
pyso.feats.plot(vtlp_stft, sr=sr, feature_type='stft', 
               title='VTLP (factor {})'.format(a))

##########################################################
# Vocal tract length perturbation (by factor 0.8 to 1.2)

vtlp_stft, a = pyso.augment.vtlp(f, sr=sr, win_size_ms = 50,
                                 percent_overlap = 0.5,
                                 random_seed = 43) 

##########################################################
# In order to listen to this, we need to turn the stft into 
# samples:
vtlp_y = pyso.feats.feats2audio(vtlp_stft, sr = sr,
                                feature_type = 'stft',
                                win_size_ms = 50,
                                percent_overlap = 0.5)
ipd.Audio(vtlp_y,rate=sr)

##########################################################
pyso.feats.plot(vtlp_stft, sr=sr, feature_type='stft', 
               title='VTLP (factor {})'.format(a))


##########################################################
# Augmentation appropriate for non-speech signals 
# -----------------------------------------------


##########################################################
# Hear and see sound signal 
# ~~~~~~~~~~~~~~~~~~~~~~~~~

sr = 16000
h, sr = pyso.loadsound(honk, sr=sr)
ipd.Audio(h,rate=sr)

##########################################################
pyso.plotsound(h, sr=sr, feature_type='stft', 
               title='Car Horn Sound')


##########################################################
# Time Shift
# ~~~~~~~~~~

##########################################################
# We'll apply a random shift to the sound
h_shift = pyso.augment.time_shift(h, sr=sr)

##########################################################
ipd.Audio(h_shift,rate=sr)

##########################################################
pyso.plotsound(h_shift, sr=sr, feature_type='stft', 
               title='Car horn: time shifted')


##########################################################
# Shuffle the Sound
# ~~~~~~~~~~~~~~~~~

h_shuffle = pyso.augment.shufflesound(h, sr=sr,
                                      num_subsections = 5)

##########################################################
ipd.Audio(h_shuffle,rate=sr)

##########################################################
pyso.plotsound(h_shuffle, sr=sr, feature_type='stft', 
               title='Car horn: shuffled')


##########################################################
# Add Noise
# ~~~~~~~~~

##########################################################
# Add white noise 
h_noisy = pyso.augment.add_white_noise(h, sr=sr, snr = 10)

##########################################################
ipd.Audio(h_noisy,rate=sr)

##########################################################
pyso.plotsound(h_noisy, sr=sr, feature_type='stft', 
               title='Car horn with white noise (10 SNR)')

