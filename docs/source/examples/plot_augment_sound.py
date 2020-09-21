
# coding: utf-8
"""
======================================================
Augment Speech and Sound for Machine and Deep Learning
======================================================

Augment audio to expanding datasets and train resilient models.

To see how SoundPy implements this, see the module `soundpy.augment`.


Note:
~~~~~
Consideration of what type of sound one is working with must be taken when performing augmentation. Not all speech and non-speech sounds should be handled the same. For example, you may want to augment speech differently if you are training a speech recognition model versus an emotion recognition model. Additionally, not all non-speech sounds behave the same, for example stationary (white noise) vs non-stationary (car horn) sounds.

In sum, awareness of how your sound data behave and what features of the sound are relevant for training models are important factors for sound data augmentation. 

Below are a few augmentation techniques I have seen implemented in sound research; this is in no way a complete list of augmentation techniques.
"""

###############################################################################################
# 


#####################################################################
import soundpy as sp
import IPython.display as ipd

#############################################
# Augmenting Speech
# ^^^^^^^^^^^^^^^^^

##########################################################
# Designate the path relevant for accessing audiodata
# Note: the speech and sound come with the soundpy repo.
sp_dir = '../../../'

##########################################################
# Speech sample:
speech = '{}audiodata/python.wav'.format(sp_dir)
speech = sp.utils.string2pathlib(speech)

################################################################
# Hear and see speech
# ~~~~~~~~~~~~~~~~~~~

sr = 44100
f, sr = sp.loadsound(speech, sr=sr)
ipd.Audio(f,rate=sr)

##########################################################
sp.plotsound(f, sr=sr, feature_type='stft', title='Female Speech: "Python"', sub_process=True)

##########################################################
# Change Speed
# ~~~~~~~~~~~~

##########################################################
# Let's increase the speed by 15%:

fast = sp.augment.speed_increase(f, sr=sr, perc = 0.15) 

##########################################################
ipd.Audio(fast,rate=sr)

##########################################################
sp.plotsound(fast, sr = sr, feature_type = 'stft', 
               title = 'Female speech: 15%  faster',
               sub_process=True)

##########################################################
# Let's decrease the speed by 15%:

slow = sp.augment.speed_decrease(f, sr = sr, perc = 0.15) 

##########################################################
ipd.Audio(slow, rate = sr)

##########################################################
sp.plotsound(slow, sr = sr, feature_type = 'stft', 
               title = 'Speech: 15%  slower', sub_process=True)


##########################################################
# Add Noise
# ~~~~~~~~~

##########################################################
# Add white noise: 10 SNR

noisy = sp.augment.add_white_noise(f, sr=sr, snr = 10) 

##########################################################
ipd.Audio(noisy,rate=sr)

##########################################################
sp.plotsound(noisy, sr=sr, feature_type='stft', 
               title='Speech with white noise: 10 SNR', sub_process=True)


##########################################################
# Harmonic Distortion
# ~~~~~~~~~~~~~~~~~~~

hd = sp.augment.harmonic_distortion(f, sr=sr) 

##########################################################
ipd.Audio(hd,rate=sr)

##########################################################
sp.plotsound(hd, sr=sr, feature_type='stft', 
               title='Speech with harmonic distortion', sub_process=True)


##########################################################
# Pitch Shift
# ~~~~~~~~~~~

##########################################################
# Pitch shift increase

psi = sp.augment.pitch_increase(f, sr=sr, num_semitones = 2) 

##########################################################
ipd.Audio(psi,rate=sr)

##########################################################
sp.plotsound(psi, sr=sr, feature_type='stft', 
               title='Speech with pitch shift increase', sub_process=True)

##########################################################
# Pitch shift decrease

psd = sp.augment.pitch_decrease(f, sr=sr, num_semitones = 2) 

##########################################################
ipd.Audio(psd,rate=sr)

##########################################################
sp.plotsound(psd, sr=sr, feature_type='stft', 
               title='Speech with pitch shift decrease', sub_process=True)



##########################################################
# Vocal Tract Length Perturbation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Note: this is still experimental.
##########################################################
# Vocal tract length perturbation (by factor 0.8 to 1.2)

vtlp_stft, a = sp.augment.vtlp(f, sr=sr, win_size_ms = 50,
                                 percent_overlap = 0.5,
                                 random_seed = 41) 

##########################################################
# In order to listen to this, we need to turn the stft into 
# samples:
vtlp_y = sp.feats.feats2audio(vtlp_stft, sr = sr,
                                feature_type = 'stft',
                                win_size_ms = 50,
                                percent_overlap = 0.5)
ipd.Audio(vtlp_y,rate=sr)

##########################################################
sp.feats.plot(vtlp_stft, sr=sr, feature_type='stft', 
               title='VTLP (factor {})'.format(a), sub_process=True)

##########################################################
# Vocal tract length perturbation (by factor 0.8 to 1.2)

vtlp_stft, a = sp.augment.vtlp(f, sr=sr, win_size_ms = 50,
                                 percent_overlap = 0.5,
                                 random_seed = 43) 

##########################################################
# In order to listen to this, we need to turn the stft into 
# samples:
vtlp_y = sp.feats.feats2audio(vtlp_stft, sr = sr,
                                feature_type = 'stft',
                                win_size_ms = 50,
                                percent_overlap = 0.5)
ipd.Audio(vtlp_y,rate=sr)

##########################################################
sp.feats.plot(vtlp_stft, sr=sr, feature_type='stft', 
               title='VTLP (factor {})'.format(a), sub_process=True)


#############################################
# Augmenting non-speech signals
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Car horn sample:
honk = '{}audiodata/car_horn.wav'.format(sp_dir)
honk = sp.utils.string2pathlib(honk)

##########################################################
# Hear and see sound signal 
# ~~~~~~~~~~~~~~~~~~~~~~~~~
h, sr = sp.loadsound(honk, sr=sr)
ipd.Audio(h,rate=sr)

##########################################################
sp.plotsound(h, sr=sr, feature_type='stft', 
               title='Car Horn', sub_process=True)

##########################################################
# Change Speed
# ~~~~~~~~~~~~

##########################################################
# Let's increase the speed by 15%:

fast = sp.augment.speed_increase(h, sr=sr, perc = 0.15) 

##########################################################
ipd.Audio(fast,rate=sr)

##########################################################
sp.plotsound(fast, sr=sr, feature_type='stft', 
               title='Car horn: 15%  faster', sub_process=True)

##########################################################
# Let's decrease the speed by 15%:

slow = sp.augment.speed_decrease(h, sr=sr, perc = 0.15) 

##########################################################
ipd.Audio(slow,rate=sr)

##########################################################
sp.plotsound(slow, sr=sr, feature_type='stft', 
               title='Car horn: 15%  slower', sub_process=True)

##########################################################
# Add Noise
# ~~~~~~~~~

##########################################################
# Add white noise 
h_noisy = sp.augment.add_white_noise(h, sr=sr, snr = 10)

##########################################################
ipd.Audio(h_noisy,rate=sr)

##########################################################
sp.plotsound(h_noisy, sr=sr, feature_type='stft', 
               title='Car horn with white noise (10 SNR)', 
               sub_process=True)

##########################################################
# Harmonic Distortion
# ~~~~~~~~~~~~~~~~~~~

hd = sp.augment.harmonic_distortion(h, sr=sr) 

##########################################################
ipd.Audio(hd,rate=sr)

##########################################################
sp.plotsound(hd, sr=sr, feature_type='stft', 
               title='Car horn with harmonic distortion', 
               sub_process=True)


##########################################################
# Pitch Shift
# ~~~~~~~~~~~

##########################################################
# Pitch shift increase

psi = sp.augment.pitch_increase(h, sr=sr, num_semitones = 2) 

##########################################################
ipd.Audio(psi,rate=sr)

##########################################################
sp.plotsound(psi, sr=sr, feature_type='stft', 
               title='Car horn with pitch shift increase', 
               sub_process=True)

##########################################################
# Pitch shift decrease

psd = sp.augment.pitch_decrease(h, sr=sr, num_semitones = 2) 

##########################################################
ipd.Audio(psd,rate=sr)

##########################################################
sp.plotsound(psd, sr=sr, feature_type='stft', 
               title='Car horn with pitch shift decrease', 
               sub_process=True)

##########################################################
# Time Shift
# ~~~~~~~~~~

##########################################################
# We'll apply a random shift to the sound
h_shift = sp.augment.time_shift(h, sr=sr)

##########################################################
ipd.Audio(h_shift,rate=sr)

##########################################################
sp.plotsound(h_shift, sr=sr, feature_type='stft', 
               title='Car horn: time shifted', 
               sub_process=True)


##########################################################
# Shuffle the Sound
# ~~~~~~~~~~~~~~~~~

h_shuffle = sp.augment.shufflesound(h, sr=sr,
                                      num_subsections = 5)

##########################################################
ipd.Audio(h_shuffle,rate=sr)

##########################################################
sp.plotsound(h_shuffle, sr=sr, feature_type='stft', 
               title='Car horn: shuffled', sub_process=True)


##########################################################
# Just for kicks let's do the same to speech and see how 
# that influences the signal:

h_shuffle = sp.augment.shufflesound(f, sr=sr,
                                      num_subsections = 5)

##########################################################
ipd.Audio(h_shuffle,rate=sr)

##########################################################
sp.plotsound(h_shuffle, sr=sr, feature_type='stft', 
               title='Speech: shuffled ', sub_process=True)
