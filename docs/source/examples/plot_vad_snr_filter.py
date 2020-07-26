
# coding: utf-8
"""
===========================================
Voice Activity Detection (upcoming release)
===========================================

Plot the VAD in signals and remove silences.

To see how PySoundTool implements this, see `pysoundtool.dsp.vad`, 
`pysoundtool.feats.get_vad_samples`, `pysoundtool.feats.get_vad_stft`, 
and `pysoundtool.feats.plot_vad`.

This is scheduled to release with the next pypi package release in August/ September 2020.
"""


###############################################################################################
# 


#####################################################################

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parparentdir = os.path.dirname(parentdir)
packagedir = os.path.dirname(parparentdir)
sys.path.insert(0, packagedir)

import pysoundtool as pyso 
import numpy as np
import IPython.display as ipd

package_dir = '../../../'
os.chdir(package_dir)
pyso_dir = package_dir
######################################################
# Load sample speech audio
# ^^^^^^^^^^^^^^^^^^^^^^^^
# Note: this file is available in the PySoundTool repo.
speech = '{}audiodata/python.wav'.format(pyso_dir)
# VAD and filtering work best with high sample rates (48000)
y, sr = pyso.loadsound(speech, sr=48000)
ipd.Audio(y,rate=sr)

######################################################
# Generate white noise as background noise
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
white_noise = pyso.dsp.generate_noise(len(y), random_seed = 40)

######################################################
# Generate speech audio at various SNR levels
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


######################################################
# Speech and Noise SNR 20
# ~~~~~~~~~~~~~~~~~~~~~~~
y_snr20, snr20 = pyso.dsp.add_backgroundsound(y, white_noise, sr=sr, snr = 20,
                                           delay_mainsound_sec = 1,
                                           total_len_sec = 3,
                                           wrap = True, 
                                           random_seed = 40)
# round the measured snr:
snr20 = int(round(snr20))
snr20

######################################################
pyso.plotsound(y_snr20, sr = sr, feature_type = 'signal', 
               title = 'Speech SNR {}'.format(snr20))
ipd.Audio(y_snr20,rate=sr)

######################################################
# Speech and Noise SNR 5
# ~~~~~~~~~~~~~~~~~~~~~~
y_snr05, snr05 = pyso.dsp.add_backgroundsound(y, white_noise, sr=sr, snr = 5,
                                           delay_mainsound_sec = 1,
                                           total_len_sec = 3,
                                           wrap = True, 
                                           random_seed = 40)
# round the measured snr:
snr05 = int(round(snr05))
snr05

######################################################
pyso.plotsound(y_snr05, sr = sr, feature_type = 'signal', 
               title = 'Speech SNR {}'.format(snr05))
ipd.Audio(y_snr05,rate=sr)


######################################################
# Plot Voice Activity
# ^^^^^^^^^^^^^^^^^^^
# NOTE: If no VAD, yellow dots are placed at the bottom. 
# If VAD , yellow dots are placed at the top.

######################################################
# If VAD window should be expanded
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set amount of time VAD should be extended in milliseconds.
# This may be useful if the one wants to capture more speech, 
# despite noise.
extend_window_ms = 400

######################################################
# Set background noise reference
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For measuring background noise in signal, set amount 
# of beginning noise in milliseconds to use.
use_beg_ms = 120


######################################################
# VAD (SNR 20)
# ~~~~~~~~~~~~
pyso.feats.plot_vad(y_snr20, sr=sr, use_beg_ms = use_beg_ms,
                    title = 'VAD SNR {}'.format(snr20))

######################################################
pyso.feats.plot_vad(y_snr20, sr=sr, use_beg_ms = use_beg_ms,
                    extend_window_ms=extend_window_ms,
                    title = 'VAD SNR {} (Padded by {} ms)'.format(snr20, extend_window_ms))

######################################################
# VAD (SNR 5)
# ~~~~~~~~~~~
pyso.feats.plot_vad(y_snr05, sr=sr, use_beg_ms = use_beg_ms, 
                    title = 'VAD SNR {}'.format(snr05))

######################################################
pyso.feats.plot_vad(y_snr05, sr=sr, use_beg_ms = use_beg_ms, 
                    extend_window_ms=extend_window_ms,
                    title = 'VAD SNR {} (Padded by {} ms)'.format(snr05, extend_window_ms))

######################################################
# Filter out Noise: Wiener Filter
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# See `pysoundtool.builtin.filtersignal`
filter_scale = 1 # how strong the filter should be
apply_postfilter = False # to reduce musical noise
     
######################################################
# VAD after WF (SNR 20)
# ~~~~~~~~~~~~~~~~~~~~~
y_snr20_wf, sr = pyso.filtersignal(y_snr20, filter_scale=filter_scale, 
                                   apply_postfilter=apply_postfilter)
ipd.Audio(y_snr20_wf,rate=sr)

######################################################
pyso.feats.plot_vad(y_snr20_wf, sr = sr,  use_beg_ms = use_beg_ms,
                    title = 'VAD SNR {} with Filter (scaled {})'.format(
                        snr20, filter_scale))
######################################################
pyso.feats.plot_vad(y_snr20_wf, sr = sr,  use_beg_ms = use_beg_ms,
                    extend_window_ms=extend_window_ms,
                    title = 'VAD SNR {} (Padded by {} ms)\nwith Filter (scaled {})'.format(
                        snr20, extend_window_ms, filter_scale))

######################################################
# VAD after WF (SNR 5)
# ~~~~~~~~~~~~~~~~~~~~
y_snr05_wf, sr = pyso.filtersignal(y_snr05, filter_scale=filter_scale, 
                                   apply_postfilter=apply_postfilter)
ipd.Audio(y_snr05_wf,rate=sr)

######################################################
pyso.feats.plot_vad(y_snr05_wf, sr = sr,  use_beg_ms = use_beg_ms,
                    title = 'VAD SNR {} with Filter (scaled {})'.format(
                        snr05, filter_scale))

######################################################
pyso.feats.plot_vad(y_snr05_wf, sr = sr,  use_beg_ms = use_beg_ms,
                    extend_window_ms=extend_window_ms,
                    title = 'VAD SNR {} (Padded by {} ms)\nwith Filter (scaled {})'.format(
                        snr05, extend_window_ms, filter_scale))

######################################################
# Remove Non-Speech
# ^^^^^^^^^^^^^^^^^

######################################################
# Speech Only (SNR 20)
# ~~~~~~~~~~~~~~~~~~~~
y_snr20_speech, sr = pyso.feats.get_vad_samples(y_snr20, sr=sr, 
                                          use_beg_ms = use_beg_ms)
y_snr20_speech_ew, sr = pyso.feats.get_vad_samples(y_snr20, sr=sr, 
                                          use_beg_ms = use_beg_ms,
                                          extend_window_ms = extend_window_ms)

######################################################
pyso.plotsound(y_snr20_speech, sr=sr, feature_type = 'signal', 
               title = 'Speech Only SNR {}'.format(snr20))
ipd.Audio(y_snr20_speech,rate=sr)

######################################################
pyso.plotsound(y_snr20_speech_ew, sr=sr, feature_type = 'signal',
                    title = 'Speech Only SNR {} (Padded by {} ms)'.format(snr20, extend_window_ms))
ipd.Audio(y_snr20_speech_ew,rate=sr)

######################################################
# Speech Only (SNR 5)
# ~~~~~~~~~~~~~~~~~~~
y_snr05_speech, sr = pyso.feats.get_vad_samples(y_snr05, sr=sr, 
                                         use_beg_ms = use_beg_ms)
y_snr05_speech_ew, sr = pyso.feats.get_vad_samples(y_snr05, sr=sr, 
                                         use_beg_ms = use_beg_ms,
                                         extend_window_ms = extend_window_ms)

######################################################
pyso.plotsound(y_snr05_speech, sr=sr, feature_type = 'signal', 
               title = 'Speech Only SNR {}'.format(snr05))
ipd.Audio(y_snr05_speech,rate=sr)

######################################################
pyso.plotsound(y_snr05_speech_ew, sr=sr, feature_type = 'signal',
                    title = 'Speech Only SNR {} (Padded by {} ms)'.format(snr05, extend_window_ms))
ipd.Audio(y_snr05_speech_ew,rate=sr)


######################################################
# Remove Non-Speech (with Filter)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


######################################################
# Speech Only (SNR 20 with Filter)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
y_snr20_speech_wf, sr = pyso.feats.get_vad_samples(y_snr20_wf, sr=sr, 
                                             use_beg_ms = use_beg_ms)

y_snr20_speech_wf_ew, sr = pyso.feats.get_vad_samples(y_snr20_wf, sr=sr, 
                                             use_beg_ms = use_beg_ms,
                                                extend_window_ms=extend_window_ms)

######################################################
pyso.plotsound(y_snr20_speech_wf, sr=sr, feature_type = 'signal', 
               title = 'Speech Only SNR {} with Filter'.format(snr20))
ipd.Audio(y_snr20_speech_wf,rate=sr)

######################################################
pyso.plotsound(y_snr20_speech_wf_ew, sr=sr, feature_type = 'signal', 
               title = 'Speech Only SNR {} (Padded by {} ms)\nwith Filter'.format(snr20, extend_window_ms))
ipd.Audio(y_snr20_speech_wf_ew,rate=sr)

######################################################
# Speech Only (SNR 5 with Filter)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
y_snr05_speech_wf, sr = pyso.feats.get_vad_samples(y_snr05_wf, sr=sr, 
                                            use_beg_ms = use_beg_ms)

y_snr05_speech_wf_ew, sr = pyso.feats.get_vad_samples(y_snr05_wf, sr=sr, 
                                            use_beg_ms = use_beg_ms,
                                            extend_window_ms=extend_window_ms)

######################################################
pyso.plotsound(y_snr05_speech_wf, sr=sr, feature_type = 'signal', 
               title = 'Speech Only SNR {} with Filter'.format(snr05))
ipd.Audio(y_snr05_speech_wf,rate=sr)

######################################################
pyso.plotsound(y_snr05_speech_wf_ew, sr=sr, feature_type = 'signal', 
               title = 'Speech Only SNR {} (Padded by {} ms)\nwith Filter'.format(snr05, extend_window_ms))
ipd.Audio(y_snr05_speech_wf_ew,rate=sr)
