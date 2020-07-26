
# coding: utf-8
"""
========================
Voice Activity Detection
========================

Get the audio frames that have voice activity.

To see how PySoundTool implements this, see `pysoundtool.dsp.vad`.

Note: these have not yet been implemented in the pypi package release.
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
pyso_dir = './'

######################################################
# Load sample speech audio
# ^^^^^^^^^^^^^^^^^^^^^^^^
# Note: these files are available in the PySoundTool repo.
speech = '{}audiodata/python.wav'.format(pyso_dir)
y, sr = pyso.loadsound(speech, sr=48000)
ipd.Audio(y,rate=sr)

######################################################
# Generate white noise as background noise
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Note: these files are available in the PySoundTool repo.
white_noise = pyso.dsp.generate_noise(len(y), random_seed = 40)

######################################################
# Generate speech audio at various SNR levels
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

######################################################
# Speech and Noise SNR 40
# ~~~~~~~~~~~~~~~~~~~~~~~
y_snr40, snr40 = pyso.dsp.add_backgroundsound(y, white_noise, sr=sr, snr = 40,
                                           delay_mainsound_sec = 1,
                                           total_len_sec = 3,
                                           wrap = True, 
                                           random_seed = 40)
# round the measured snr:
snr40 = int(round(snr40))
ipd.Audio(y_snr40,rate=sr)
pyso.plotsound(y_snr40, sr = sr, feature_type = 'signal', 
               title = 'Speech {} SNR'.format(snr40))

######################################################
# Speech and Noise SNR 30
# ~~~~~~~~~~~~~~~~~~~~~~~
y_snr30, snr30 = pyso.dsp.add_backgroundsound(y, white_noise, sr=sr, snr = 30,
                                           delay_mainsound_sec = 1,
                                           total_len_sec = 3,
                                           wrap = True, 
                                           random_seed = 40)
# round the measured snr:
snr30 = int(round(snr30))
ipd.Audio(y_snr30,rate=sr)
pyso.plotsound(y_snr30, sr = sr, feature_type = 'signal', 
               title = 'Speech {} SNR'.format(snr30))


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
ipd.Audio(y_snr20,rate=sr)
pyso.plotsound(y_snr20, sr = sr, feature_type = 'signal', 
               title = 'Speech {} SNR'.format(snr20))


######################################################
# Speech and Noise SNR 10
# ~~~~~~~~~~~~~~~~~~~~~~~
y_snr10, snr10 = pyso.dsp.add_backgroundsound(y, white_noise, sr=sr, snr = 10,
                                           delay_mainsound_sec = 1,
                                           total_len_sec = 3,
                                           wrap = True, 
                                           random_seed = 40)
# round the measured snr:
snr10 = int(round(snr10))
ipd.Audio(y_snr10,rate=sr)
pyso.plotsound(y_snr10, sr = sr, feature_type = 'signal', 
               title = 'Speech {} SNR'.format(snr10))

######################################################
# Speech and Noise SNR 0
# ~~~~~~~~~~~~~~~~~~~~~~
y_snr0, snr0 = pyso.dsp.add_backgroundsound(y, white_noise, sr=sr, snr = 0,
                                           delay_mainsound_sec = 1,
                                           total_len_sec = 3,
                                           wrap = True, 
                                           random_seed = 40)
# round the measured snr:
snr0 = int(round(snr0))
ipd.Audio(y_snr0,rate=sr)
pyso.plotsound(y_snr0, sr = sr, feature_type = 'signal', 
               title = 'Speech {} SNR'.format(snr0))

######################################################
# Speech and Noise SNR -10
# ~~~~~~~~~~~~~~~~~~~~~~~~
y_snr_minus10, snr_minus10 = pyso.dsp.add_backgroundsound(y, white_noise, sr=sr, snr = -10,
                                           delay_mainsound_sec = 1,
                                           total_len_sec = 3,
                                           wrap = True, 
                                           random_seed = 40)
# round the measured snr:
snr_minus10 = int(round(snr_minus10))
ipd.Audio(y_snr_minus10,rate=sr)
pyso.plotsound(y_snr_minus10, sr = sr, feature_type = 'signal', 
               title = 'Speech {} SNR'.format(snr_minus10))


######################################################
# Plot Voice Activity
# ^^^^^^^^^^^^^^^^^^^

######################################################
# Set amount of beginning noise as reference
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
use_beg_ms = 120

######################################################
# VAD (SNR 40)
# ~~~~~~~~~~~~
# For each audio frame, if no VAD, yellow dots are placed at 0. 
# If VAD , yellow dots are placed at 1.

pyso.feats.plot_vad(y_snr40, sr=sr, use_beg_ms = use_beg_ms, 
                    title = 'VAD {} SNR'.format(snr40))

######################################################
# VAD (SNR 30)
# ~~~~~~~~~~~~
pyso.feats.plot_vad(y_snr30, sr=sr, use_beg_ms = use_beg_ms,
                    title = 'VAD {} SNR'.format(snr30))

######################################################
# VAD (SNR 20)
# ~~~~~~~~~~~~
pyso.feats.plot_vad(y_snr20, sr=sr, use_beg_ms = use_beg_ms,
                    title = 'VAD {} SNR'.format(snr20))

######################################################
# VAD (SNR 10)
# ~~~~~~~~~~~~
pyso.feats.plot_vad(y_snr10, sr=sr, use_beg_ms = use_beg_ms,
                    title = 'VAD {} SNR'.format(snr10))

######################################################
# VAD (SNR 0)
# ~~~~~~~~~~~
pyso.feats.plot_vad(y_snr0, sr=sr, use_beg_ms = use_beg_ms, 
                    title = 'VAD {} SNR'.format(snr0))

######################################################
# VAD (SNR -10)
# ~~~~~~~~~~~~~
pyso.feats.plot_vad(y_snr_minus10, sr=sr, use_beg_ms = use_beg_ms, 
                    title = 'VAD {} SNR'.format(snr_minus10))


######################################################
# Filter out Noise: Wiener Filter
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
filter_scale = 1
apply_postfilter = False


######################################################
# VAD after WF (SNR 40)
# ~~~~~~~~~~~~~~~~~~~~~
y_snr40_wf, sr = pyso.filtersignal(y_snr40, filter_scale=filter_scale, 
                                   apply_postfilter=apply_postfilter)
ipd.Audio(y_snr40_wf,rate=sr)
pyso.feats.plot_vad(y_snr40_wf, sr = sr,  use_beg_ms = use_beg_ms,
                    title = 'VAD {} SNR with Filter (scaled {})'.format(
                        snr40, filter_scale))
                    
######################################################
# VAD after WF (SNR 30)
# ~~~~~~~~~~~~~~~~~~~~~
y_snr30_wf, sr = pyso.filtersignal(y_snr30, filter_scale=filter_scale, 
                                   apply_postfilter=apply_postfilter)
ipd.Audio(y_snr30_wf,rate=sr)
pyso.feats.plot_vad(y_snr30_wf, sr = sr,  use_beg_ms = use_beg_ms,
                    title = 'VAD {} SNR with Filter (scaled {})'.format(
                        snr30, filter_scale))
                    
                    
######################################################
# VAD after WF (SNR 20)
# ~~~~~~~~~~~~~~~~~~~~~
y_snr20_wf, sr = pyso.filtersignal(y_snr20, filter_scale=filter_scale, 
                                   apply_postfilter=apply_postfilter)
ipd.Audio(y_snr20_wf,rate=sr)
pyso.feats.plot_vad(y_snr20_wf, sr = sr,  use_beg_ms = use_beg_ms,
                    title = 'VAD {} SNR with Filter (scaled {})'.format(
                        snr20, filter_scale))
                    
                    
######################################################
# VAD after WF (SNR 10)
# ~~~~~~~~~~~~~~~~~~~~~
y_snr10_wf, sr = pyso.filtersignal(y_snr10, filter_scale=filter_scale, 
                                   apply_postfilter=apply_postfilter)
ipd.Audio(y_snr10_wf,rate=sr)
pyso.feats.plot_vad(y_snr10_wf, sr = sr,  use_beg_ms = use_beg_ms,
                    title = 'VAD {} SNR with Filter (scaled {})'.format(
                        snr10, filter_scale))
                    
######################################################
# VAD after WF (SNR 0)
# ~~~~~~~~~~~~~~~~~~~~
y_snr0_wf, sr = pyso.filtersignal(y_snr0, filter_scale=filter_scale, 
                                   apply_postfilter=apply_postfilter)
ipd.Audio(y_snr0_wf,rate=sr)
pyso.feats.plot_vad(y_snr0_wf, sr = sr,  use_beg_ms = use_beg_ms,
                    title = 'VAD {} SNR with Filter (scaled {})'.format(
                        snr0, filter_scale))
                    
######################################################
# VAD after WF (SNR -10)
# ~~~~~~~~~~~~~~~~~~~~~~
y_snr_minus10_wf, sr = pyso.filtersignal(y_snr_minus10, filter_scale=filter_scale, 
                                   apply_postfilter=apply_postfilter)
ipd.Audio(y_snr_minus10_wf,rate=sr)
pyso.feats.plot_vad(y_snr_minus10_wf, sr = sr,  use_beg_ms = use_beg_ms,
                    title = 'VAD {} SNR with Filter (scaled {})'.format(
                        snr_minus10, filter_scale))
                    
        
######################################################
# Remove Non-Speech
# ^^^^^^^^^^^^^^^^^
percent_vad = 0.75


######################################################
# Speech Only (SNR 40)
# ~~~~~~~~~~~~~~~~~~~~

stft_snr40_speech, sr = pyso.dsp.get_vad_stft(y_snr40, sr = sr, 
                                          use_beg_ms = use_beg_ms,
                                          percent_vad = percent_vad)
y_snr40_speech, sr = pyso.dsp.get_vad_samples(y_snr40, sr = sr, 
                                          use_beg_ms = use_beg_ms,
                                          percent_vad = percent_vad)
ipd.Audio(y_snr40_speech,rate=sr) 
pyso.plotsound(stft_snr40_speech, sr=sr, feature_type = 'stft', 
               title = 'Speech Only SNR {}'.format(snr40))

######################################################
# Speech Only (SNR 30)
# ~~~~~~~~~~~~~~~~~~~~
y_snr30_speech, sr = pyso.dsp.get_vad_samples(y_snr30, sr=sr,
                                          use_beg_ms = use_beg_ms,
                                          percent_vad = percent_vad)
ipd.Audio(y_snr30_speech,rate=sr)
pyso.plotsound(y_snr30_speech, sr=sr, feature_type = 'signal',
               title = 'Speech Only SNR {}'.format(snr30))

######################################################
# Speech Only (SNR 20)
# ~~~~~~~~~~~~~~~~~~~~
stft_snr20_speech, sr = pyso.dsp.get_vad_stft(y_snr20, sr=sr, 
                                          use_beg_ms = use_beg_ms,
                                          percent_vad = percent_vad)
y_snr20_speech, sr = pyso.dsp.get_vad_samples(y_snr20, sr=sr, 
                                          use_beg_ms = use_beg_ms,
                                          percent_vad = percent_vad)
ipd.Audio(y_snr20_speech,rate=sr)
pyso.plotsound(stft_snr20_speech, sr=sr, feature_type = 'stft',
               title = 'Speech Only SNR {}'.format(snr20))

######################################################
# Speech Only (SNR 10)
# ~~~~~~~~~~~~~~~~~~~~
y_snr10_speech, sr = pyso.dsp.get_vad_samples(y_snr10, sr=sr, 
                                          use_beg_ms = use_beg_ms,
                                          percent_vad = percent_vad)
ipd.Audio(y_snr10_speech,rate=sr)
pyso.plotsound(y_snr10_speech, sr=sr, feature_type = 'signal', 
               title = 'Speech Only SNR {}'.format(snr10))

######################################################
# Speech Only (SNR 0)
# ~~~~~~~~~~~~~~~~~~~
stft_snr0_speech, sr = pyso.dsp.get_vad_stft(y_snr0, sr=sr, 
                                         use_beg_ms = use_beg_ms,
                                         percent_vad = percent_vad)
y_snr0_speech, sr = pyso.dsp.get_vad_samples(y_snr0, sr=sr, 
                                         use_beg_ms = use_beg_ms,
                                         percent_vad = percent_vad)
ipd.Audio(y_snr0_speech,rate=sr)
pyso.plotsound(stft_snr0_speech, sr=sr, feature_type = 'stft', 
               title = 'Speech Only SNR {}'.format(snr0))

######################################################
# Speech Only (SNR -10)
# ~~~~~~~~~~~~~~~~~~~~~
y_snr_minuse10_speech, sr = pyso.dsp.get_vad_samples(y_snr_minus10, sr=sr,
                                                 use_beg_ms = use_beg_ms,
                                                 percent_vad = percent_vad)
ipd.Audio(y_snr_minuse10_speech,rate=sr)
pyso.plotsound(y_snr_minuse10_speech, sr=sr, feature_type = 'signal', 
               title = 'Speech Only SNR {}'.format(snr_minus10))


######################################################
# Remove Non-Speech (with Filter)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
use_beg_ms = 250
######################################################
# Speech Only (SNR 40 with Filter)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The sr has not been adjusted for the wiener filtered data.
y_snr40_speech_wf, sr = pyso.dsp.get_vad_samples(y_snr40_wf, sr=sr, 
                                             use_beg_ms = use_beg_ms,
                                             percent_vad = percent_vad)
ipd.Audio(y_snr40_speech_wf,rate=sr) 
pyso.plotsound(y_snr40_speech_wf, sr=sr, feature_type = 'signal', 
               title = 'Speech Only SNR {} with Filter'.format(snr40))

######################################################
# Speech Only (SNR 30 with Filter)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
y_snr30_speech_wf, sr = pyso.dsp.get_vad_samples(y_snr30_wf, sr=sr, 
                                             use_beg_ms = use_beg_ms,
                                             percent_vad = percent_vad)
ipd.Audio(y_snr30_speech_wf,rate=sr)
pyso.plotsound(y_snr30_speech_wf, sr=sr, feature_type = 'signal', 
               title = 'Speech Only SNR {} with Filter'.format(snr30))

######################################################
# Speech Only (SNR 20 with Filter)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
y_snr20_speech_wf, sr = pyso.dsp.get_vad_samples(y_snr20_wf, sr=sr, 
                                             use_beg_ms = use_beg_ms,
                                             percent_vad = percent_vad)
ipd.Audio(y_snr20_speech_wf,rate=sr)
pyso.plotsound(y_snr20_speech_wf, sr=sr, feature_type = 'signal', 
               title = 'Speech Only SNR {} with Filter'.format(snr20))

######################################################
# Speech Only (SNR 10 with Filter)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
y_snr10_speech_wf, sr = pyso.dsp.get_vad_samples(y_snr10_wf, sr=sr, 
                                             use_beg_ms = use_beg_ms,
                                             percent_vad = percent_vad)
ipd.Audio(y_snr10_speech_wf,rate=sr)
pyso.plotsound(y_snr10_speech_wf, sr=sr, feature_type = 'signal', 
               title = 'Speech Only SNR {} with Filter'.format(snr10))

######################################################
# Speech Only (SNR 0 with Filter)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
y_snr0_speech_wf, sr = pyso.dsp.get_vad_samples(y_snr0_wf, sr=sr, 
                                            use_beg_ms = use_beg_ms,
                                            percent_vad = percent_vad)
ipd.Audio(y_snr0_speech_wf,rate=sr)
pyso.plotsound(y_snr0_speech_wf, sr=sr, feature_type = 'signal', 
               title = 'Speech Only SNR {} with Filter'.format(snr0))

######################################################
# Speech Only (SNR -10 with Filter)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
y_snr_minuse10_speech_wf, sr = pyso.dsp.get_vad_samples(y_snr_minus10_wf, sr=sr, 
                                                    use_beg_ms = use_beg_ms,
                                                    percent_vad = percent_vad)
if y_snr_minuse10_speech_wf.any():
    ipd.Audio(y_snr_minuse10_speech_wf,rate=sr)
pyso.plotsound(y_snr_minuse10_speech_wf, sr=sr, feature_type = 'signal', 
               title = 'Speech Only SNR {} with Filter'.format(snr_minus10))
