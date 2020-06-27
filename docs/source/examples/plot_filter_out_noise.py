
# coding: utf-8
"""
===========================
Filter Out Background Noise
===========================

Use PySoundTool to filter out background noise from audio signals.
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
# Define the noisy and clean speech audio files.
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

######################################################
# PySoundTool offers example audio files. Let's use them.

##########################################################
# Speech sample:
speech_noisy = '{}audiodata/python_traffic.wav'.format(package_dir)
speech_noisy = pyst.utils.string2pathlib(speech_noisy)
print(speech_noisy)
speech_clean = '{}audiodata/python.wav'.format(package_dir)
speech_clean = pyst.utils.string2pathlib(speech_clean)
print(speech_clean)

##########################################################
# Hear and see the noisy speech 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# For filtering, we will set the sample rate to be quite high:
sr = 48000
noisy, sr = pyst.loadsound(speech_noisy, sr=sr)
ipd.Audio(noisy,rate=sr)

##########################################################
pyst.plotsound(noisy, sr=sr, feature_type='signal', title='Noisy Speech')


##########################################################
# The same for the clean speech:
clean, sr = pyst.loadsound(speech_clean, sr=sr)
ipd.Audio(clean,rate=sr)

##########################################################
pyst.plotsound(clean, sr=sr, feature_type='signal', title='Clean Speech')


##########################################################
# Filter the noisy speech
# ^^^^^^^^^^^^^^^^^^^^^^^

##########################################################
# Let's filter with a Wiener filter:
noisy_wf, sr = pyst.filtersignal(noisy,
                                 sr=sr,
                                 filter_type='wiener') # default

##########################################################
# Wiener Filter 
# ~~~~~~~~~~~~~

##########################################################
ipd.Audio(noisy_wf,rate=sr)

##########################################################
pyst.plotsound(noisy_wf, sr=sr, feature_type='signal', 
               title='Noisy Speech: Wiener Filter')

##########################################################
# Let's filter with a Wiener filter and postfilter
noisy_wfpf, sr = pyst.filtersignal(noisy,
                                 sr=sr,
                                 filter_type='wiener',
                                 apply_postfilter = True) 

#################################################################
# Wiener Filter with Postfilter
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##########################################################
ipd.Audio(noisy_wfpf,rate=sr)

##########################################################
pyst.plotsound(noisy_wfpf, sr=sr, feature_type='signal', 
               title='Noisy Speech: Wiener Filter with Postfilter')


##########################################################
# Let's filter using band spectral subtraction
noisy_bs, sr = pyst.filtersignal(noisy,
                                 sr=sr,
                                 filter_type='bandspec') 

#################################################################
# Band Spectral Subtraction
# ~~~~~~~~~~~~~~~~~~~~~~~~~

##########################################################
ipd.Audio(noisy_bs,rate=sr)

##########################################################
pyst.plotsound(noisy_bs, sr=sr, feature_type='signal', 
               title='Noisy Speech: Band Spectral Subtraction')



#########################################################################
# Finally, let's filter using band spectral subtraction with a postfilter
noisy_bspf, sr = pyst.filtersignal(noisy,
                                 sr=sr,
                                 filter_type='bandspec', 
                                 apply_postfilter = True) 

#################################################################
# Band Spectral Subtraction with Postfilter
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##########################################################
ipd.Audio(noisy_bspf,rate=sr)

##########################################################
pyst.plotsound(noisy_bspf, sr=sr, feature_type='signal', 
               title='Noisy Speech: Band Spectral Subtraction with Postfilter')


##########################################################
# Filter: increase the scale
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

##########################################################
# Let's filter with a Wiener filter:
filter_scale = 5
noisy_wf, sr = pyst.filtersignal(noisy,
                                 sr=sr,
                                 filter_type='wiener',
                                 filter_scale = filter_scale)

##########################################################
# Wiener Filter
# ~~~~~~~~~~~~~

##########################################################
ipd.Audio(noisy_wf,rate=sr)

##########################################################
pyst.plotsound(noisy_wf, sr=sr, feature_type='signal', 
               title='Noisy Speech: Wiener Filter Scale {}'.format(filter_scale))

##########################################################
# Let's filter with a Wiener filter and postfilter
noisy_wfpf, sr = pyst.filtersignal(noisy,
                                 sr=sr,
                                 filter_type='wiener',
                                 apply_postfilter = True,
                                 filter_scale = filter_scale) 

#################################################################
# Wiener Filter with Postfilter
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##########################################################
ipd.Audio(noisy_wfpf,rate=sr)

##########################################################
pyst.plotsound(noisy_wfpf, sr=sr, feature_type='signal', 
               title='Noisy Speech: Wiener Filter with Postfilter Scale {}'.format(filter_scale))


##########################################################
# Let's filter using band spectral subtraction at higher scale
noisy_bs, sr = pyst.filtersignal(noisy,
                                 sr=sr,
                                 filter_type = 'bandspec',
                                 filter_scale = filter_scale) 

#################################################################
# Band Spectral Subtraction
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Note: I haven't noticed much of a change with band spectral subtraction. 

##########################################################
ipd.Audio(noisy_bs,rate=sr)

##########################################################
pyst.plotsound(noisy_bs, sr=sr, feature_type='signal', 
               title='Noisy Speech: Band Spectral Subtraction Scale {}'.format(filter_scale))



#########################################################################
# Let's filter using band spectral subtraction with a postfilter
noisy_bspf, sr = pyst.filtersignal(noisy,
                                 sr=sr,
                                 filter_type = 'bandspec', 
                                 apply_postfilter = True,
                                 filter_scale = filter_scale) 

#################################################################
# Band Spectral Subtraction with Postfilter
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##########################################################
ipd.Audio(noisy_bspf,rate=sr)

##########################################################
pyst.plotsound(noisy_bspf, sr=sr, feature_type='signal', 
               title='Noisy Speech: Band Spectral Subtraction with Postfilter Scale {}'.format(
                   filter_scale))

##########################################################
# Filter: alter number of bands
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#########################################################################
# Let's alter the number of bands for band spectral subtraction
num_bands = 10
noisy_bs, sr = pyst.filtersignal(noisy,
                                 sr=sr,
                                 filter_type = 'bandspec', 
                                 num_bands = num_bands) # default 6 

#################################################################
# Band Spectral Subtraction
# ~~~~~~~~~~~~~~~~~~~~~~~~~

##########################################################
ipd.Audio(noisy_bs,rate=sr)

##########################################################
pyst.plotsound(noisy_bs, sr=sr, feature_type='signal', 
               title='Noisy Speech: Band Spectral Subtraction: {} Bands'.format(
                   num_bands))
               
               

#########################################################################
# Let's alter the number of bands for band spectral subtraction
noisy_bspf, sr = pyst.filtersignal(noisy,
                                 sr=sr,
                                 filter_type = 'bandspec', 
                                 apply_postfilter = True,
                                 num_bands = num_bands) # default 6 

#################################################################
# Band Spectral Subtraction with Postfilter 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##########################################################
ipd.Audio(noisy_bspf,rate=sr)

##########################################################
pyst.plotsound(noisy_bspf, sr=sr, feature_type='signal', 
               title='Noisy Speech: Band Spectral Subtraction with Postfilter: {} Bands'.format(
                   num_bands))
