# coding: utf-8
"""
========================================
Audio Dataset Exploration and Formatting
========================================

Examine audio files within a dataset, and reformat them if desired.  

To see how soundpy implements this, see `soundpy.builtin.dataset_logger` and 
`soundpy.builtin.dataset_formatter`.
"""

#####################################################################
# Let's import soundpy 
import soundpy as sp

###############################################################################################
#  
# Dataset Exploration
# ^^^^^^^^^^^^^^^^^^^

##########################################################
# Designate path relevant for accessing audiodata
sp_dir = '../../../'

##########################################################
# I will explore files in a small dataset on my computer with varying file formats.
dataset_path = '{}audiodata2/'.format(sp_dir)
dataset_info_dict = sp.builtin.dataset_logger('{}audiodata2/'.format(sp_dir));

#########################################################################
# This returns our data in a dictionary, perfect for exploring via Pandas
import pandas as pd
all_data = pd.DataFrame(dataset_info_dict).T
all_data.head()

###################################
# Let's have a look at the audio files and how uniform they are:
print('formats: ', all_data.format_type.unique())
print('bitdepth (types): ', all_data.bitdepth.unique())
print('mean duration (sec): ', all_data.dur_sec.mean())
print('std dev duration (sec): ', all_data.dur_sec.std())
print('min sample rate: ', all_data.sr.min())
print('max sample rate: ', all_data.sr.max())
print('number of channels: ', all_data.num_channels.unique())


##########################################################
# For a visual example, let's plot the count of various sample rates. (48000 Hz is high definition sound, 16000 Hz is wideband, and 8000 Hz is narrowband, similar to how speech sounds on the telephone.)
all_data.groupby('sr').count().plot(kind = 'bar', title = 'Sample Rate Counts')

###############################################################################################
# Reformat a Dataset
# ^^^^^^^^^^^^^^^^^^

##############################################################
# Let's say we have a dataset that we want to make consistent. 
# We can do that with soundpy
new_dataset_dir = sp.builtin.dataset_formatter(
    dataset_path, 
    recursive = True, # we want all the audio, even in nested directories
    format='WAV',
    bitdepth = 16, # if set to None, a default bitdepth will be applied
    sr = 16000, # wideband
    mono = True, # ensure data all have 1 channel
    dur_sec = 3, # audio will be limited to 3 seconds
    zeropad = True, # audio shorter than 3 seconds will be zeropadded
    new_dir = './example_dir/', # if None, a time-stamped directory will be created for you
    overwrite = False # can set to True if you want to overwrite files
    );
        
###############################################
# Let's see what the audio data looks like now:
dataset_formatted_dict = sp.builtin.dataset_logger(new_dataset_dir, recursive=True);
formatted_data = pd.DataFrame(dataset_formatted_dict).T

#####################
formatted_data.head()

###################################
print('audio formats: ', formatted_data.format_type.unique())
print('bitdepth (types): ', formatted_data.bitdepth.unique())
print('mean duration (sec): ', formatted_data.dur_sec.mean())
print('std dev duration (sec): ', formatted_data.dur_sec.std())
print('min sample rate: ', formatted_data.sr.min())
print('max sample rate: ', formatted_data.sr.max())
print('number of channels: ', formatted_data.num_channels.unique())

##########################################################
# Now all the audio data is sampled at the same rate: 8000 Hz
formatted_data.groupby('sr').count().plot(kind = 'bar', title = 'Sample Rate Counts')

###########################################
# There we go! 
# You can reformat only parts of the audio files, e.g. format or bitdepth.
# If you leave parameters in sp.builtin.dataset_formatter as None, the original
# settings of the audio file will be maintained (except for bitdepth. 
# A default bitdepth will be applied according to the format of the file); see `soundfile.default_subtype`.
