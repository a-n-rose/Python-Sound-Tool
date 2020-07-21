# coding: utf-8
"""
========================================
Audio Dataset Exploration and Formatting
========================================

Use PySoundTool to examine audio files within a dataset, and to reformat them if desired.  

To see how PySoundTool implements this, see `pysoundtool.builtin.dataset_logger` and 
`pysoundtool.builtin.dataset_formatter`.
"""


###############################################################################################
#  
# Dataset Exploration
# ^^^^^^^^^^^^^^^^^^^


##########################################################
# Ignore this snippet of code: it is only for this example
import matplotlib.pyplot as plt
import os
package_dir = '../../../'
os.chdir(package_dir)


#####################################################################
# Let's import pysoundtool, assuming it is in your working directory:
import pysoundtool as pyso;


##########################################################
# I will explore files in a small dataset on my computer
dataset_path = '{}test_audio/'.format(package_dir)
dataset_info_dict = pyso.builtin.dataset_logger('{}test_audio/'.format(package_dir));

#########################################################################
# This returns our data in a dictionary, perfect for exploring via Pandas
import pandas as pd
all_data = pd.DataFrame(dataset_info_dict).T
all_data.head()

###################################
# Let's have a look at our dataset:
print('formats: ', all_data.format_type.unique())
print('bitdepth (types): ', all_data.bitdepth.unique())
print('mean duration (sec): ', all_data.dur_sec.mean())
print('std dev duration (sec): ', all_data.dur_sec.std())
print('min sample rate: ', all_data.sr.min())
print('max sample rate: ', all_data.sr.max())
print('number of channels: ', all_data.num_channels.unique())

##########################################################
all_data.groupby('sr').count().plot(kind = 'bar', title = 'Sample Rate Counts')

###############################################################################################
# Reformat a Dataset
# ^^^^^^^^^^^^^^^^^^

##############################################################
# Let's say we have a dataset that we want to make consistent. 
# We can do that with PySoundTool
new_dataset_dir = pyso.builtin.dataset_formatter(dataset_path, 
                                              recursive = True, # we want all the audio, even in nested directories
                                              format='WAV',
                                              bitdepth = 16, # if set to None, a default bitdepth will be applied
                                              sr = 8000, # narrowband
                                              mono = True, # ensure data all have 1 channel
                                              dur_sec = 3, # audio will be limited to 3 seconds
                                              zeropad = True, # audio shorter than 3 seconds will be zeropadded
                                              new_dir = './example_dir/', # if None, a time-stamped directory will be created for you
                                              overwrite = False # can set to True if you want to overwrite files
                                             );
        
###############################################
# Let's see what the audio data looks like now:
dataset_formatted_dict = pyso.builtin.dataset_logger(new_dataset_dir, recursive=True);
formatted_data = pd.DataFrame(dataset_formatted_dict).T

#####################
formatted_data.head()

###################################
# And how about all the audio data?
print('formats: ', formatted_data.format_type.unique())
print('bitdepth (types): ', formatted_data.bitdepth.unique())
print('mean duration (sec): ', formatted_data.dur_sec.mean())
print('std dev duration (sec): ', formatted_data.dur_sec.std())
print('min sample rate: ', formatted_data.sr.min())
print('max sample rate: ', formatted_data.sr.max())
print('number of channels: ', formatted_data.num_channels.unique())

##########################################################
formatted_data.groupby('sr').count().plot(kind = 'bar', title = 'Sample Rate Counts')

###########################################
# There we go! 
# You can reformat only parts of the audio files, e.g. format or bitdepth.
# If you leave parameters in pyso.builtin.dataset_formatter as None, the original
# settings of the audio file will be maintained (except for bitdepth. A default
# bitdepth will be applied according to the format of the file).
