# coding: utf-8
"""
=======================================================
Feature Extraction for Denoising: Clean and Noisy Audio
=======================================================

This notebook offers an example for how you can use PySoundTool to extract
acoustic features from clean and noisy datasets, and save them to .npy files.
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
import pysoundtool as pyst


######################################################
# Prepare for Extraction: Data Organization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

######################################################
# PySoundTool offers example datasets. Let's use them.

# Example noisy data:
data_noisy_dir = '{}audiodata/minidatasets/denoise/noisy/'.format(package_dir)
# Example clean data:
data_clean_dir = '{}audiodata/minidatasets/denoise/clean/'.format(package_dir)
# Where to save extracted features:
data_features_dir = './audiodata/example_feats_models/denoiser/'

######################################################
# Which type of feature:

# We can also extract 'mfcc', 'powspec', and 'stft'
feature_type = 'fbank'

######################################################
# how much audio in seconds used from each audio file:
dur_sec = 3

######################################################
# How many sections should each sample be broken into? (optional)
frames_per_sample = 11


#############################################################
# Built-In Functionality: PySoundTool does everything for you
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

############################################################
# Define which data to use and which features to extract

# Everything else is based on defaults. A feature folder with
# the feature data will be created in the current working directory.

# (Although, you can set this under the parameter `data_features_dir`)
extraction_dir = pyst.denoiser_feats(data_clean_dir = data_clean_dir, 
                                     data_noisy_dir = data_noisy_dir,
                                     feature_type = feature_type, 
                                     dur_sec = dur_sec,
                                     frames_per_sample = frames_per_sample,
                                     visualize=True)

################################################################
# The extracted features, extraction settings applied, and 
# which audio files were assigned to which datasets
# will be saved in the following directory:
print(extraction_dir)

############################################################
# And that's it!


############################################################
# A bit more hands-on (PySoundTool does a bit for you)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#######################################################
## create unique directory for feature extraction session:
#feat_extraction_dir = 'features_'+feature_type + '_' + pyst.utils.get_date()

#######################################################
## Ensure clean and noisy data directories exist 
## and turn into pathlib.PosixPath objects:
#audio_clean_path = pyst.utils.check_dir(data_clean_dir, make=False)
#audio_noisy_path = pyst.utils.check_dir(data_noisy_dir, make=False)

#######################################################
## create directory for what we need to save:
#denoise_data_path = pyst.utils.check_dir(data_features_dir, make=True)
#feat_extraction_dir = denoise_data_path.joinpath(feat_extraction_dir)
#feat_extraction_dir = pyst.utils.check_dir(feat_extraction_dir, make=True)

###########################################################
## create paths to save noisy train, val, and test datasets
#data_train_noisy_path = feat_extraction_dir.joinpath('{}_data_{}_{}.npy'.format('train',
                                                                        #'noisy',
                                                                    #feature_type))
#data_val_noisy_path = feat_extraction_dir.joinpath('{}_data_{}_{}.npy'.format('val',
                                                                    #'noisy',
                                                                    #feature_type))
#data_test_noisy_path = feat_extraction_dir.joinpath('{}_data_{}_{}.npy'.format('test',
                                                                    #'noisy',
                                                                    #feature_type))

###########################################################
## create paths to save clean train, val, and test datasets
#data_train_clean_path = feat_extraction_dir.joinpath('{}_data_{}_{}.npy'.format('train',
                                                                        #'clean',
                                                                    #feature_type))
#data_val_clean_path = feat_extraction_dir.joinpath('{}_data_{}_{}.npy'.format('val',
                                                                    #'clean',
                                                                    #feature_type))
#data_test_clean_path = feat_extraction_dir.joinpath('{}_data_{}_{}.npy'.format('test',
                                                                    #'clean',
                                                                    #feature_type))

########################################################################
## collect audiofiles and divide them into train, val, and test datasets

###########################################################
## noisy data
#noisyaudio = pyst.utils.collect_audiofiles(audio_noisy_path, 
                                                #hidden_files = False,
                                                #wav_only = False,
                                                #recursive = False)
## sort audio (can compare if noisy and clean datasets are compatible)
#noisyaudio = sorted(noisyaudio)

###########################################################
## clean data
#cleanaudio = pyst.utils.collect_audiofiles(audio_clean_path, 
                                                #hidden_files = False,
                                                #wav_only = False,
                                                #recursive = False)
#cleanaudio = sorted(cleanaudio)

#############################################################################
## check if they match up: (expects clean file name to be in noisy file name)
#for i, audiofile in enumerate(noisyaudio):
    #if not pyst.utils.check_noisy_clean_match(audiofile, cleanaudio[i]):
        #raise ValueError('The noisy and clean audio datasets do not appear to match.')

#######################################################################
## save collected audiofiles for noisy and clean datasets to dictionary
#noisy_audio_dict = dict([('noisy', noisyaudio)])
#clean_audio_dict = dict([('clean', cleanaudio)])

#############################################################################
## separate into datasets, with random seed set so noisy and clean data match

###########################################################
## first noisy data
#train_noisy, val_noisy, test_noisy = pyst.data.audio2datasets(noisy_audio_dict,
                                                              #perc_train=0.8,
                                                              #seed=40)

###########################################################
## then clean data (order doesn't matter)
#train_clean, val_clean, test_clean = pyst.data.audio2datasets(clean_audio_dict,
                                                              #perc_train=0.8,
                                                              #seed=40)

###########################################################
## save train, val, test dataset assignments to dict
#dataset_dict_noisy = dict([('train', train_noisy),('val', val_noisy),('test', test_noisy)])
#dataset_dict_clean = dict([('train', train_clean),('val', val_clean),('test', test_clean)])

######################################################################
## keep track of paths to save data, once features have been extracted

###########################################################
## noisy data paths
#dataset_paths_noisy_dict = dict([('train',data_train_noisy_path),
                                #('val', data_val_noisy_path),
                                #('test',data_test_noisy_path)])

###########################################################
## clean data paths
#dataset_paths_clean_dict = dict([('train',data_train_clean_path),
                                #('val', data_val_clean_path),
                                #('test',data_test_clean_path)])

###########################################################
## ensure the noisy and clean audio match up:
#for key, value in dataset_dict_noisy.items():
    #for j, audiofile in enumerate(value):
        #if not pyst.utils.check_noisy_clean_match(audiofile,
                                                #dataset_dict_clean[key][j]):
            #raise ValueError('There is a mismatch between noisy and clean audio. '+\
                #'\nThe noisy file:\n{}'.format(dataset_dict_noisy[key][i])+\
                    #'\ndoes not seem to match the clean file:\n{}'.format(audiofile))



#######################################################################
## Visualize Audio Samples and Features
## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#######################################################
## for fun, let's visualize the audio:

#######################################################################
## first noisy audio sample in its raw signal
#pyst.plotsound(noisyaudio[0], feature_type='signal')

#######################################################################
## first clean audio sample in its raw signal
#pyst.plotsound(cleanaudio[0], feature_type='signal')

#######################################################################
## visualize the features that will be extracted

#######################################################
## first noisy audio sample
#pyst.plotsound(noisyaudio[0], feature_type='fbank', power_scale='power_to_db')

#######################################################################
## first clean audio sample
#pyst.plotsound(cleanaudio[0], feature_type='fbank', power_scale='power_to_db')

#######################################################
## Extract and Save Features
## ^^^^^^^^^^^^^^^^^^^^^^^^^
#import time
#start = time.time()

###########################################################
## extract clean data first (again, order doesn't matter)
#dataset_dict_clean, dataset_paths_clean_dict = pyst.feats.save_features_datasets(
    #datasets_dict = dataset_dict_clean,
    #datasets_path2save_dict = dataset_paths_clean_dict,
    #feature_type = feature_type + ' clean',
    #dur_sec = dur_sec,
    #frames_per_sample = frames_per_sample,
    #win_size_ms = 16,
    #visualize=True, # saves plots of features
    #vis_every_n_frames=200) # limits how often plots are generated
    
###########################################################
## then noisy data
#dataset_dict_noisy, dataset_paths_noisy_dict = pyst.feats.save_features_datasets(
    #datasets_dict = dataset_dict_noisy,
    #datasets_path2save_dict = dataset_paths_noisy_dict,
    #feature_type = feature_type + ' noisy',
    #dur_sec = dur_sec,
    #frames_per_sample = frames_per_sample,
    #win_size_ms = 16,
    #visualize=True, # saves plots of features 
    #vis_every_n_frames=200) # limits how often plots are generated
#end = time.time()

#total_dur_sec = round(end-start,2)
#total_dur, units = pyst.utils.adjust_time_units(total_dur_sec)
#print('\nFinished! Total duration: {} {}.'.format(total_dur, units))
#print('\nFeatures can be found here:')
#print(feat_extraction_dir)

#########################################################################
## Have a look in `feat_extraction_dir` and there will be your features.

#########################################################################
## Logging Dataset Audio Assignments
## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#######################################################
## In order to keep track of which audio files were 
## assigned to each dataset file, you can save this information
## so:
#filename = feat_extraction_dir.joinpath('Noisy_Dataset_Assignments.csv')
#noisy_datasets_dict_paths = pyst.utils.save_dict(dataset_dict_noisy, 
                                                 #filename = filename)

#filename = feat_extraction_dir.joinpath('Clean_Dataset_Assignments.csv')
#clean_datasets_dict_paths = pyst.utils.save_dict(dataset_dict_clean, 
                                                 #filename = filename)

##########################################################
## You will find these files in the `feat_extraction_dir`.


#######################################################
## Large Datasets
## ^^^^^^^^^^^^^^
## If you have very large amounts of audio you would like to process, you can 
## divide the datasets into smaller sections:

#dataset_dict_clean, dataset_paths_clean_dict = pyst.feats.save_features_datasets(
    #datasets_dict = dataset_dict_clean,
    #datasets_path2save_dict = dataset_paths_clean_dict,
    #feature_type = feature_type + ' clean',
    #dur_sec = dur_sec,
    #frames_per_sample = frames_per_sample,
    #win_size_ms = 16,
    #subsection_data = True, # if you want to subsection at least largest dataset
    #divide_factor = 10) # how many times you want the data to be sectioned.


