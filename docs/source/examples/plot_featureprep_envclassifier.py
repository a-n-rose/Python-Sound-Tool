# coding: utf-8
"""
=====================================
Feature Extraction for Classification
=====================================

Extract acoustic features from labeled data for 
training an environment or speech classifier.

To see how PySoundTool implements this, see `pysoundtool.builtin.envclassifier_feats`.
"""


###############################################################################################
# 


#####################################################################
import pysoundtool as pyso

######################################################
# Prepare for Extraction: Data Organization
# -----------------------------------------

######################################################
# I will use a sample speech commands data set:

##########################################################
# Designate path relevant for accessing audiodata
data_dir = '../../../../mini-audio-datasets/speech_commands/'

######################################################
# Choose Feature Type 
# ~~~~~~~~~~~~~~~~~~~
# We can extract 'mfcc', 'fbank', 'powspec', and 'stft'.
# if you are working with speech, I suggest 'fbank', 'powspec', or 'stft'.

feature_type = 'stft'

######################################################
# Set Duration of Audio 
# ~~~~~~~~~~~~~~~~~~~~~
# How much audio in seconds used from each audio file.
# The example noise and speech files are only 1 second long
dur_sec = 1


#############################################################
# Option 1: Built-In Functionality - PySoundTool extracts the features for you
# ----------------------------------------------------------------------------

############################################################
# Define which data to use and which features to extract
# Everything else is based on defaults. A feature folder with
# the feature data will be created in the current working directory.
# (Although, you can set this under the parameter `data_features_dir`)
# `visualize` saves periodic images of the features extracted.
# This is useful if you want to know what's going on during the process.
extraction_dir = pyso.envclassifier_feats(data_dir, 
                                          feature_type=feature_type, 
                                          dur_sec=dur_sec,
                                          visualize=True);

################################################################
# The extracted features, extraction settings applied, and 
# which audio files were assigned to which datasets
# will be saved in the following directory:
extraction_dir

############################################################
# And that's it!


############################################################
# Option 2: A bit more hands-on (PySoundTool does a bit for you)
# --------------------------------------------------------------

######################################################
# Where to save extracted features:
data_features_dir = './audiodata/example_feats_models/classifier/'

######################################################
# create unique directory for feature extraction session (so important files won't get overwritten)
feat_extraction_dir = 'features_'+feature_type + '_' + pyso.utils.get_date()

######################################################
# Ensure data directories exist 
# and turn into pathlib.PosixPath object:
data_dir = pyso.utils.check_dir(data_dir, make=False)

######################################################
# collect labels associated with the audio data

# the labels are expected to be the titles of subfolders
labels = []
for label in data_dir.glob('*/'):
    if not label.suffix:
        # directory, not filename
        labels.append(label.stem)
labels = set(labels)
print(labels)

######################################################
# create directory for what we need to save:
data_features_dir = pyso.utils.check_dir(data_features_dir, make=True)
feat_extraction_dir = data_features_dir.joinpath(feat_extraction_dir)
feat_extraction_dir = pyso.utils.check_dir(feat_extraction_dir, make=True)

##########################################################
# dictionaries containing encoding and decoding labels:
dict_encode_path = feat_extraction_dir.joinpath('dict_encode.csv')
dict_decode_path = feat_extraction_dir.joinpath('dict_decode.csv')
# dictionary for which audio paths are assigned to which labels:
dict_encdodedlabel2audio_path = feat_extraction_dir.joinpath('dict_encdodedlabel2audio.csv')

##########################################################
# designate where to save train, val, and test data
data_train_path = feat_extraction_dir.joinpath('{}_data_{}.npy'.format('train',
                                                                    feature_type))
data_val_path = feat_extraction_dir.joinpath('{}_data_{}.npy'.format('val',
                                                                    feature_type))
data_test_path = feat_extraction_dir.joinpath('{}_data_{}.npy'.format('test',
                                                                    feature_type))

##########################################################
# create and save encoding/decoding labels dicts
dict_encode, dict_decode = pyso.datasets.create_dicts_labelsencoded(labels)
dict_encode_path = pyso.utils.save_dict(
    dict2save = dict_encode, 
    filename = dict_encode_path,
    overwrite=True)
dict_decode_path = pyso.utils.save_dict(
    dict2save = dict_decode, 
    filename = dict_decode_path,
    overwrite=True)


##########################################################
# Let's look at the encode and decod dicts:
print('Encode Dict:')
count = 0
for key, value in dict_encode.items():
    print(key, ' --> ', value)
    count +=1
    if count > 5:
        break
    
print('\nDecode Dict:')
count = 0
for key, value in dict_decode.items():
    print(key, ' --> ', value)
    count +=1
    if count > 5:
        break

##########################################################
# save audio paths to each label in dict 
paths_list = pyso.files.collect_audiofiles(data_dir, recursive=True)
paths_list = sorted(paths_list)

dict_encodedlabel2audio = pyso.datasets.create_encodedlabel2audio_dict(dict_encode,
                                                    paths_list)
dict_encdodedlabel2audio_path = pyso.utils.save_dict(
    dict2save = dict_encodedlabel2audio, 
    filename = dict_encdodedlabel2audio_path, 
    overwrite=True)

##########################################################
# See what this dictionary looks like:
count = 0
for key, value in dict_encodedlabel2audio.items():
    print(key, ' --> ', value)
    count +=1
    if count > 5:
        break


##########################################################
# assign audiofiles into train, validation, and test datasets
train, val, test = pyso.datasets.audio2datasets(dict_encdodedlabel2audio_path,
                                            perc_train=0.8,
                                            limit=None,
                                            seed=40)

##########################################################
# save audiofiles for each dataset to dict and save
dataset_dict = dict([('train',train),('val', val),('test',test)])
dataset_dict_path = feat_extraction_dir.joinpath('dataset_audiofiles.csv')
dataset_dict_path = pyso.utils.save_dict(
    dict2save = dataset_dict, 
    filename = dataset_dict_path, 
    overwrite=True)
    
##########################################################
# save paths to where extracted features of each dataset will be saved to dict w same keys
datasets_path2save_dict = dict([('train',data_train_path),
                                ('val', data_val_path),
                                ('test',data_test_path)])


######################################################################
# Visualize Audio Samples and Features
# ------------------------------------

######################################################
# for fun, let's visualize the audio:

######################################################################
# first audio sample in its raw signal
pyso.plotsound(paths_list[0], feature_type='signal', 
               title=paths_list[0].parent.stem+': signal')

######################################################################
# visualize the features that will be extracted

######################################################
# first audio sample
pyso.plotsound(paths_list[0], feature_type=feature_type, energy_scale='power_to_db',
               title = paths_list[0].parent.stem+': '+feature_type)

######################################################
# Extract and Save Features
# -------------------------
import time
start = time.time()

##########################################################
# extract features:
dataset_dict, datasets_path2save_dict = pyso.feats.save_features_datasets(
    datasets_dict = dataset_dict,
    datasets_path2save_dict = datasets_path2save_dict,
    labeled_data = True,
    feature_type = feature_type,
    dur_sec = dur_sec,
    win_size_ms = 20,
    visualize=True, # saves plots of features
    vis_every_n_frames=200); # limits how often plots are generated

end = time.time()

total_dur_sec = round(end-start,2)
total_dur, units = pyso.utils.adjust_time_units(total_dur_sec)
print('\nFinished! Total duration: {} {}.'.format(total_dur, units))
print('\nFeatures can be found here:')
print(feat_extraction_dir)

########################################################################
# Have a look in `feat_extraction_dir` and there will be your features.

###########################################################################
# Examining what info is logged
# -----------------------------

################################################################
# Dataset assigned to audio files and their labels (0, 1, or 2):
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
count = 0
for key, value in dataset_dict.items():
    print(key, ' --> ', value)
    count +=1
    if count > 5:
        break


###################################################################
# Dataset assigned to a pathway where .npy file will be saved:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
count = 0
for key, value in datasets_path2save_dict.items():
    print(key, ' --> ', value)
    count +=1
    if count > 5:
        break

######################################################
# Large Datasets
# --------------
# If you have very large amounts of audio you would like to process, you can 
# divide the datasets into smaller sections:

dataset_dict, datasets_path2save_dict = pyso.feats.save_features_datasets(
    datasets_dict = dataset_dict,
    datasets_path2save_dict = datasets_path2save_dict,
    labeled_data = True,
    feature_type = feature_type,
    dur_sec = dur_sec,
    win_size_ms = 20,
    subsection_data = True, # if you want to subsection at least largest dataset
    divide_factor = 3); # how many times you want the data to be sectioned.

###########################################################################
# Examining what info is logged: Subdividing datasets
# ---------------------------------------------------
# If you subdivide your data, this will be updated in the dataset_dict

count = 0
for key, value in dataset_dict.items():
    print(key, ' --> ', value)
    count +=1
    if count > 5:
        break

###################################################################
count = 0
for key, value in datasets_path2save_dict.items():
    print(key, ' --> ', value)
    count +=1
    if count > 5:
        break
