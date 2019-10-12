#!/bin/bash
# Copyright 2019 Peggy Sylopp und Aislyn Rose GbR
# All rights reserved
# This file is part of the  NoIze-framework
# The NoIze-framework is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by the  
# Free Software Foundation, either version 3 of the License, or (at your option) 
# any later version.
#
#@author Aislyn Rose
#@version 0.1
#@date 31.08.2019
#
# The  NoIze-framework  is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details. 
#
# You should have received a copy of the GNU AFFERO General Public License 
# along with the NoIze-framework. If not, see http://www.gnu.org/licenses/.

'''
The module paths.py contains functionality that manages how files are 
stored. 
'''
###############################################################################
import pathlib
import csv
# for saving numpy files
import numpy as np
# for saving wavfiles
from scipy.io import wavfile

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
packagedir = os.path.dirname(parentdir)
sys.path.insert(0, packagedir)

import pysoundtool as pyst


class PathSetup:
    '''Manages paths for files specific to this smart filter instance

    Based on the headpath and feature settings, directories and files are 
    created. Data pertaining to feature extraction and model training are
    stored and accessed via paths built by this class instance.

    Attributes
    ----------

    smartfilt_headpath : pathlib.PosixPath
        The path to the project's directory where all feature, model and 
        sound files will be saved. The name of this directory is created by
        the `project_name` parameter when initializing this class.

        Hint: as both the features and models rely heavily on 
        the data used, include a reference to that data here. Only features and
        models trained with the same dataset should be allowed to be here.

    audiodata_dir : pathlib.PosixPath
        The path to the directory where audio training data can be found.
        One should ensure folders exist here, titled according to the sound
        data stored inside of them. 

        For example, to train a model on classifying sounds as either a
        dishwasher, air conditioner, or running toilet, you should have three
        folders, titled 'dishwasher', 'air_conditioner' and 'running_toilet',
        respectively. 

    labels_encoded_path : None, pathlib.PosixPath
        Once created by the program, the path to the .csv file containing 
        the labels found in the `àudiodata_dir` and to which integer the labels
        were encoded. 

        These pairings, label names (e.g. 'air_conditioner', 'dishwasher', 
        'toilet') and the integers they are encoded with (0,1,2), is important 
        for training the neural network - it won't understand letters - and for 
        knowing which label the network categorizes new acoustic data.

    labels_waves_path : None, pathlib.PosixPath
        Once created by the program, the path to the .csv file that stores
        the audio file paths belonging to each audio class. None otherwise.


    _labels_wavfile_filename : str 
        The name this program expects to find when looking for the .csv 
        containing audio class labels and the audiofile paths belonging to
        that class.

    _encoded_labels_filename : str 
        The name this program expects to find when looking for the .csv 
        containing the audio class labels their encoded pairings.

    features : None, True 
        None if features have not yet been successfully extracted and True if 
        features have been fully extracted from entire dataset and saved.

        These are relevant for the training of the CNN model for scene 
        classification.

    powspec : None, True
        True if audio class average audio spectrum data are collected. None
        otherwise.

        These are values relevant for the noise filtering of future sound data.

    model : None, pathlib.PosixPath
        Once a model has been traind on these features and saved, the path
        and filename of that model. None otherwise.

    feature_dirname : str 
        The generated directory name to store all data from this instance of
        feature extraction. 

        This directory is named according to the
        type of features extracted, the number of filters applied during
        extraction, as well as the number of seconds of audio data from each
        audio file used to extract features. 

        For example, if 'mfcc' features with 40 filters are extracted, and a
        1.5 second segment of audio data from each audio file is used for that
        extraction, the directory name is: 'mfcc_40_1.5'.

    features_dir : pathlib.PosixPath
        The path to the directory titled `feature_dirname`, generated by 
        the program.

    _powspec_settings : str 
        The name this program expects to find when looking for the .csv 
        containing the settings used when calculating the average power
        spectrum of each audio class. This is relevant for applying the filter:
        the same settings are ideally used when calculating the power spectrum
        of the signal that needs filtering.

    powspec_path : pathlib.PosixPath
        The path to where the audio class average power spectrum files will
        be or are located for the entire dataset. These values are calculated
        independent from the features extracted for machine learning. 

    modelname : str 
        The name generated and applied to models trained on these features.

    model_dir : pathlib.PosixPath
        The path to the directory where `model` and related data will be or
        are currently stored.

    model_settings_path : None, pathlib.PosixPath
        If a model has been trained and saved, the path to the .csv file
        holding the settings for that specific model.
    '''

    def __init__(self,
                 project_name='test_data',
                 smartfilt_headpath='/home/airos/Desktop/testing_ground/default_model/',
                 audiodata_dir=None,
                 feature_type='mfcc',
                 num_filters=40,
                 segment_length_ms=1000,
                 ):
        self.smartfilt_headpath = prep_path([smartfilt_headpath, project_name])
        self.audiodata_dir = prep_path(audiodata_dir, create_new=False)
        self._labels_wavfile_filename = 'label_wavfiles.csv'
        self._encoded_labels_filename = 'encoded_labels.csv'
        self._features_settings_filename = 'settings_PrepFeatures.csv'
        # initialize variables
        self.feature_settings_path = None
        self.labels_encoded_path = None
        self.labels_waves_path = None
        self.features = None
        self.powspec = None
        self.model = None
        self.feature_dirname = self.prep_feat_dirname(feature_type,
                                                      num_filters,
                                                      segment_length_ms)
        self.features_dir = self.get_features_path()
        self._powspec_settings = 'powspec_settings.csv'
        self.powspec_path = self.get_avepowspec_path()
        self.modelname = project_name+'_model'
        self.model_dir = self.get_modelpath()
        self._model_settings_filename = 'settings_SceneClassifier.csv'
        self.model_settings_path = self.get_modelsettings_path()

    def prep_feat_dirname(self, feature_type, num_filters, segment_length_ms):
        len_sec = str(round(segment_length_ms/1000.0, 1))
        if 'stft' in feature_type:
            feat_dirname = feature_type+'_'+len_sec
        else:
            feat_dirname = feature_type+'_'+str(num_filters)+'_'+len_sec
        return feat_dirname

    def get_features_path(self):
        features_dir = self.smartfilt_headpath.joinpath('features',
                                                        self.feature_dirname)
        features_parent_dir = features_dir.parent
        npy_files = list(features_parent_dir.glob('**/*.npy'))
        if npy_files:
            powspec_files = []
            training_files = []
            for item in npy_files:
                if 'powspec_average' in str(item.parent):
                    powspec_files.append(item)
                elif features_dir.parts[-1] in item.parts[-2]:
                    training_files.append(item)
            if powspec_files:
                self.powspec = True
                self.powspec_path = powspec_files[0].parent
            if training_files:
                self.features = True  
                label_waves_filename = check4files(features_dir,
                                                   self._labels_wavfile_filename)
                if label_waves_filename:
                    self.labels_waves_path = label_waves_filename
                encoded_labels_filename = check4files(features_dir,
                                                      self._encoded_labels_filename)
                if encoded_labels_filename:
                    self.labels_encoded_path = encoded_labels_filename
                return features_dir
            else:
                encoded_labels_filename = check4files(features_dir,
                                                      self._encoded_labels_filename)
                feature_settings_filename = check4files(features_dir,
                                                        self._features_settings_filename)
                if encoded_labels_filename and feature_settings_filename:
                    self.labels_encoded_path = encoded_labels_filename
                    self.feature_settings_path = feature_settings_filename
                    self.features = False # not necessary - already used for training
                else:
                    self.features = None # not yet extracted
        features_dir = prep_path(features_dir, create_new=True)
        self.labels_waves_path = features_dir.joinpath(
            self._labels_wavfile_filename)
        self.labels_encoded_path = features_dir.joinpath(
            self._encoded_labels_filename)
        self.feature_settings_path = features_dir.joinpath(
            self._features_settings_filename)
        return features_dir

    def get_modelpath(self):
        '''expects model related information to be in same directory as the model
        '''
        model_dir = self.smartfilt_headpath.joinpath('models',
                                                     self.feature_dirname,
                                                     self.modelname)
        #relevant_features = self.feature_dirname
        models_list = list(model_dir.glob('**/*.h5'))
        if models_list:
            # just one model to choose from
            if len(models_list) == 1:
                modelpath = models_list[0]
            # want the best model available
            else:
                modelpath = [j for j in models_list if 'best' in str(j)]
                if modelpath:
                    modelpath = modelpath[0]
                    print('multiple models found. chose this model:')
                    print(modelpath)
            self.model = modelpath
            return model_dir
        # create a new model directory
        model_dir = prep_path(model_dir, create_new=True)
        return model_dir

    def get_modelsettings_path(self):
        '''sets the path to the model settings file

        If the model already exists, uses that model's parent directory. Otherwise
        sets the path to where a new model will be trained and saved.
        '''
        if self.model:
            model_settings_path = load_settings_file(self.model.parent,
                                                     keyword=["settings"])
        else:
            model_settings_path = self.model_dir.joinpath(
                self._model_settings_filename)
        return model_settings_path

    def get_avepowspec_path(self):
        if self.powspec is None:
            path = self.features_dir.parent.joinpath(
                'powspec_average')
            path = prep_path(path, create_new=True)
        else:
            path = self.powspec_path
        return path

    def cleanup_feats(self):
        '''Checks for feature extraction settings and training data files.
    
        If setting files (i.e. csv files) exist without training data files 
        (i.e. npy files), and a directory for training data has been provided,
        delete csv files.
        '''
        feat_csvs = list(self.features_dir.glob('**/*.csv'))
        npy_files = list(self.features_dir.glob('**/*.npy'))
        if feat_csvs and not npy_files:
            if not self.audiodata_dir:
                return True # No feature extraction necessary.
            else:
                print('\nPress Y to remove the following conflicting files:')
                print('\n'.join(map(str, feat_csvs)))
                remove = input()
                if 'y' in remove.lower():
                    for f in feat_csvs:
                        os.remove(f)
                    print('Conflicting files have been removed.')
                    return True
                else:
                    print('Files not removed.')
                    print('You may experience file conflicts if you rerun program')
                    return False
        return None

    def cleanup_powspec(self):
        '''Checks for power spectrum settings and filter data files.
    
        If setting files (i.e. csv files) exist without or with too few 
        data files (should be one data file for each audio class in training
        data), the setting files and data files will be deleted.
        '''
        powspec_csv = list(self.powspec_path.glob('**/*.csv'))
        if powspec_csv:
            for filename in powspec_csv:
                if 'setting' in str(filename):
                    powspec_settings_dict = load_dict(filename)
                    num_audio_classes = int(
                        powspec_settings_dict['num_audio_classes'])
        else:
            num_audio_classes = 0
        powspec_npy = list(self.powspec_path.glob('**/*.npy'))
        if powspec_csv and not powspec_npy or \
                num_audio_classes > 0 and num_audio_classes > len(powspec_npy):
            print('\nAverage Power Extraction Error: Some files are missing.')
            print('Before continuing, delete the following files:')
            print('\nPress Y to delete/remove:')
            print('\n'.join(map(str, powspec_csv)))
            print('\n'.join(map(str, powspec_npy)))
            remove = input()
            if 'y' in remove.lower():
                for f in powspec_csv:
                    os.remove(f)
                for g in powspec_npy:
                    os.remove(g)
                print('Conflicting files have been deleted.')
                return True
            else:
                print('Files have not been deleted. You may have filename conflicts.')
                return False
        return None

    def cleanup_models(self):
        '''Checks for model creation settings and model files.
    
        If setting files (i.e. csv files) exist without model file(s) 
        (i.e. h5 files), delete csv files.
        '''
        models_csvs = list(self.model_dir.glob('**/*.csv'))
        models_list = list(self.model_dir.glob('**/*.h5'))
        if models_csvs and not models_list:
            print('\nFile Conflict Error:\
                \nUnpaired csv files found and need to be removed before building model.')
            print('\nPress Y to remove the following files:')
            print('\n'.join(map(str, models_csvs)))
            remove = input()
            if 'y' in remove.lower():
                for f in models_csvs:
                    os.remove(f)
                print('Files have been removed. There should be no conflicts.')
                return True
            else:
                print('Files have not been removed')
                print('Please change the name of the model to avoid file conflicts')
                return False
        return None


def check4files(path, filename):
    '''checks for a filename in the subdirectores of a pathlib object
    '''
    file_list = list(path.glob(filename))
    if file_list:
        return file_list[0]


def prep_path(path, create_new=True):
    if isinstance(path, list):
        for i, path in enumerate(path):
            if i == 0:
                headpath = pathlib.Path(path)
            else:
                headpath = headpath.joinpath(path)
        path = headpath
    if path is not None:
        path = pathlib.Path(path)
        if not os.path.exists(path):
            if create_new:
                os.makedirs(path)
            else:
                print('The following path does not exist:\n{}'.format(path))
                sys.exit()
        return path
    return None


def load_dict(csv_path):
    '''Loads a dictionary from csv file. Expands csv limit if too large.
    '''
    try:
        with open(csv_path, mode='r') as infile:
            reader = csv.reader(infile)
            dict_prepped = {rows[0]: rows[1] for rows in reader}
    except csv.Error:
        print('Dictionary values or size is too large.')
        print('Maxing out field size limit for loading this dictionary:')
        print(csv_path)
        print('\nThe new field size limit is:')
        maxInt = sys.maxsize
        print(maxInt)
        csv.field_size_limit(maxInt)
        dict_prepped = load_dict(csv_path)
    except OverflowError as e:
        print(e)
        maxInt = int(maxInt/10)
        print('Reducing field size limit to: ', maxInt)
        dict_prepped = load_dict(csv_path)
    return dict_prepped


def save_dict(dict2save, filename, replace=False):
    '''Saves dictionary as csv file to indicated path and filename

    Parameters
    ----------
    dict2save : dict
        The dictionary that is to be saved 
    filename : str 
        The path and name to save the dictionary under. If '.csv' 
        extension is not given, it is added.
    replace : bool, optional
        Whether or not the saved dictionary should overwrite a 
        preexisting file (default False)

    Returns
    ----------
    path : pathlib.PosixPath
        The path where the dictionary was saved
    '''
    if not isinstance(filename, pathlib.PosixPath):
        filename = pathlib.Path(filename)
    if filename.parts[-1][-4:] != '.csv':
        filename_str = filename.resolve()
        filename_csv = filename_str+'.csv'
        filename = pathlib.Path(filename_csv)
    if not replace:
        if os.path.exists(filename):
            raise FileExistsError(
                'The file {} already exists at this path:\
                \n{}'.format(filename.parts[-1], filename))
    with open(filename, 'w') as f:
        w = csv.writer(f)
        w.writerows(dict2save.items())
    return filename

def load_settings_file(directory, keyword=['settings', 'PrepFeatures']):
    files = list(directory.glob('**/*.csv'))
    if files:
        files_r = [f for f in files if all(ele in str(f) for ele in keyword)]
        if len(files_r) > 1:
            print('There are multiple "{}" .csv files found in this directory:'.format(
                ' and '.join(keyword)))
            print(directory)
            print('There should only be one to designate feature data settings')
            sys.exit()
        elif files_r:
            data_settings_path = files_r[0]
            return data_settings_path
    print('No relevant files found at the following location:')
    print(directory)
    sys.exit()

def check_extension(filename, extension, replace=False):
    '''Adds expected extension if it not included in the filename

    If extension is an empty string, it assumes the filename should be
    a directory.

    Parameters
    ----------
    filename : str, pathlib.PosixPath
        The path and filename of the file to be checked
    extension : str
        The expected extension the filename to have. 
    replace : bool
        If True and the old and new extensions don't match, the new one
        will replace the old extension. If false, the new extension will 
        follow the old one.

    Returns
    -------
    filename : str, pathlib.PosixPath
        The corrected filename with correct extension. Returned as the same
        type as provided.

    Examples
    ----------
    >>> npy = check_extension('data','npy')
    >>> npy2 = check_extension('data','.npy')
    >>> npy3 = check_extension('data.npy','npy')
    >>> npy
    'data.npy'
    >>> assert npy == npy2 == npy3
    >>> txt_posixpath = check_extension(
    ...                    pathlib.Path('data'),
    ...                    'txt')
    >>> txt_str = check_extension('data','.txt')
    >>> assert isinstance(txt_posixpath,
    ...                    pathlib.PosixPath)
    >>> assert isinstance(txt_str, str)
    >>> txt_posixpath 
    PosixPath('data.txt')
    >>> txt_str
    'data.txt'
    >>> check_extension('data.txt', 'npy', replace = True)
    'data.npy'
    '''
    filename_orig = filename
    if extension and isinstance(extension, str):
        if extension[0] != '.':
            extension = '.'+extension
    elif extension and not isinstance(extension, str):
        raise TypeError('Expected extension to be of type string. \
            \nReceived type {}.'.format(type(extension)))
    else:
        print('No extension provided. No change made to the following filename:\
            \n{}'.format(filename))
        return filename
    if isinstance(filename, str):
        filename = pathlib.Path(filename)
    if isinstance(filename, pathlib.PosixPath):
        if not filename.suffix:
            filename_str = str(filename)
            filename = pathlib.Path(filename_str+extension)
        elif filename.suffix != extension:
            if replace:
                ext_prev = filename.suffix
                filename_str = str(filename)
                filename_replace_ext = filename_str.replace(
                    ext_prev, extension)
                filename = pathlib.Path(filename_replace_ext)
            else:
                filename_str = str(filename)
                filename = pathlib.Path(filename_str+extension)
    else:
        raise TypeError('Unexpected type for filename: {} \
            \nExpected string or pathlib.PosixPath object'.format(type(filename)))
    if isinstance(filename_orig, str):
        filename = str(filename)
    assert type(filename) == type(filename_orig)
    return filename

def is_audio_ext_allowed(audiofile):
    '''Checks that the audiofile extension is allowed

    Parameters
    ----------
    audiofile : pathlib.PosixPath, str

    Returns
    -------
    Return value : bool 
        True if the extension is allowed, False otherwise.
    '''
    allowed_ext = ['.wav']
    try:
        if audiofile.suffix and audiofile.suffix in allowed_ext:
            return True
        if not audiofile.suffix:
            if audiofile.parts and audiofile.parts[0] in allowed_ext:
                return True
    except AttributeError:
        if isinstance(audiofile, str):
            if audiofile in allowed_ext or '.'+audiofile in allowed_ext:
                return True
        if isinstance(audiofile, str):
            audiofile = pathlib.Path(audiofile)
        return is_audio_ext_allowed(audiofile)
    return False

def collect_audio_and_labels(data_path):
    '''Collects class label names and the wavfiles within each class

    Acceptable extensions: '.wav'

    Expects wavfiles to be in subdirectory: 'data'
    labels are expected to be the names of each subdirectory in 'data'
    does not include waves with filenames starting with '_'
    '''
    p = pathlib.Path(data_path)
    if not os.path.exists(p):
        raise pyst.errors.pathinvalid_error(p)
    all_files = p.glob('**/*')
    audiofiles = [f for f in all_files if is_audio_ext_allowed(f)]
    if not audiofiles:
        raise pyst.errors.noaudiofiles_error(p)
    # remove directories with "_" at the beginning
    paths = [p for p in audiofiles if p.parent.parts[-1][0] != "_"]
    labels = [j.parts[-2] for j in paths]
    return paths, labels

def string2list(list_paths_string):
    '''Take a string of wavfiles list and establishes back to list

    This handles lists of strings, lists of pathlib.PosixPath objects, and lists 
    of pathlib.PurePosixPath objects that were converted into a type string object.

    Parameters
    ----------
    list_paths_string : str 
        The list that was converted into a string object 

    Returns
    -------
    list_paths : list 
        The list converted back to a list of paths as pathlib.PosixPath objects.

    Examples
    --------
    >>> input_string = "[PosixPath('data/audio/vacuum/vacuum1.wav')]"
    >>> type(input_string)
    <class 'str'>
    >>> typelist = string2list(input_string)
    >>> typelist
    [PosixPath('data/audio/vacuum/vacuum1.wav')]
    >>> type(typelist)
    <class 'list'>
    '''
    # remove the string brackets and separate by space and comma --> list
    list_string_red = list_paths_string[1:-1].split(', ')
    if 'PurePosixPath' in list_paths_string:
        remove_str = "PurePosixPath('"
        end_index = -2
    elif 'PosixPath' in list_paths_string:
        remove_str = "PosixPath('"
        end_index = -2
    else:
        remove_str = "('"
        end_index = -2
    # remove unwanted sections of the string items
    list_paths = []
    for path in list_string_red:
        list_paths.append(pathlib.Path(
            path.replace(remove_str, '')[:end_index]))
    return list_paths

def save_feature_data(filename, matrix_data):
    '''Function to manage the saving of numpy arrays/matrices to numpy files

    Parameters
    ----------
    filename : str, pathlib.PosixPath
        The path and filename the matrix data will be saved under. 
    matrix_data : ndarray
        The data in a numpy ndarray that is to be saved
    '''
    filename = check_extension(filename, '.npy')
    np.save(filename, matrix_data)

def load_feature_data(filename):
    '''Uses path to data files to load the features

    Parameters
    ----------
    filename : str, pathlib.PosixPath
        the path and filename to the data to be loaded. The file must be
        a numpy file; if the extension '.npy' is not included, it will be added.
    '''
    filename = check_extension(filename, '.npy')
    data = np.load(filename)
    return data

def save_wave(wavfile_name, signal_values, sampling_rate, overwrite=False):
    """saves the wave at designated path

    Parameters
    ----------
    wavfile_name : str
        path and name the wave is to be saved under
    signal_values : ndarray
        values of real signal to be saved

    Returns
    ----------
    True if successful, otherwise False
    """
    if isinstance(wavfile_name, str):
        wavfile_name = pathlib.Path(wavfile_name)
    directory = wavfile_name.parent
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not overwrite:
        wavfile_name = if_exist_tweek_filename(wavfile_name)
    try:
        wavfile.write(wavfile_name, sampling_rate, signal_values)
        return True, wavfile_name
    except Exception as e:
        print(e)
    return False, wavfile_name

if __name__ == "__main__":
    import doctest
    doctest.testmod()
