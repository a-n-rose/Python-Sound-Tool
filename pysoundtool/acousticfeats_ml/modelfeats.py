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

###############################################################################
import pathlib
import numpy as np

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
packagedir = os.path.dirname(parentdir)
sys.path.insert(0, packagedir)

import pysoundtool as pyst


class PrepFeatures:

    def __init__(self,
                 feature_type='fbank',
                 sampling_rate=48000,
                 num_filters=40,
                 num_mfcc=None,
                 window_size=25,
                 window_shift=25/2.,
                 training_segment_ms=1000,
                 num_columns=None,
                 num_images_per_audiofile=None,
                 num_waves=None,
                 feature_sets=None,
                 window_type=None,
                 augment_data=False
                 ):
        '''
        Set defaults for feature extraction according to the methodology of 
        the paper by Sehgal and Kehtarnavaz (2018).


        Parameters
        ----------
        feature_type : str
            Which type of features extracted. Options: 'mfcc', 'fbank'. 
            (default 'fbank')
        sampling_rate : int, default=48000
            The audio files will be processed by this sampling rate.
        num_filters : int, default=40
            Number of mel filters applied when calculating the melspectrogram
        window_size : int or float, default=25
            In ms, the window size to calculate melspectrogram
        window_shift : int or float, default=12.5
            The amount of overlap of melspectrogram calculations. Default is
            50% overlap
        training_segment_ms : int or float, default=62.5
            In ms, the size of melspectrogram 'image' to feed to CNN.
        '''
        self.feature_type = feature_type
        self.sr = sampling_rate
        if 'fbank' in feature_type:
            self.num_filters = num_filters
            self.num_mfcc = None
        elif 'mfcc' in feature_type and num_mfcc is not None:
            self.num_filters = num_filters
            self.num_mfcc = num_mfcc
        elif 'mfcc' in feature_type and num_mfcc is None:
            self.num_filters = num_filters
            self.num_mfcc = num_filters
        elif 'stft' in feature_type:
            raise ValueError("Unfortunately, 'stft' is not yet supported.")
        else:
            raise ValueError("Sorry. Feature type not recognized.")
        self.window_size = window_size
        if window_shift:
            self.window_shift = window_shift
        else:
            self.window_shift = window_size/2.
        self.frame_length = pyst.dsp.calc_frame_length(self.window_size,
                                                  self.sr)
        self.fft_bins = self.frame_length
        self.training_segment_ms = training_segment_ms
        self.num_columns = num_columns or self.num_filters
        if 'mfcc' in feature_type:
            if self.num_mfcc is None:
                self.num_mfcc = self.num_filters
            self.num_columns = num_columns or self.num_mfcc
        self.num_images_per_audiofile = num_images_per_audiofile or 1
        self.num_waves = num_waves
        if 'fbank' in self.feature_type or 'mfcc' in self.feature_type:
            self.feature_sets = feature_sets or self.calc_filter_image_sets()
        self.window_type = window_type
        self.augment_data = augment_data

    def calc_filter_image_sets(self):
        '''
        calculates how many feature sets create a full image, given window size, 
        window shift, and desired image length in milliseconds.
        '''
        sets = pyst.dsp.calc_num_subframes(self.training_segment_ms,
                                      self.window_size,
                                      self.window_shift)
        return sets

    def get_max_samps(self, filter_features, num_sets):
        '''
        calculates the maximum  number of samples of a particular wave's 
        features that would also create a full image
        '''
        max_num_samps = (filter_features.shape[0]//num_sets) * num_sets
        return int(max_num_samps)

    def samps2feats(self, y, augment_data=None):
        '''Gets features from section of samples, at varying volumes.
        '''
        # when applying this to new data, dont't want to augment it.
        if not augment_data and self.augment_data:
            samples = pyst.augmentdata.spread_volumes(y)
        else:
            samples = (y,)
        feat_matrix = pyst.matrixfun.create_empty_matrix(
            (int(self.num_images_per_audiofile*len(samples)),
             self.feature_sets,
             self.num_columns))
        row = 0
        for sample_set in samples:
            feats, frame_length, window_size_ms = pyst.dsp.collect_features(
                feature_type=self.feature_type,
                samples=y,
                sr=self.sr,
                window_size_ms=self.window_size,
                window_shift_ms=self.window_shift,
                num_filters=self.num_filters,
                num_mfcc=self.num_mfcc,
                window_function=self.window_type)
            assert frame_length == self.fft_bins
            feat_matrix[row:row+len(feats)] = feats[:self.feature_sets]
            row += len(feats)
        return feat_matrix

    def extractfeats(self, sounddata, dur_sec=None, augment_data=None):
        '''Organizes feat extraction of each audiofile according to class attributes.
        '''
        if isinstance(sounddata, str):
            sounddata = pathlib.Path(sounddata)
        if isinstance(sounddata, pathlib.PosixPath) and \
            pyst.paths.is_audio_ext_allowed(sounddata):
            y, sr = pyst.dsp.load_signal(sounddata,
                                    sampling_rate=self.sr,
                                    dur_sec=dur_sec)
        else:
            print('The following datatype for sounddata is not understood:')
            print(type(sounddata))
            print('Data presented: ', sounddata)
            sys.exit()
        if augment_data is not None and augment_data != self.augment_data:
            feats = self.samps2feats(y, augment_data=augment_data)
        else:
            feats = self.samps2feats(y)
        assert feats.shape[1] == self.feature_sets
        assert feats.shape[2] == self.num_columns
        feats = feats[:self.feature_sets]
        return feats

    def get_feats(self, list_waves, dur_sec=None):
        '''collects fbank or mfcc features of entire wavfile list
        '''
        tot_waves = len(list_waves)
        num_images = 1
        if self.augment_data:
            num_versions_samples = 3  # three different volume augmentaions
        else:
            num_versions_samples = 1
        # for training scene classifier, just want 1 'image' per scen
        tot_len_matrix = int(tot_waves * num_images * num_versions_samples)
        # create empty matrix to fill features into
        # +1 is for the label column
        shape = (tot_len_matrix, self.feature_sets, self.num_columns+1)
        feats_matrix = pyst.matrixfun.create_empty_matrix(shape, complex_vals=False)
        row = 0  # keep track of where we are in filling the empty matrix
        for i, label_wave in enumerate(list_waves):
            # collect label information
            # list_waves contains tuples with (soundclass_int, wavfile)
            label = int(label_wave[0])
            wave = label_wave[1]
            if not os.path.exists(wave):
                print('Not Found: ', wave)
            else:
                feats = self.extractfeats(wave, dur_sec=dur_sec)
                # add label column - need label to stay with the features!
                label_col = np.full(
                    (feats.shape[0], self.feature_sets, 1), label)
                feats = np.concatenate((feats, label_col), axis=2)
                # fill the matrix with the features and labels
                feats_matrix[row:row+feats.shape[0]] = feats
                # actualize the row for the next set of features to fill it with
                row += feats.shape[0]
                # print on screen the progress
                progress = row / tot_len_matrix * 100
                sys.stdout.write("\r%d%% through current dataset" % progress)
                sys.stdout.flush()
                if not progress < 100:
                    print("\nCompleted feature extraction.")
        if row < tot_len_matrix:
            diff = tot_len_matrix - row
            print('A total of {} wavfiles were not found. \
                  \nExpected amount: {} total waves.'.format(
                      diff, tot_len_matrix))
            feats_matrix = feats_matrix[:row]
        # randomize rows
        np.random.shuffle(feats_matrix)
        return feats_matrix

    def get_save_feats(self, wave_list, directory4features, filename):
        if self.num_waves is None:
            self.num_waves = len(wave_list)
        else:
            self.num_waves += len(wave_list)
        feats = self.get_feats(
            wave_list, dur_sec=self.training_segment_ms/1000.0)
        save2file = directory4features.joinpath(filename)
        pyst.paths.save_feature_data(save2file, feats)
        return None

    def save_class_settings(self, path, replace=False):
        '''saves class settings to dictionary
        '''
        class_settings = self.__dict__
        filename = 'settings_{}.csv'.format(self.__class__.__name__)
        featuresettings_path = path.joinpath(filename)
        pyst.paths.save_dict(
            class_settings, featuresettings_path, replace=replace)
        return None


def prepfeatures(filter_class, feature_type='mfcc', num_filters=40,
                 segment_dur_ms=1000, limit=None, augment_data=False,
                 sampling_rate=48000):
    '''Pulls info from 'filter_class' instance to then extract, save features

    Parameters
    ----------
    filter_class : class
        The class instance holding attributes relating to path structure
        and filenames necessary for feature extraction
    feature_type : str, optional
        Acceptable inputs: 'mfcc' and 'fbank'. These are the features that
        will be extracted from the audio and saved (default 'mfcc')
    num_filters : int, optional
        The number of mel filters used during feature extraction. This number 
        ranges for 'mfcc' extraction between 13 and 40 and for 'fbank'
        extraction between 20 and 128. The higher the number, the greater the 
        computational load and memory requirement. (default 40)
    segment_dur_ms : int, optional
        The length in milliseconds of the acoustic data to extract features
        from. If 1000 ms, 1 second of acoustic data will be processed; 1 sec 
        of feature data will be extracted. If not enough audio data is present,
        the feature data will be zero padded. (default 1000)

    Returns
    ----------
    feats_class : class
        The class instance holding attributes relating to the current 
        feature extraction session
    filter_class : class
        The updated class instance holding attributes relating to path
        structure
    '''
    # extract features
    # create namedtuple with train, val, and test wavfiles and labels
    datasetwaves = pyst.featorg.audio2datasets(filter_class.audiodata_dir,
                                          filter_class.labels_encoded_path,
                                          filter_class.labels_waves_path,
                                          limit=limit)

    feats_class = PrepFeatures(feature_type=feature_type,
                               num_filters=num_filters,
                               training_segment_ms=segment_dur_ms,
                               augment_data=augment_data)
    # incase an error occurs; save this before extraction starts
    feats_class.save_class_settings(filter_class.features_dir)
    for i, dataset in enumerate(datasetwaves._fields):
        feats_class.get_save_feats(datasetwaves[i],
                                   filter_class.features_dir,
                                   '{}.npy'.format(dataset))
    # save again with added information, ie the total number of wavfiles
    feats_class.save_class_settings(filter_class.features_dir, replace=True)
    filter_class.features = filter_class.features_dir
    return feats_class, filter_class


def loadfeature_settings(feature_info):
    '''Loads prev extracted feature settings into new feature class instance

    This is useful if one wants to extract new features that match the
    dimensions and settings of previously extracted features.

    Parameters
    ----------
    feature_info : dict, class
        Either a dictionary or a class instance that holds the path
        attribute to a dictionary. 

    Returns
    -------
    feats_class : class 
        Feature extraction class instance with the same settings as the 
        settings dictionary
    '''
    if isinstance(feature_info, dict):
        feature_settings = feature_info
    else:
        featuresettings_path = pyst.paths.load_settings_file(
            feature_info.features_dir)
        feature_settings = pyst.paths.load_dict(featuresettings_path)
    sr = pyst.featorg.make_number(feature_settings['sr'])
    window_size = pyst.featorg.make_number(feature_settings['window_size'])
    window_shift = pyst.featorg.make_number(feature_settings['window_shift'])
    feature_sets = pyst.featorg.make_number(feature_settings['feature_sets'])
    feature_type = feature_settings['feature_type']
    num_columns = pyst.featorg.make_number(feature_settings['num_columns'])
    num_images_per_audiofile = pyst.featorg.make_number(
        feature_settings['num_images_per_audiofile'])
    training_segment_ms = pyst.featorg.make_number(
        feature_settings['training_segment_ms'])
    if 'fbank' in feature_type.lower():
        feature_type = 'fbank'
        num_filters = num_columns
        num_mfcc = None
    elif 'mfcc' in feature_type.lower():
        feature_type = 'mfcc'
        if num_columns != 40:
            num_filters = 40
        else:
            num_filters = num_columns
        num_mfcc = num_columns
    feats_class = PrepFeatures(feature_type=feature_type,
                               sampling_rate=sr,
                               num_filters=num_filters,
                               num_mfcc=num_mfcc,
                               window_size=window_size,
                               window_shift=window_shift,
                               training_segment_ms=training_segment_ms,
                               num_images_per_audiofile=num_images_per_audiofile)
    assert feature_sets == feats_class.feature_sets
    return feats_class
