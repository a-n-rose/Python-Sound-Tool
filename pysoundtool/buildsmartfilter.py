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

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)

import pysoundtools as pyst
import pysoundtool.models as models



from pysoundtool.filterfun.filters import calc_audioclass_powerspecs,\
    coll_beg_audioclass_samps
from pysoundtool.filterfun.applyfilter import filtersignal


def mysmartfilter(name_dataset, headpath, audio_classes_dir,
                  feature_type='mfcc', num_filters=40,
                  sounddata=None,
                  scale=1, segment_length_ms=1000,
                  apply_postfilter=False,
                  augment_data=False,
                  limit=None,
                  use_rand_noisefile=False,
                  force_label=None,
                  classify_noise=True,
                  max_vol = 0.4):
    '''Applies feature prep, model training, and filtering to wavfile.
    '''
    if scale == 0:
        raise ValueError('scale cannot be set to 0')
        sys.exit()
    my_filter = pyst.PathSetup(name_dataset,
                          headpath,
                          audio_classes_dir,
                          feature_type=feature_type,
                          num_filters=num_filters,
                          segment_length_ms=segment_length_ms)

    # Have features been extracted?
    # check for file conflicts
    no_conflicts_feats = my_filter.cleanup_feats()
    if no_conflicts_feats == False:
        raise FileExistsError('Program cannot run.\
            \nMove the conflicting files or change project name.')
    feat_dir = my_filter.feature_dirname
    if my_filter.features is True and \
            feat_dir in str(my_filter.features_dir) or \
                my_filter.features is False:
        print('\nFeatures have been extracted.')
        print('\nLoading corresponding feature settings.')
        prep_feats = pyst.getfeatsettings(my_filter)
    elif audio_classes_dir:
        print('\nExtracting dataset features --> train.npy, val.npy, test.npy')
        prep_feats, my_filter = pyst.run_featprep(
            my_filter,
            feature_type=feature_type,
            num_filters=num_filters,
            segment_dur_ms=segment_length_ms,
            limit=limit, augment_data=augment_data)
    # Has the averaged power of the dataset been calculated?
    # check for file conflicts
    no_conflicts_ps = my_filter.cleanup_powspec()
    # conflicts found but were not resolved:
    if no_conflicts_ps is False:
        raise FileExistsError('Program cannot run.\
            \nMove the conflicting files or change project name.')
    # conflicts found but were resolved:
    elif no_conflicts_ps is True:
        my_filter.powspec = None
    if my_filter.powspec is None:
        if not use_rand_noisefile:
            print("\nConducting Welch's method on each class in dataset..")
            pyst.welch2class(my_filter, segment_length_ms,
                                    augment_data=augment_data)
        else:
            files_per_audioclass=1
            dur_ms = 1000
            print("\nSaving beg {}ms from {} files from each audioclass.".format(
                dur_ms,
                files_per_audioclass))
            pyst.save_class_noise(my_filter, 
                            prep_feats, 
                            num_each_audioclass=files_per_audioclass,
                            dur_ms=dur_ms)
    else:
        print("\nNoise class data extraction already performed on this dataset.")

    # Has a model been trained and saved?
    if my_filter.model is not None \
            and feat_dir in str(my_filter.model.parts[-3]):
        print('\nLoading previously trained scene classifier.')
        scene = models.loadclassifier(my_filter)
    else:
        print('\nNow training scene classifier with train, val, test datasets.')
        # check for file conflicts
        no_conflicts = my_filter.cleanup_models()
        if no_conflicts == False:
            raise FileExistsError('Program cannot run.\
                \nMove the conflicting files or change project name.')
        scene = models.buildclassifier(my_filter)

    if classify_noise:
        # Smart filtering begins:
        # Work with new audiofile: classify the background noise and
        # filter it out
        env = models.ClassifySound(sounddata, my_filter, prep_feats, scene)
        if force_label and isinstance(force_label,str):
            encoded_labels = scene.load_labels()
            for key, value in encoded_labels.items():
                if force_label.lower() == value.lower():
                    label = value
                    label_encoded = key
        else:
            label, label_encoded = env.get_label()
        print('label applied: ', label)
        # load average power spectrum of detected environment
        noise_powspec = env.load_assigned_avepower(label_encoded)
    else:
        noise_powspec = None
    # apply filter:
    if isinstance(sounddata, str):
        sounddata = pathlib.Path(sounddata)
    if isinstance(sounddata, pathlib.PosixPath):
        base_name = sounddata.parts[-1]
    else:
        base_name = 'output.wav'
    
    #adjust filename based on settings
    if apply_postfilter:
        postfilter = 'postfilter_'
    else:
        postfilter = '_'
    if classify_noise:
        label = label
    else:
        label = 'backgroundnoise'
    
    if force_label and classify_noise:
        forced = True
    else:
        forced = False
    
    outputname = 'filtered_scale*{}_{}_{}_forced{}_{}{}'.format(
        scale, name_dataset, label, forced, postfilter, base_name)

    if len(outputname) > 4 and outputname[-4:] == '.wav':
        outputname = outputname
    else:
        outputname = outputname+'.wav'

    filtersignal(
        output_filename = my_filter.features_dir.joinpath(outputname),
        wavfile = sounddata,
        noise_file = noise_powspec,
        scale = scale,
        apply_postfilter = apply_postfilter,
        max_vol = max_vol)
    
    return my_filter.features_dir.joinpath(outputname)
