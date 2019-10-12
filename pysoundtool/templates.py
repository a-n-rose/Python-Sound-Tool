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

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)
 
def noisefilter(filter_project_name, 
                headpath, 
                target_wavfile, 
                noise_wavfile = None, 
                scale = 1, 
                apply_postfilter = False,
                max_vol = 0.4):
    '''Example code for implementing NoIze as just a noise filter.
    '''
    import pysoundtool as pyst
    if not noise_wavfile:
        #use background noise to filter out local noise
        output_filename = '{}/background_noise.wav'.format(headpath+filter_project_name)
        pyst.filtersignal(output_filename,
                           target_wavfile,
                           scale = scale,
                           apply_postfilter = apply_postfilter,
                           max_vol = max_vol)
        return None
    else:
        #use a separate noise file to reduce noise
        output_filename = '{}/separate_noise.wav'.format(headpath+filter_project_name)
        pyst.filtersignal(output_filename,
                           target_wavfile,
                           noise_file = noise_wavfile,
                           scale = scale,
                           apply_postfilter = apply_postfilter,
                           max_vol = max_vol)
        return None

def soundclassifier(classifier_project_name, 
                    headpath,
                    target_wavfile = None, 
                    audiodir = None,
                    feature_type = 'fbank',
                    audioclass_wavfile_limit = None):
    '''Example code for implementing NoIze as just a sound classifier.
    '''
    import pysoundtool as pyst
    #extract data
    my_project = pyst.PathSetup(classifier_project_name,
                            headpath,
                            audiodir,
                            feature_type = feature_type)
    
    no_conflicts_feats = my_project.cleanup_feats()
    if no_conflicts_feats == False:
        raise FileExistsError('Program cannot run.\
            \nMove the conflicting files or change project name.')
    feat_dir = my_project.feature_dirname
    if my_project.features is True and \
            feat_dir in str(my_project.features_dir) or \
                my_project.features is False:
        print('\nFeatures have been extracted.')
        print('\nLoading corresponding feature settings.')
        feats_class = pyst.getfeatsettings(my_project)
    elif audiodir:
        feats_class, my_project = pyst.run_featprep(
            my_project,
            feature_type=feature_type,
            limit=audioclass_wavfile_limit)
    import pysoundtool.models
    # Has a model been trained and saved?
    if my_project.model is not None \
            and feat_dir in str(my_project.model.parts[-3]):
        print('\nLoading previously trained classifier.')
        classifier_class = pyst.models.loadclassifier(my_project)
    else:
        print('\nNow training classifier with train, val, test datasets.')
        # check for file conflicts
        no_conflicts = my_project.cleanup_models()
        if no_conflicts == False:
            raise FileExistsError('Program cannot run.\
                \nMove the conflicting files or change project name.')
        classifier_class = pyst.models.buildclassifier(my_project)
    
    if target_wavfile:
        classify = pyst.models.ClassifySound(target_wavfile, 
                                              my_project, 
                                              feats_class, 
                                              classifier_class)
        label, label_encoded = classify.get_label()
        print('\nLabel classified: ', label)
