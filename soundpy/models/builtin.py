'''The soundpy.models.builtin module includes example functions that train neural
networks on sound data.
''' 
import time
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import librosa
import collections

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)
import soundpy as sp
from soundpy import models as spdl


def denoiser_train(feature_extraction_dir,
                   model_name = 'model_autoencoder_denoise',
                   feature_type = None,
                   use_generator = True,
                   normalize = True,
                   patience = 10, 
                   **kwargs):
    '''Collects training features and train autoencoder denoiser.
    
    Parameters
    ----------
    feature_extraction_dir : str or pathlib.PosixPath
        Directory where extracted feature files are located (format .npy).
        
    model_name : str
        The name for the model. This can be quite generic as the date up to 
        the millisecond will be added to ensure a unique name for each trained model.
        (default 'model_autoencoder_denoise')
        
    feature_type : str, optional
        The type of features that will be used to train the model. This is 
        only for the purposes of naming the model. If set to None, it will 
        not be included in the model name.
        
    use_generator : bool 
        If True, a generator will be used to feed training data to the model. Otherwise
        the entire training data will be used to train the model all at once.
        (default True)
        
    normalize : bool 
        If True, the data will be normalized before feeding to the model.
        (default False)
        
    patience : int 
        Number of epochs to train without improvement before early stopping.
        
    **kwargs : additional keyword arguments
        The keyword arguments for keras.fit(). Note, 
        the keyword arguments differ for validation data so be sure to use the 
        correct keyword arguments, depending on if you use the generator or not.
        TODO: add link to keras.fit().
        
    Returns
    -------
    model_dir : pathlib.PosixPath
        The directory where the model and associated files can be found.
        
    See Also
    --------
    soundpy.datasets.separate_train_val_test_files
        Generates paths lists for train, validation, and test files. Useful
        for noisy vs clean datasets and also for multiple training files.
    
    soundpy.models.generator
        The generator function that feeds data to the model.
        
    soundpy.models.modelsetup.setup_callbacks
        The function that sets up callbacks (e.g. logging, save best model, early
        stopping, etc.)
        
    soundpy.models.template_models.autoencoder_denoise
        Template model architecture for basic autoencoder denoiser.
    '''
    if use_generator is False:
        import warnings
        msg = '\nWARNING: It is advised to set `use_generator` to True '+\
            'as memory issues are avoided and training is more reliable. '+\
                'There may be bugs in the functionality set to False. '
    dataset_path = sp.utils.check_dir(feature_extraction_dir, make=False)
    
    # designate where to save model and related files
    if feature_type:
        model_name += '_'+feature_type + '_' + sp.utils.get_date() 
    else:
        model_name += '_' + sp.utils.get_date() 
    model_dir = dataset_path.joinpath(model_name)
    model_dir = sp.utils.check_dir(model_dir, make=True)
    model_name += '.h5'
    model_path = model_dir.joinpath(model_name)
    
    # prepare features files to load for training
    features_files = list(dataset_path.glob('*.npy'))
    # NamedTuple: 'datasets.train.noisy', 'datasets.train.clean', etc.
    datasets = sp.datasets.separate_train_val_test_files(
        features_files)
    
    # TODO test this:
    if not datasets.train:
        # perhaps data files located in subdirectories 
        features_files = list(dataset_path.glob('**/*.npy'))
        datasets = sp.datasets.separate_train_val_test_files(
            features_files)
        if not datasets.train:
            raise FileNotFoundError('Could not locate train, validation, or test '+\
                '.npy files in the provided directory: \n{}'.format(dataset_path) +\
                    '\nThis program expects "train", "val", or "test" to be '+\
                        'included in each filename (not parent directory/ies) names.')
    
    # only need train and val feature data for autoencoder 
    train_paths_noisy, train_paths_clean = datasets.train.noisy, datasets.train.clean
    val_paths_noisy, val_paths_clean = datasets.val.noisy, datasets.val.clean
    
    # make sure both dataset pathways match in length and order:
    try:
        assert len(train_paths_noisy) == len(train_paths_clean)
        assert len(val_paths_noisy) == len(val_paths_clean)
    except AssertionError:
        raise ValueError('Noisy and Clean datasets do not match in length. '+\
            'They must be the same length.')
    train_paths_noisy = sorted(train_paths_noisy)
    train_paths_clean = sorted(train_paths_clean)
    val_paths_noisy = sorted(val_paths_noisy)
    val_paths_clean = sorted(val_paths_clean)
    
    # load smaller dataset to determine input size:
    data_val_noisy = np.load(val_paths_noisy[0])
    # expect shape (num_audiofiles, batch_size, num_frames, num_features)
    # but don't need batch size for denoiser... combine w num_frames in generator
    # with 'combine_axes_0_1' = True
    if len(data_val_noisy.shape) == 4:
        input_shape = (data_val_noisy.shape[1] * data_val_noisy.shape[2], 
                       data_val_noisy.shape[3], 
                       1)
        combine_axes_0_1 = True
    # expect shape (num_audiofiles, num_frames, num_features)
    elif len(data_val_noisy.shape) == 3:
        input_shape = data_val_noisy.shape[1:] + (1,)
        combine_axes_0_1 = False
    del data_val_noisy
    
    # setup model 
    denoiser, settings_dict = spdl.autoencoder_denoise(
        input_shape = input_shape)
    
    # create callbacks variable if not in kwargs
    # allow users to use different callbacks if desired
    if 'callbacks' not in kwargs:
        callbacks = spdl.setup_callbacks(patience = patience,
                                                best_modelname = model_path, 
                                                log_filename = model_dir.joinpath('log.csv'))
    adm = tf.keras.optimizers.Adam(learning_rate=0.0001)
    denoiser.compile(optimizer=adm, loss='binary_crossentropy')

    # TODO remove?
    # save variables that are not too large:
    local_variables = locals()
    global_variables = globals()
    sp.utils.save_dict(
        dict2save = local_variables, 
        filename = model_dir.joinpath('local_variables_{}.csv'.format(
                            model_name)),
                        overwrite=True)
    sp.utils.save_dict(
        dict2save = global_variables,
        filename = model_dir.joinpath('global_variables_{}.csv'.format(
                            model_name)),
        overwrite = True)
        
    # start training
    start = time.time()

    for i, train_path in enumerate(train_paths_noisy):
        if i == 0:
            if 'epochs' in kwargs:
                epochs = kwargs['epochs']
            else:
                epochs = 10 # default in Keras
                kwargs['epochs'] = epochs
            total_epochs = epochs * len(train_paths_noisy)
            print('\n\nThe model will be trained {} epochs per '.format(epochs)+\
                'training session. \nTotal possible epochs: {}\n\n'.format(total_epochs))
        start_session = time.time()
        data_train_noisy_path = train_path
        data_train_clean_path = train_paths_clean[i]
        # just use first validation data file
        data_val_noisy_path = val_paths_noisy[0]
        data_val_clean_path = val_paths_clean[0]

        print('\nTRAINING SESSION ',i+1)
        print("Training on: ")
        print(data_train_noisy_path)
        print(data_train_clean_path)
        print()
        
        data_train_noisy = np.load(data_train_noisy_path)
        data_train_clean = np.load(data_train_clean_path)
        data_val_noisy = np.load(data_val_noisy_path)
        data_val_clean = np.load(data_val_clean_path)

        # reinitiate 'callbacks' for additional iterations
        # TODO test for when callbacks already in **kwargs
        if i > 0: 
            if 'callbacks' not in kwargs:
                callbacks = spdl.setup_callbacks(
                    patience = patience,
                    best_modelname = model_path, 
                    log_filename = model_dir.joinpath('log.csv'))
            #else:
                ## apply callbacks set in **kwargs
                #callbacks = kwargs['callbacks']
        tensor = (1,)
        if use_generator:
            train_generator = spdl.Generator(
                data_matrix1 = data_train_noisy, 
                data_matrix2 = data_train_clean,
                normalize = normalize,
                desired_input_shape = tensor + input_shape,
                combine_axes_0_1 = combine_axes_0_1) # don't need batchsize / context window
            val_generator = spdl.Generator(
                data_matrix1 = data_val_noisy,
                data_matrix2 = data_val_clean,
                normalize = normalize,
                desired_input_shape = tensor + input_shape,
                combine_axes_0_1 = combine_axes_0_1) # don't need batchsize / context window

            feats_noisy, feats_clean = next(train_generator.generator())
            
            ds_train = tf.data.Dataset.from_generator(
                spdl.make_gen_callable(train_generator.generator()),
                output_types=(feats_noisy.dtype, feats_clean.dtype), 
                output_shapes=(feats_noisy.shape, 
                                feats_clean.shape))
            ds_val = tf.data.Dataset.from_generator(
                spdl.make_gen_callable(val_generator.generator()),
                output_types=(feats_noisy.dtype, feats_clean.dtype), 
                output_shapes=(feats_noisy.shape, 
                                feats_clean.shape))
                
            print(ds_train)
            print(ds_val)
                
            try:
                history = denoiser.fit(
                    ds_train,
                    steps_per_epoch = data_train_noisy.shape[0],
                    callbacks = callbacks,
                    validation_data = ds_val,
                    validation_steps = data_val_noisy.shape[0], 
                    **kwargs)
            except ValueError as e:
                print('\nValueError: ', e)
                raise ValueError('Try setting changing the parameter '+\
                    '`add_tensor_last` (in function '+\
                        '`soundpy.models.dataprep.Generator`)'+\
                        ' to either True, False, or None.')

        else:
            # reshape to mix samples and batchsizes:
            # if batch sizes are prevalent
            # need better way to distinguish this
            if len(data_train_noisy.shape) > 3:
                train_shape = (data_train_noisy.shape[0]*data_train_noisy.shape[1],)+ data_train_noisy.shape[2:] + (1,)
                val_shape = (data_val_noisy.shape[0]*data_val_noisy.shape[1],)+ data_val_noisy.shape[2:] + (1,)
            else:
                train_shape = data_train_noisy.shape + (1,)
                val_shape = data_val_noisy.shape + (1,)
            
            if normalize:
                data_train_noisy = sp.feats.normalize(data_train_noisy)
                data_train_clean = sp.feats.normalize(data_train_clean)
                data_val_noisy = sp.feats.normalize(data_val_noisy)
                data_val_clean = sp.feats.normalize(data_val_clean)
                
            X_train = data_train_noisy.reshape(train_shape)
            y_train = data_train_clean.reshape(train_shape)
            X_val = data_val_noisy.reshape(val_shape)
            y_val = data_val_clean.reshape(val_shape)
            
            history = denoiser.fit(X_train, y_train,
                            batch_size = data_train_noisy.shape[1],
                            callbacks = callbacks, 
                            validation_data = (X_val, y_val),
                            **kwargs)
        end_session = time.time()
        total_dur_sec_session = round(end_session-start_session,2)
        model_features_dict = dict(model_path = model_path,
                                data_train_noisy_path = data_train_noisy_path,
                                data_val_noisy_path = data_val_noisy_path, 
                                data_train_clean_path = data_train_clean_path, 
                                data_val_clean_path = data_val_clean_path,
                                total_dur_sec_session = total_dur_sec_session,
                                use_generator = use_generator,
                                kwargs = kwargs)
        model_features_dict.update(settings_dict)
        if i == len(train_paths_noisy)-1:
            end = time.time()
            total_duration_seconds = round(end-start,2)
            time_dict = dict(total_duration_seconds=total_duration_seconds)
            model_features_dict.update(time_dict)

        model_features_dict_path = model_dir.joinpath('info_{}_{}.csv'.format(
            model_name, i))
        model_features_dict_path = sp.utils.save_dict(
            dict2save = model_features_dict,
            filename = model_features_dict_path)
    print('\nFinished training the model. The model and associated files can be '+\
        'found here: \n{}'.format(model_dir))
    
    return model_dir, history

###############################################################################


# TODO include example extraction data in feature_extraction_dir?
def envclassifier_train(feature_extraction_dir,
                        model_name = 'model_cnn_classifier',
                        feature_type = None,
                        use_generator = True,
                        normalize = True,
                        patience = 15,
                        add_tensor_last = True,
                        num_layers = 3,
                        **kwargs):
    '''Collects training features and trains cnn environment classifier.
    
    This model may be applied to any speech and label scenario, for example, 
    male vs female speech, clinical vs healthy speech, simple speech / word
    recognition, as well as noise / scene / environment classification.
    
    Parameters
    ----------
    feature_extraction_dir : str or pathlib.PosixPath
        Directory where extracted feature files are located (format .npy).
    
    model_name : str
        The name for the model. This can be quite generic as the date up to 
        the millisecond will be added to ensure a unique name for each trained model.
        (default 'model_cnn_classifier')
        
    feature_type : str, optional
        The type of features that will be used to train the model. This is 
        only for the purposes of naming the model. If set to None, it will 
        not be included in the model name.
        
    use_generator : bool 
        If True, a generator will be used to feed training data to the model. Otherwise
        the entire training data will be used to train the model all at once.
        (default True)
        
    normalize : bool 
        If True, the data will be normalized before feeding to the model.
        (default False)
        
    patience : int 
        Number of epochs to train without improvement before early stopping.
        
    num_layers : int 
        The number of convolutional neural network layers desired. (default 3)
        
    **kwargs : additional keyword arguments
        The keyword arguments for keras.fit(). Note, 
        the keyword arguments differ for validation data so be sure to use the 
        correct keyword arguments, depending on if you use the generator or not.
        TODO: add link to keras.fit().
        
    Returns
    -------
    model_dir : pathlib.PosixPath
        The directory where the model and associated files can be found.
        
    See Also
    --------
    soundpy.datasets.separate_train_val_test_files
        Generates paths lists for train, validation, and test files. Useful
        for noisy vs clean datasets and also for multiple training files.
    
    soundpy.models.generator
        The generator function that feeds data to the model.
        
    soundpy.models.modelsetup.setup_callbacks
        The function that sets up callbacks (e.g. logging, save best model, early
        stopping, etc.)
        
    soundpy.models.template_models.cnn_classifier
        Template model architecture for a low-computational CNN sound classifier.
    '''
    # ensure feature_extraction_folder exists:
    if feature_extraction_dir is None:
        feature_extraction_dir = './audiodata/example_feats_models/envclassifier/'+\
            'features_fbank_6m20d0h18m11s123ms/'
    dataset_path = sp.utils.check_dir(feature_extraction_dir, make=False)
    
    # designate where to save model and related files
    if feature_type:
        model_name += '_'+feature_type + '_' + sp.utils.get_date() 
    else:
        model_name += '_' + sp.utils.get_date() 
    model_dir = dataset_path.joinpath(model_name)
    model_dir = sp.utils.check_dir(model_dir, make=True)
    model_name += '.h5'
    model_path = model_dir.joinpath(model_name)
    
    # prepare features files to load for training
    features_files = list(dataset_path.glob('*.npy'))
    # NamedTuple: 'datasets.train', 'datasets.val', 'datasets.test'
    datasets = sp.datasets.separate_train_val_test_files(
        features_files)
    
    # TODO test
    if not datasets.train:
        # perhaps data files located in subdirectories 
        features_files = list(dataset_path.glob('**/*.npy'))
        datasets = sp.datasets.separate_train_val_test_files(
            features_files)
        if not datasets.train:
            raise FileNotFoundError('Could not locate train, validation, or test '+\
                '.npy files in the provided directory: \n{}'.format(dataset_path) +\
                    '\nThis program expects "train", "val", or "test" to be '+\
                        'included in each filename (not parent directory/ies) names.')
    
    train_paths = datasets.train
    val_paths = datasets.val 
    test_paths = datasets.test
    
    # need dictionary for decoding labels:
    dict_decode_path = dataset_path.joinpath('dict_decode.csv')
    if not os.path.exists(dict_decode_path):
        raise FileNotFoundError('Could not find {}.'.format(dict_decode_path))
    dict_decode = sp.utils.load_dict(dict_decode_path)
    num_labels = len(dict_decode)
    
    # load smaller dataset to determine input size:
    data_val = np.load(val_paths[0])
    # expect shape (num_audiofiles, batch_size, num_frames, num_features + label_column)
    if len(data_val.shape) == 4:
        input_shape =  (data_val.shape[2], data_val.shape[3] - 1, 1) 
    # expect shape (num_audiofiles, num_frames, num_features + label_column)
    elif len(data_val.shape) == 3:
        input_shape = (data_val.shape[1], data_val.shape[2] - 1, 1) 
    # remove unneeded variable
    del data_val
    
    # setup model 

    
    feature_maps, kernels = spdl.setup_layers(num_features = input_shape[-2], 
                                              num_layers = num_layers)    
    
    envclassifier, settings_dict = spdl.cnn_classifier(
        feature_maps = feature_maps,
        kernel_size = kernels,
        input_shape = input_shape,
        num_labels = num_labels)
    if envclassifier is None:
        raise sp.errors.numfeatures_incompatible_templatemodel()
    
    # create callbacks variable if not in kwargs
    # allow users to use different callbacks if desired
    if 'callbacks' not in kwargs:
        callbacks = spdl.setup_callbacks(patience = patience,
                                                best_modelname = model_path, 
                                                log_filename = model_dir.joinpath('log.csv'))
    optimizer = 'adam'
    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
    envclassifier.compile(optimizer = optimizer,
                          loss = loss,
                          metrics = metrics)

    # TODO remove?
    # save variables that are not too large:
    local_variables = locals()
    global_variables = globals()
    sp.utils.save_dict(
        dict2save = local_variables, 
        filename = model_dir.joinpath('local_variables_{}.csv'.format(
                            model_name)),
        overwrite=True)
    sp.utils.save_dict(
        dict2save = global_variables,
        filename = model_dir.joinpath('global_variables_{}.csv'.format(
                            model_name)),
        overwrite = True)
        
    # start training
    start = time.time()

    for i, train_path in enumerate(train_paths):
        if i == 0:
            if 'epochs' in kwargs:
                epochs = kwargs['epochs']
            else:
                epochs = 10 # default in Keras
            total_epochs = epochs * len(train_paths)
            print('\n\nThe model will be trained {} epochs per '.format(epochs)+\
                'training session. \nTotal possible epochs: {}\n\n'.format(total_epochs))
        start_session = time.time()
        data_train_path = train_path
        # just use first validation data file
        data_val_path = val_paths[0]
        # just use first test data file
        data_test_path = test_paths[0]
        
        print('\nTRAINING SESSION ',i+1)
        print("Training on: ")
        print(data_train_path)
        print()
        
        data_train = np.load(data_train_path)
        data_val = np.load(data_val_path)
        data_test = np.load(data_test_path)
        
        # reinitiate 'callbacks' for additional iterations
        if i > 0: 
            if 'callbacks' not in kwargs:
                callbacks = spdl.setup_callbacks(patience = patience,
                                                        best_modelname = model_path, 
                                                        log_filename = model_dir.joinpath('log.csv'))
            else:
                # apply callbacks set in **kwargs
                callbacks = kwargs['callbacks']

        # might need to add tensor dimension to `desired_input_shape`
        tensor = (1,)
        if use_generator:
            train_generator = spdl.Generator(
                data_matrix1 = data_train, 
                data_matrix2 = None,
                normalize = normalize,
                desired_input_shape = tensor + input_shape)
            val_generator = spdl.Generator(
                data_matrix1 = data_val,
                data_matrix2 = None,
                normalize = normalize,
                desired_input_shape = tensor + input_shape)
            test_generator = spdl.Generator(
                data_matrix1 = data_test,
                data_matrix2 = None,
                normalize = normalize,
                desired_input_shape = tensor + input_shape)
            # resource:
            # https://www.tensorflow.org/guide/data
            
            feats, label = next(train_generator.generator())
        
            ds_train = tf.data.Dataset.from_generator(
                spdl.make_gen_callable(train_generator.generator()),
                output_types=(feats.dtype, label.dtype), 
                output_shapes=(feats.shape, 
                                label.shape))
            ds_val = tf.data.Dataset.from_generator(
                spdl.make_gen_callable(val_generator.generator()),
                output_types=(feats.dtype, label.dtype), 
                output_shapes=(feats.shape, 
                                label.shape))
            ds_test = tf.data.Dataset.from_generator(
                spdl.make_gen_callable(test_generator.generator()),
                output_types=(feats.dtype, label.dtype), 
                output_shapes=(feats.shape, 
                                label.shape))

            print(ds_train)
            print(ds_val)
            print(ds_test)
            
            history = envclassifier.fit(
                ds_train,
                steps_per_epoch = data_train.shape[0],
                callbacks = callbacks,
                validation_data = ds_val,
                validation_steps = data_val.shape[0],
                **kwargs)
            
            ## TODO test how well prediction works. use simple predict instead?
            ## need to define `y_test`
            #X_test, y_test = sp.feats.separate_dependent_var(data_test)
            #y_predicted = envclassifier.predict(
                #ds_train,
                #steps = data_test.shape[0])
            score = envclassifier.evaluate(ds_test, steps=500) 

        else:
            # TODO make scaling data optional?
            # data is separated and shaped for this classifier in scale_X_y..
            X_train, y_train, scalars = sp.feats.scale_X_y(data_train,
                                                                is_train=True)
            X_val, y_val, __ = sp.feats.scale_X_y(data_val,
                                                    is_train=False, 
                                                    scalars=scalars)
            X_test, y_test, __ = sp.feats.scale_X_y(data_test,
                                                        is_train=False, 
                                                        scalars=scalars)
            
            history = envclassifier.fit(X_train, y_train, 
                                        callbacks = callbacks, 
                                        validation_data = (X_val, y_val),
                                        **kwargs)
            
            score = envclassifier.evaluate(X_test, y_test)
            
        print('Test loss:', score[0]) 
        print('Test accuracy:', score[1])
        end_session = time.time()
        total_dur_sec_session = round(end_session-start_session,2)
        model_features_dict = dict(model_path = model_path,
                                data_train_path = data_train_path,
                                data_val_path = data_val_path, 
                                data_test_path = data_test_path, 
                                total_dur_sec_session = total_dur_sec_session,
                                use_generator = use_generator,
                                score = score,
                                kwargs = kwargs)
        model_features_dict.update(settings_dict)
        if i == len(train_paths)-1:
            end = time.time()
            total_duration_seconds = round(end-start,2)
            time_dict = dict(total_duration_seconds=total_duration_seconds)
            model_features_dict.update(time_dict)

        model_features_dict_path = model_dir.joinpath('info_{}_{}.csv'.format(
            model_name, i))
        model_features_dict_path = sp.utils.save_dict(
            filename = model_features_dict_path,
            dict2save = model_features_dict)
    print('\nFinished training the model. The model and associated files can be '+\
        'found here: \n{}'.format(model_dir))
    
    return model_dir, history

def denoiser_run(model, new_audio, feat_settings_dict, remove_dc=True):
    '''Implements a pre-trained denoiser
    
    Parameters
    ----------
    model : str or pathlib.PosixPath
        The path to the denoising model.
    
    new_audio : str, pathlib.PosixPath, or np.ndarray
        The path to the noisy audiofile.
        
    feat_settings_dict : dict 
        Dictionary containing necessary settings for how the features were
        extracted for training the model. Expected keys: 'feature_type', 
        'win_size_ms', 'percent_overlap', 'sr', 'window', 'frames_per_sample',
        'input_shape', 'desired_shape', 'dur_sec', 'num_feats'.
        
    Returns
    -------
    cleaned_audio : np.ndarray [shape = (num_samples, )]
        The cleaned audio samples ready for playing or saving as audio file.
    sr : int 
        The sample rate of `cleaned_audio`.
        
    See Also
    --------
    soundpy.feats.get_feats
        How features are extracted.
        
    soundpy.feats.feats2audio
        How features are transformed back into audio samples.
    '''    
    featsettings = sp.feats.load_feat_settings(feat_settings_dict)
    
    feats = sp.feats.get_feats(
        new_audio, 
        sr = featsettings.sr, 
        feature_type = featsettings.feature_type,
        win_size_ms = featsettings.win_size_ms,
        percent_overlap = featsettings.percent_overlap,
        window = featsettings.window, 
        dur_sec = featsettings.dur_sec,
        num_filters = featsettings.num_feats,
        num_mfcc = featsettings.num_mfcc,
        fft_bins = featsettings.fft_bins,
        remove_first_coefficient = featsettings.remove_first_coefficient,
        sinosoidal_liftering = featsettings.sinosoidal_liftering,
        mono = featsettings.mono,
        rate_of_change = featsettings.rate_of_change,
        rate_of_acceleration = featsettings.rate_of_acceleration,
        subtract_mean = featsettings.subtract_mean,
        real_signal = featsettings.real_signal,
        fmin = featsettings.fmin,
        fmax = featsettings.fmax,
        zeropad = featsettings.zeropad)
    
    # are phase data still present? (only in stft features)
    if feats.dtype == np.complex and np.min(feats) < 0:
        original_phase = sp.dsp.calc_phase(feats,
                                               radians=False)
    elif 'stft' in feature_type or 'powspec' in featsettings.feature_type:
        feats_stft = sp.feats.get_feats(
            new_audio, 
            sr = featsettings.sr, 
            feature_type = 'stft',
            win_size_ms = featsettings.win_size_ms,
            percent_overlap = featsettings.percent_overlap,
            window = featsettings.window, 
            dur_sec = featsettings.dur_sec,
            fft_bins = featsettings.fft_bins,
            mono = featsettings.mono)
        original_phase = sp.dsp.calc_phase(feats_stft,
                                               radians = False)
    else:
        original_phase = None
    
    if 'signal' in featsettings.feature_type:
        feats_zeropadded = np.zeros(featsettings.base_shape)
        feats_zeropadded = feats_zeropadded.flatten()
        if len(feats.shape) > 1:
            feats_zeropadded = feats_zeropadded.reshape(feats_zeropadded.shape[0],
                                                        feats.shape[1])
        if len(feats) > len(feats_zeropadded):
            feats = feats[:len(feats_zeropadded)]
        feats_zeropadded[:len(feats)] += feats
        # reshape here to avoid memory issues if total # samples is large
        feats = feats_zeropadded.reshape(featsettings.base_shape)
    
    # add a tensor dimension to either first or last channel.. whatever works I guess?
    # keras..
    tensor = (1,)
    feats = sp.feats.prep_new_audiofeats(feats,
                                           featsettings.base_shape,
                                           featsettings.input_shape)# tensor alread included

    # ensure same shape as feats
    if original_phase is not None:
        original_phase = sp.feats.prep_new_audiofeats(original_phase,
                                                        featsettings.base_shape,
                                                        featsettings.input_shape)
    
    feats_normed = sp.feats.normalize(feats)
    denoiser = load_model(model)
    if len(feats_normed.shape) >= 3:
        batch_size = feats_normed.shape[0]
        # newer version soundpy 0.1.0a3
        #feats_normed = feats_normed.reshape((1,) + feats_normed.shape)
        # ValueError: Error when checking input: expected conv2d_1_input to have shape (11, 177, 1) but got array with shape (35, 11, 177)
        feats_normed = feats_normed.reshape(feats_normed.shape + (1,))
        try:
            cleaned_feats = denoiser.predict(feats_normed, batch_size = batch_size)
        except ValueError:
            # newer version soundpy 0.1.0a3
            import warnings 
            msg = '\nWARNING: adjustments to feature extraction in a more recent'+\
                ' SoundPy version may result in imperfect feature alignment '+\
                    'with a model trained with features generated with a previous'+\
                        ' SoundPy version. Sincerest apologies!'
            warnings.warn(msg)
            feats_normed = feats_normed.reshape(feats_normed.shape[1:])
            cleaned_feats = denoiser.predict(feats_normed, batch_size = batch_size)
    else:
        feats_normed = feats_normed.reshape((1,)+feats_normed.shape)
        cleaned_feats = denoiser.predict(feats_normed)
    
    try:
        # need to change shape back to 2D
        # current shape is (batch_size, num_frames, num_features, 1)
        # need (num_frames, num_features)

        # remove last tensor dimension
        if feats_normed.shape[-1] == 1:
            feats_normed = feats_normed.reshape(feats_normed.shape[:-1])
        feats_flattened = feats_normed.reshape(-1, 
                                                feats_normed.shape[-1])
        audio_shape = (feats_flattened.shape)
        
        cleaned_feats = cleaned_feats.reshape(audio_shape)
        if original_phase is not None:
            original_phase = original_phase.reshape(audio_shape)
        
        # now combine them to create audio samples:
        cleaned_audio = sp.feats.feats2audio(
            cleaned_feats, 
            feature_type = featsettings.feature_type,
            sr = featsettings.sr, 
            win_size_ms = featsettings.win_size_ms,
            percent_overlap = featsettings.percent_overlap,
            phase = original_phase)
        if not isinstance(new_audio, np.ndarray):
            noisy_audio, __ = sp.loadsound(new_audio, 
                                           sr = featsettings.sr,
                                           remove_dc = remove_dc)
        else:
            noisy_audio = new_audio
        if len(cleaned_audio) > len(noisy_audio):
            cleaned_audio = cleaned_audio[:len(noisy_audio)]
        
        max_energy_original = np.max(noisy_audio)
        # match the scale of the original audio:
        cleaned_audio = sp.dsp.scalesound(cleaned_audio, max_val = max_energy_original)
    except librosa.ParameterError as e:
        import warnings
        msg = '\nlibrosa.ParameterError: {}'.format(e)+\
            '\nUnable to convert cleaned features to raw audio samples.'+\
                '\nReturning cleaned audio in {} features.'.format(featsettings.feature_type)
        warnings.warn(msg)
        cleaned_audio = cleaned_feats
    return cleaned_audio, featsettings.sr


def envclassifier_run(model, new_audio, feat_settings_dict, dict_decode):
    '''Implement a convnet model with `new_audio`.
    
    Parameters
    ----------
    model : str, pathlib.PosixPath
        The pathway to the pre-trained model.
        
    new_audio : str, pathlib.PosixPath
        The pathway to the audio file to be classified.
        
    feat_settings_dict : dict 
        Dictionary containing necessary settings for feature extraction, such
        as sample rate, feature type, etc.
        
    dict_decode : dict 
        Dictionary containing encoded labels as keys and string labels as values.
        for example {0:'office', 1:'traffic', 2:'park'}.
        
    Returns
    -------
    label : int 
        The encoded label applied to the `new_audio`.
    
    label_string : str 
        The string label applied to the `new_audio`.
    
    strength : float 
        The confidence of the model's assignment. For example, 0.99 would be very 
        confident, 0.51 would not be very confident.
    '''
    featsettings = sp.feats.load_feat_settings(feat_settings_dict)
    
    feats = sp.feats.get_feats(
        new_audio, 
        sr = featsettings.sr, 
        feature_type = featsettings.feature_type,
        win_size_ms = featsettings.win_size_ms,
        percent_overlap = featsettings.percent_overlap,
        window = featsettings.window, 
        dur_sec = featsettings.dur_sec,
        num_filters = featsettings.num_feats,
        num_mfcc = featsettings.num_mfcc,
        fft_bins = featsettings.fft_bins,
        remove_first_coefficient = featsettings.remove_first_coefficient,
        sinosoidal_liftering = featsettings.sinosoidal_liftering,
        mono = featsettings.mono,
        rate_of_change = featsettings.rate_of_change,
        rate_of_acceleration = featsettings.rate_of_acceleration,
        subtract_mean = featsettings.subtract_mean,
        real_signal = featsettings.real_signal,
        fmin = featsettings.fmin,
        fmax = featsettings.fmax,
        zeropad = featsettings.zeropad)
    
    # load info csv with model input shape
    model_path = sp.utils.string2pathlib(model)
    model_info_path = model.parent.glob('*.csv')
    model_info_path = [i for i in model_info_path if 'info' in i.stem][0]
    model_info = sp.utils.load_dict(model_info_path)
    for key, value in model_info.items():
        model_info[key] = sp.utils.restore_dictvalue(value)
    input_shape = model_info['input_shape']
    
    feats = sp.feats.prep_new_audiofeats(feats,
                                         featsettings.base_shape,
                                         input_shape)
    
    feats_normed = sp.feats.normalize(feats)
    envclassifier = load_model(model)
    tensor = (1,)
    feats_normed = feats_normed.reshape(tensor + feats_normed.shape)
    prediction = envclassifier.predict(feats_normed)
    label = np.argmax(prediction)
    strength = prediction[0][label]
    try:
        label_string = dict_decode[label]
    except KeyError:
        label_string = dict_decode[str(int(label))]
    return label, label_string, strength


def collect_classifier_settings(feature_extraction_dir):
    '''Collects relevant information for some models from files in the feature directory.
    
    These relevant files have been generated in `soundpy.models.builtin.envclassifier_train`.
    
    Parameters
    ----------
    feature_extraction_dir : str, pathlib.PosixPath
        The directory where extracted files are located, included .npy and .csv log files.
        
    Returns
    -------
    datasets : NamedTuple
        A named tuple containing train, val, and test data
    
    num_labels : int 
        The number of labels used for the data.
    
    feat_shape : tuple
        The initial shape of the features when they were extracted. For example, labels 
        or context window not applied.
    
    num_feats : int 
        The number of features used to train the pre-trained model.
    
    feature_type : str 
        The `feature_type` used to train the pre-trained model. For example, 'fbank', 
        'mfcc', 'stft', 'signal', 'powspec'.
        
    See Also
    --------
    soundpy.models.builtin.envclassifier_train
        The builtin functionality for training a simple scene/environment/speech
        classifier. This function generates the files expected by this function.
    '''
    # ensure feature_extraction_folder exists:
    dataset_path = sp.utils.check_dir(feature_extraction_dir, make=False)
    
    # prepare features files to load for training
    features_files = list(dataset_path.glob('*.npy'))
    # NamedTuple: 'datasets.train', 'datasets.val', 'datasets.test'
    datasets = sp.datasets.separate_train_val_test_files(
        features_files)
    # TODO test
    if not datasets.train:
        # perhaps data files located in subdirectories 
        features_files = list(dataset_path.glob('**/*.npy'))
        datasets = sp.datasets.separate_train_val_test_files(
            features_files)
        if not datasets.train:
            raise FileNotFoundError('Could not locate train, validation, or test '+\
                '.npy files in the provided directory: \n{}'.format(dataset_path) +\
                    '\nThis program expects "train", "val", or "test" to be '+\
                        'included in each filename (not parent directory/ies) names.')
        
    train_paths = datasets.train
    val_paths = datasets.val 
    test_paths = datasets.test
    
    # need dictionary for decoding labels:
    dict_decode_path = dataset_path.joinpath('dict_decode.csv')
    if not os.path.exists(dict_decode_path):
        raise FileNotFoundError('Could not find {}.'.format(dict_decode_path))
    dict_decode = sp.utils.load_dict(dict_decode_path)
    num_labels = len(dict_decode)
    
    settings_dict = sp.utils.load_dict(
        dataset_path.joinpath('log_extraction_settings.csv'))
    if 'kwargs' in settings_dict:
        kwargs = sp.utils.restore_dictvalue(settings_dict['kwargs'])
        settings_dict.update(kwargs)
    # should the shape include the label column or not?
    # currently not
    try:
        feat_shape = sp.utils.restore_dictvalue(settings_dict['desired_shape'])
    except KeyError:
        feat_shape = sp.utils.restore_dictvalue(settings_dict['feat_base_shape'])
    try:
        num_feats = sp.utils.restore_dictvalue(settings_dict['num_feats'])
    except KeyError:
        num_feats = feat_shape[-1]
    try:
        feature_type = settings_dict['feat_type']
    except KeyError:
        feature_type = settings_dict['feature_type']
    return datasets, num_labels, feat_shape, num_feats, feature_type

# TODO cleanup
# TODO test
def cnnlstm_train(feature_extraction_dir,
                  model_name = 'model_cnnlstm_classifier',
                  use_generator = True,
                  normalize = True,
                  patience = 15,
                  timesteps = 10,
                  context_window = 5,
                  frames_per_sample = None,
                  colorscale = 1,
                  total_training_sessions = None,
                  add_tensor_last = False,
                  **kwargs):
    '''Example implementation of a Convnet+LSTM model for speech recognition.
    
    Note: improvements must still be made, for example with the `context_window`. However,
    this still may be useful as an example of a simple CNN and LSTM model.
    
    Parameters
    ----------
    feature_extraction_dir : str, pathlib.PosixPath
        The directory where feature data will be saved.
        
    model_name : str 
        The name of the model. (default 'model_cnnlstm_classifier')
    
    use_generator : True 
        If True, data will be fed to the model via generator. This parameter will likely 
        be removed and set as a default. (default True)
    
    normalize : bool 
        If True, the data will be normalized before being fed to the model. (default True)
    
    patience : int 
        The number of epochs to allow with no improvement in either val accuracy or loss.
        (default 15)
        
    timesteps : int 
        The frames dedicated to each subsection of each sample. This allows the long-short
        term memory model to process each subsection consecutively.
        
    context_window : int 
        The number of frames surrounding a central frame that make up sound context. Note:
        this needs improvement and further exploration.
        
    frames_per_sample : int 
        Serves basically same role as `context_window` does currently: `frames_per_sample`
        equals `context_window` * 2 + 1. This parameter will likely be removed in future 
        versions.
        
    colorscale : int 
        The colorscale relevant for the convolutional neural network. (default 1)
        
    total_training_sessions : int 
        Option to limit number of audiofiles used for training, if `use_generator` is 
        set to False. This parameter will likely be removed in future versions. But as
        this is just an example model, the low priority may result in this parameter
        living forever.
        
    add_tensor_last : bool 
        No longer used in the code. Irrelevant. 
        
    kwargs : additional keyword arguments.
        Keyword arguments for `keras.model.fit`.
        
    Returns
    -------
    model_dir : pathlib.PosixPath 
        The directory where model and log files are saved.
    
    history : tf.keras.callbacks.History
        Contains model training and validation accuracy and loss throughout training.
    
    References
    ----------
    Kim, Myungjong & Cao, Beiming & An, Kwanghoon & Wang, Jun. (2018). Dysarthric Speech Recognition Using Convolutional LSTM Neural Network. 10.21437/interspeech.2018-2250.
    '''
    
    datasets, num_labels, feat_shape, num_feats, feature_type =\
        collect_classifier_settings(feature_extraction_dir)
    
    train_paths = datasets.train
    val_paths = datasets.val
    test_paths = datasets.test
    
    # Save model directory inside feature directory
    dataset_path = train_paths[0].parent
    if feature_type:
        model_name += '_'+feature_type + '_' + sp.utils.get_date() 
    else:
        model_name += '_' + sp.utils.get_date() 
    model_dir = dataset_path.joinpath(model_name)
    model_dir = sp.utils.check_dir(model_dir, make=True)
    model_name += '.h5'
    model_path = model_dir.joinpath(model_name)
    
    if frames_per_sample is not None:
        raise DeprecationWarning('In future versions, the `frames_per_sample` '+\
            'parameter will be no longer used.\n'+\
                'Instead features can be segmented in generator functions using the '+\
                    'parameter `context_window`: `soundpy.models.dataprep.Generator`.')
        
    if context_window is not None: # by default it is not None
        if frames_per_sample is None:
            frame_width = context_window * 2 + 1 # context window w central frame
        else:
            frame_width = frames_per_sample
    elif frames_per_sample is not None:
        frame_width = frames_per_sample
    input_shape = (timesteps, frame_width, num_feats, colorscale)
    model, settings = spdl.cnnlstm_classifier(num_labels = num_labels, 
                                                    input_shape = input_shape, 
                                                    lstm_cells = num_feats)
    
    #print('cnnlstm desired input shape: ', input_shape)
    #cnnlstm desired input shape:  (10, 11, 221, 1)
    #train data shape:  (7433, 99, 222)
    
    #start
    #(99, 221)
    #timestep
    #(10, 10, 221)
    #context_window (with zeropadding)
    #(10, 11, 221)

    # create callbacks variable if not in kwargs
    # allow users to use different callbacks if desired
    # TODO test how it works when callbacks set in kwargs.
    if 'callbacks' not in kwargs:
        callbacks = spdl.setup_callbacks(patience = patience,
                                                best_modelname = model_path, 
                                                log_filename = model_dir.joinpath('log.csv'))
    optimizer = 'adam'
    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer = optimizer,
                          loss = loss,
                          metrics = metrics)
    
    # update settings with optimizer etc.
    additional_settings = dict(optimizer = optimizer,
                               loss = loss,
                               metrics = metrics,
                               kwargs = kwargs)
    settings.update(additional_settings)
    
    # start training
    start = time.time()

    for i, train_path in enumerate(train_paths):
        if i == 0:
            if 'epochs' in kwargs:
                epochs = kwargs['epochs']
            else:
                epochs = 10 # default in Keras
            total_epochs = epochs * len(train_paths)
            print('\n\nThe model will be trained {} epochs per '.format(epochs)+\
                'training session. \nTotal possible epochs: {}\n\n'.format(total_epochs))
        start_session = time.time()
        data_train_path = train_path
        # just use first validation data file
        data_val_path = val_paths[0]
        # just use first test data file
        data_test_path = test_paths[0]
        
        print('\nTRAINING SESSION ',i+1)
        print("Training on: ")
        print(data_train_path)
        print()
        
        data_train = np.load(data_train_path)
        data_val = np.load(data_val_path)
        data_test = np.load(data_test_path)
        
        print('\ntrain data shape: ', data_train.shape)
        print()
        
        # shuffle data_train, just to ensure random
        np.random.shuffle(data_train) 
        
        # reinitiate 'callbacks' for additional iterations
        if i > 0: 
            if 'callbacks' not in kwargs:
                callbacks = spdl.setup_callbacks(patience = patience,
                                                        best_modelname = model_path, 
                                                        log_filename = model_dir.joinpath('log.csv'))
            else:
                # apply callbacks set in **kwargs
                callbacks = kwargs['callbacks']

        if use_generator:
            train_generator = spdl.Generator(data_matrix1 = data_train, 
                                                    data_matrix2 = None,
                                                    normalize = normalize,
                                                    timestep = timesteps,
                                                    axis_timestep = 0,
                                                    context_window = context_window,
                                                    axis_context_window = -2, 
                                                    desired_input_shape = (1,)+input_shape,
                                                    )
                                                    # expecting features in last axis
                                                    # add_tensor_last = add_tensor_last)
            val_generator = spdl.Generator(data_matrix1 = data_val,
                                                data_matrix2 = None,
                                                normalize = normalize,
                                                timestep = timesteps,
                                                axis_timestep = 0,
                                                context_window = context_window,
                                                axis_context_window = -2, 
                                                desired_input_shape = (1,)+input_shape,
                                                )
                                                    #add_tensor_last = add_tensor_last)
            test_generator = spdl.Generator(data_matrix1 = data_test,
                                                  data_matrix2 = None,
                                                  normalize = normalize,
                                                    timestep = timesteps,
                                                    axis_timestep = 0,
                                                    context_window = context_window,
                                                    axis_context_window = -2, 
                                                    desired_input_shape = (1,)+input_shape,
                                                    )

            feats, label = next(train_generator.generator())
            print('generator items:')
            print('feature shape')
            print(feats.shape)
            print('label')
            print(label)
            #sp.feats.plot(feats, feature_type='stft', save_pic = True,
                          #name4pic = 'cnnlstm_test.png')


            ds_train = tf.data.Dataset.from_generator(
                spdl.make_gen_callable(train_generator.generator()),
                output_types=(feats.dtype, label.dtype), 
                output_shapes=(feats.shape, 
                                label.shape))
            ds_val = tf.data.Dataset.from_generator(
                spdl.make_gen_callable(val_generator.generator()),
                output_types=(feats.dtype, label.dtype), 
                output_shapes=(feats.shape, 
                                label.shape))
            ds_test = tf.data.Dataset.from_generator(
                spdl.make_gen_callable(test_generator.generator()),
                output_types=(feats.dtype, label.dtype), 
                output_shapes=(feats.shape, 
                                label.shape))
                
            print(ds_train)
            print(ds_val)
            print(ds_test)

            history = model.fit(
                ds_train,
                steps_per_epoch = data_train.shape[0],
                callbacks = callbacks,
                validation_data = ds_val,
                validation_steps = data_val.shape[0],
                **kwargs)
            
            score = model.evaluate(ds_test, steps=500) 

        else:
            # TODO make scaling data optional?
            # TODO remove option for non-generator fed data..?
            # data is separated and shaped for this classifier in scale_X_y..
            X_train, y_train, scalars = sp.feats.scale_X_y(data_train,
                                                                is_train=True)
            X_val, y_val, __ = sp.feats.scale_X_y(data_val,
                                                    is_train=False, 
                                                    scalars=scalars)
            X_test, y_test, __ = sp.feats.scale_X_y(data_test,
                                                        is_train=False, 
                                                        scalars=scalars)
            
            X_train = sp.feats.adjust_shape(X_train, 
                                              (X_train.shape[0],)+input_shape,
                                              change_dims = True)
            
            X_val = sp.feats.adjust_shape(X_val, 
                                            (X_val.shape[0],)+input_shape,
                                              change_dims = True)
            X_test = sp.feats.adjust_shape(X_test, 
                                             (X_test.shape[0],)+input_shape,
                                              change_dims = True)
            
            # randomize train data
            rand_idx = np.random.choice(range(len(X_train)),
                                        len(X_train),
                                        replace=False)
            X_train = X_train[rand_idx]
            
            history = model.fit(X_train, y_train, 
                                        callbacks = callbacks, 
                                        validation_data = (X_val, y_val),
                                        **kwargs)
            
            score = model.evaluate(X_test, y_test)
            
        print('Test loss:', score[0]) 
        print('Test accuracy:', score[1])
        end_session = time.time()
        total_dur_sec_session = round(end_session-start_session,2)
        model_features_dict = dict(model_path = model_path,
                                data_train_path = data_train_path,
                                data_val_path = data_val_path, 
                                data_test_path = data_test_path, 
                                total_dur_sec_session = total_dur_sec_session,
                                use_generator = use_generator,
                                score = score,
                                kwargs = kwargs)
        model_features_dict.update(settings)
        model_features_dict_path = model_dir.joinpath('info_{}_{}.csv'.format(
            model_name, i))
        model_features_dict_path = sp.utils.save_dict(
            filename = model_features_dict_path,
            dict2save = model_features_dict)
        if total_training_sessions is None:
            total_training_sessions = len(train_paths)
        if i == total_training_sessions-1:
            end = time.time()
            total_duration_seconds = round(end-start,2)
            time_dict = dict(total_duration_seconds=total_duration_seconds)
            model_features_dict.update(time_dict)

            model_features_dict_path = model_dir.joinpath('info_{}_{}.csv'.format(
                model_name, i))
            model_features_dict_path = sp.utils.save_dict(
                filename = model_features_dict_path,
                dict2save = model_features_dict,
                overwrite = True)
            print('\nFinished training the model. The model and associated files can be '+\
            'found here: \n{}'.format(model_dir))
            model.save(model_dir.joinpath('final_not_best_model.h5'))
            return model_dir, history

# TODO cleanup
# TODO test
def resnet50_train(feature_extraction_dir,
                   model_name = 'model_resnet50_classifier',
                   use_generator = True,
                   normalize = True,
                   patience = 15,
                   colorscale = 3,
                   total_training_sessions = None,
                   **kwargs):
    '''Continue training a pre-trained resnet50 model for speech recogntion or other sound classification.
    
    Parameters
    ----------
    feature_extraction_dir : str or pathlib.PosixPath
        The directory where feature extraction files will be saved.
        
    model_name : str 
        The name for the model. (default 'model_resnet50_classifier')
        
    use_generator : True 
        If True, data will be fed to the model via generator. This parameter will likely 
        be removed and set as a default. (default True)
    
    normalize : bool 
        If True, the data will be normalized before being fed to the model. (default True)
    
    patience : int 
        The number of epochs to allow with no improvement in either val accuracy or loss.
        (default 15)
        
    timesteps : int 
        The frames dedicated to each subsection of each sample. This allows the long-short
        term memory model to process each subsection consecutively.
        
    context_window : int 
        The number of frames surrounding a central frame that make up sound context. Note:
        this needs improvement and further exploration.
        
    frames_per_sample : int 
        Serves basically same role as `context_window` does currently: `frames_per_sample`
        equals `context_window` * 2 + 1. This parameter will likely be removed in future 
        versions.
        
    colorscale : int 
        The colorscale relevant for the convolutional neural network. (default 1)
        
    total_training_sessions : int 
        Option to limit number of audiofiles used for training, if `use_generator` is 
        set to False. This parameter will likely be removed in future versions. But as
        this is just an example model, the low priority may result in this parameter
        living forever.
        
    Returns
    -------
    model_dir : pathlib.PosixPath 
        The directory where model and log files are saved.
    
    history : tf.keras.callbacks.History()
        Contains model training and validation accuracy and loss throughout training.
    '''
    datasets, num_labels, feat_shape, num_feats, feature_type =\
        collect_classifier_settings(feature_extraction_dir)
    
    train_paths = datasets.train
    val_paths = datasets.val
    test_paths = datasets.test
    
    # Save model directory inside feature directory
    dataset_path = train_paths[0].parent
    if feature_type:
        model_name += '_'+feature_type + '_' + sp.utils.get_date() 
    else:
        model_name += '_' + sp.utils.get_date() 
    model_dir = dataset_path.joinpath(model_name)
    model_dir = sp.utils.check_dir(model_dir, make=True)
    model_name += '.h5'
    model_path = model_dir.joinpath(model_name)
    
    input_shape = (feat_shape[0], num_feats, colorscale)
    model, settings = spdl.resnet50_classifier(num_labels = num_labels, 
                                                    input_shape = input_shape)

    # create callbacks variable if not in kwargs
    # allow users to use different callbacks if desired
    if 'callbacks' not in kwargs:
        callbacks = spdl.setup_callbacks(patience = patience,
                                                best_modelname = model_path, 
                                                log_filename = model_dir.joinpath('log.csv'))
    optimizer = Adam(lr=0.0001)
    loss='sparse_categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss = loss, 
                metrics = metrics)
    
    # update settings with optimizer etc.
    additional_settings = dict(optimizer = optimizer,
                               loss = loss,
                               metrics = metrics,
                               kwargs = kwargs)
    settings.update(additional_settings)
    
    
    # start training
    start = time.time()

    for i, train_path in enumerate(train_paths):
        if i == 0:
            if 'epochs' in kwargs:
                epochs = kwargs['epochs']
            else:
                epochs = 10 # default in Keras
            total_epochs = epochs * len(train_paths)
            print('\n\nThe model will be trained {} epochs per '.format(epochs)+\
                'training session. \nTotal possible epochs: {}\n\n'.format(total_epochs))
        start_session = time.time()
        data_train_path = train_path
        # just use first validation data file
        data_val_path = val_paths[0]
        # just use first test data file
        data_test_path = test_paths[0]
        
        print('\nTRAINING SESSION ',i+1)
        print("Training on: ")
        print(data_train_path)
        print()
        
        data_train = np.load(data_train_path)
        data_val = np.load(data_val_path)
        data_test = np.load(data_test_path)
        
        # shuffle data_train, just to ensure random
        np.random.shuffle(data_train) 
        
        # reinitiate 'callbacks' for additional iterations
        if i > 0: 
            if 'callbacks' not in kwargs:
                callbacks = spdl.setup_callbacks(patience = patience,
                                                        best_modelname = model_path, 
                                                        log_filename = model_dir.joinpath('log.csv'))
            else:
                # apply callbacks set in **kwargs
                callbacks = kwargs['callbacks']

        tensor = (1,)
        if use_generator:
            train_generator = spdl.Generator(
                data_matrix1 = data_train, 
                data_matrix2 = None,
                normalize = normalize,
                desired_input_shape = tensor + input_shape,
                gray2color = True)
            val_generator = spdl.Generator(
                data_matrix1 = data_val,
                data_matrix2 = None,
                normalize = normalize,
                desired_input_shape = tensor + input_shape,
                gray2color = True)
            test_generator = spdl.Generator(
                data_matrix1 = data_test,
                data_matrix2 = None,
                normalize = normalize,
                desired_input_shape = tensor + input_shape,
                gray2color = True)

            feats, label = next(train_generator.generator())
            
            ds_train = tf.data.Dataset.from_generator(
                spdl.make_gen_callable(train_generator.generator()),
                output_types=(feats.dtype, label.dtype), 
                output_shapes=(feats.shape, 
                                label.shape))
            ds_val = tf.data.Dataset.from_generator(
                spdl.make_gen_callable(val_generator.generator()),
                output_types=(feats.dtype, label.dtype), 
                output_shapes=(feats.shape, 
                                label.shape))
            ds_test = tf.data.Dataset.from_generator(
                spdl.make_gen_callable(test_generator.generator()),
                output_types=(feats.dtype, label.dtype), 
                output_shapes=(feats.shape, 
                                label.shape))


            print(ds_train)
            print(ds_val)
            print(ds_test)

            history = model.fit(
                ds_train,
                steps_per_epoch = data_train.shape[0],
                callbacks = callbacks,
                validation_data = ds_val,
                validation_steps = data_val.shape[0],
                **kwargs)
            
            score = model.evaluate(ds_test, steps=500) 
        else:
            # TODO make scaling data optional?
            # data is separated and shaped for this classifier in scale_X_y..
            X_train, y_train, scalars = sp.feats.scale_X_y(data_train,
                                                                is_train=True)
            X_val, y_val, __ = sp.feats.scale_X_y(data_val,
                                                    is_train=False, 
                                                    scalars=scalars)
            X_test, y_test, __ = sp.feats.scale_X_y(data_test,
                                                        is_train=False, 
                                                        scalars=scalars)
            
            print(X_train.shape)
            X_train = sp.feats.adjust_shape(X_train, 
                                              (X_train.shape[0],)+input_shape,
                                              change_dims = True)
            print(X_train.shape)
            X_val = sp.feats.adjust_shape(X_val, 
                                            (X_val.shape[0],)+input_shape,
                                              change_dims = True)
            X_test = sp.feats.adjust_shape(X_test, 
                                             (X_test.shape[0],)+input_shape,
                                              change_dims = True)
            
            # randomize train data
            rand_idx = np.random.choice(range(len(X_train)),
                                        len(X_train),
                                        replace=False)
            X_train = X_train[rand_idx]
            
            # make grayscale to colorscale
            X_train = sp.feats.grayscale2color(X_train, colorscale = 3)
            X_val = sp.feats.grayscale2color(X_val, colorscale = 3)
            X_test = sp.feats.grayscale2color(X_test, colorscale = 3)
            
            print(X_train.shape)
            
            history = model.fit(X_train, y_train, 
                                        callbacks = callbacks, 
                                        validation_data = (X_val, y_val),
                                        **kwargs)
            
            score = model.evaluate(X_test, y_test)
         
    
        print('Test loss:', score[0]) 
        print('Test accuracy:', score[1])
        end_session = time.time()
        total_dur_sec_session = round(end_session-start_session,2)
        model_features_dict = dict(model_path = model_path,
                                data_train_path = data_train_path,
                                data_val_path = data_val_path, 
                                data_test_path = data_test_path, 
                                total_dur_sec_session = total_dur_sec_session,
                                use_generator = use_generator,
                                score = score,
                                kwargs = kwargs)
        model_features_dict.update(settings)
        model_features_dict_path = model_dir.joinpath('info_{}_{}.csv'.format(
            model_name, i))
        model_features_dict_path = sp.utils.save_dict(
            filename = model_features_dict_path,
            dict2save = model_features_dict)
        if total_training_sessions is None:
            total_training_sessions = len(train_paths)
        if i == total_training_sessions-1:
            end = time.time()
            total_duration_seconds = round(end-start,2)
            time_dict = dict(total_duration_seconds=total_duration_seconds)
            model_features_dict.update(time_dict)

            model_features_dict_path = model_dir.joinpath('info_{}_{}.csv'.format(
                model_name, i))
            model_features_dict_path = sp.utils.save_dict(
                filename = model_features_dict_path,
                dict2save = model_features_dict,
                overwrite = True)
            print('\nFinished training the model. The model and associated files can be '+\
            'found here: \n{}'.format(model_dir))
            model.save(model_dir.joinpath('final_not_best_model.h5'))
            return model_dir, history

# TODO cleanup
# TODO test
# TODO continue docstrings
def envclassifier_extract_train(
    model_name = 'env_classifier',
    augment_dict = None,
    audiodata_path = None,
    features_dir = None,
    save_new_files_dir = None,
    labeled_data = True,
    ignore_label_marker = None,
    batch_size = 10,
    epochs = 5,
    patience = 15,
    callbacks = None,
    random_seed = None,
    visualize = False,
    vis_every_n_items = 50,
    label_silence = False,
    val_data = None,
    test_data = None,
    append_model_dir = False,
    **kwargs):
    '''Extract and augment features during training of a scene/environment/speech classifier
    
    Parameters
    ----------
    model_name : str 
        Name of the model. No extension (will save as .h5 file) (default 'env_classifier')
        
    augment_dict : dict, optional
        Dictionary containing keys (e.g. 'add_white_noise'). See 
        `soundpy.augment.list_augmentations`and corresponding True or False
        values. If the value is True, the key / augmentation gets implemented
        at random, each epoch.
        (default None)
    
    audiodata_path : str, pathlib.PosixPath
        Where audio data can be found, if no `features_dir` where previously extracted and prepared files are located.
        (default None)
        
    features_dir : str, pathlib.PosixPath
        The feature directory where previously extracted validation and test data 
        are located, as well as the relevant log files.
        
    save_new_files_dir : str, pathlib.PosixPath
        Where new files (logging, model(s), etc.) will be saved. If None, will be 
        set in a unique directory within the current working directory.
        (default None)
        
    labeled_data : bool 
        Useful in determining shape of data. If True, expected label column to exist 
        at the end of the feature column of feature data. Note: this may be removed in 
        future versions. 
        
    ignore_label_marker : str 
        When collecting labels from subdirectory names, this allows a subfolder name to be
        ignored. For example, if `ignore_label_marker` is set as '__', the folder name
        '__test__' will not be included as a label while a folder name 'dog_barking' will.
        
    **kwargs : additional keyword arguments 
        Keyword arguments for `soundpy.feats.get_feats`.
    
    '''
    if features_dir is not None:
        features_dir = sp.utils.string2pathlib(features_dir)
        feat_settings_file = features_dir.joinpath('log_extraction_settings.csv')
        feat_settings_dict = sp.utils.load_dict(feat_settings_file)
        # should be a dict
        feat_kwargs = sp.utils.restore_dictvalue(feat_settings_dict['kwargs'])
        print(feat_kwargs)
        # load decode dictionary for labeled data
        dict_decode_path = features_dir.joinpath('dict_decode.csv')
        dict_decode = sp.utils.load_dict(dict_decode_path)
        dict_encode = None
        # ensure items in dictionaries original type
        for key, value in feat_kwargs.items():
            feat_kwargs[key] = sp.utils.restore_dictvalue(value)
        for key, value in feat_settings_dict.items():
            feat_settings_dict[key] = sp.utils.restore_dictvalue(value)
        for key, value in dict_decode.items():
            # expects key to be integer
            dict_decode[key] = sp.utils.restore_dictvalue(value)
        # update kwargs with loaded feature kwargs
        kwargs = dict(feat_kwargs)
    # require 'feature_type' to be indicated
    if 'feature_type' not in kwargs:
        raise ValueError('Function `envclassifier_extract_train` expects the '+ \
            'parameter `feature_type` to be set as one of the following:\n'+ \
                '- signal\n- stft\n- powspec\n- fbank\n- mfcc\n') 
    
    #if 'stft' not in kwargs['feature_type'] and 'powspec' not in kwargs['feature_type']:
        #raise ValueError('Function `envclassifier_extract_train` can only reliably '+\
            #'work if `feature_type` parameter is set to "stft" or "powspec".'+\
                #' In future versions the other feature types will be made available.')
    
    # ensure defaults are set if not included in kwargs:
    if 'win_size_ms' not in kwargs:
        kwargs['win_size_ms'] = 20
    if 'percent_overlap' not in kwargs:
        kwargs['percent_overlap'] = 0.5
    if 'rate_of_change' not in kwargs:
        kwargs['rate_of_change'] = False
    if 'rate_of_acceleration' not in kwargs:
        kwargs['rate_of_acceleration'] = False
    if 'dur_sec' not in kwargs:
        raise ValueError('Function `envclassifier_extract_train``requires ' +\
            'the keyword argument `dur_sec` to be set. How many seconds of audio '+\
                'from each audio file would you like to use for training?')
    if 'sr' not in kwargs:
        kwargs['sr'] = 22050
    if 'fft_bins' not in kwargs:
        import warnings
        fft_bins = int(kwargs['win_size_ms'] * kwargs['sr'] // 1000)
        msg = '\nWARNING: `fft_bins` was not set. Setting it to {}'.format(fft_bins)
        warnings.warn(msg)
        kwargs['fft_bins'] = fft_bins
    if 'real_signal' not in kwargs:
        kwargs['real_signal'] = True
    if 'window' not in kwargs:
        kwargs['window'] = 'hann'
    if 'zeropad' not in kwargs:
        kwargs['zeropad'] = True
    if 'num_filters' not in kwargs:
        kwargs['num_filters'] = 40
    if 'num_mfcc' not in kwargs:
        kwargs['num_mfcc'] = 40
        
    # training will fail if patience set to a non-integer type
    if patience is None:
        patience = epochs
    
    if features_dir is None:
        # Set up directory to save new files:
        # will not raise error if not exists: instead makes the directory
        if save_new_files_dir is None:
            save_new_files_dir = './example_feats_models/envclassifer/'
        dataset_path = sp.check_dir(save_new_files_dir, make = True)
        # create unique timestamped directory to save new files
        # to avoid overwriting issues:
        dataset_path = dataset_path.joinpath(
            'features_{}_{}'.format(kwargs['feature_type'], sp.utils.get_date()))
        # create that new directory as well
        dataset_path = sp.check_dir(dataset_path, make=True)
    else:
        dataset_path = features_dir
    
    
    # set up datasets if no dataset_dict provided:
    if features_dir is None:
        if audiodata_path is None:
            raise ValueError('Function `envclassifier_extract_train` expects either:\n'+\
                '1) a `dataset_dict` with audiofile pathways assigned to datasets OR'+\
                    '\n2) a `audiodata_path` indicating where audiofiles for'+\
                        'training are located.\n**Both cannot be None.')
        
        # sp.check_dir:
        # raises error if this path doesn't exist (make = False)
        # if does exist, returns path as pathlib.PosixPath object
        data_dir = sp.check_dir(audiodata_path, make = False)
        
        # collect labels
        labels = []
        for label in data_dir.glob('*/'):
            if label.suffix:
                # avoid adding unwanted files in the directory
                # want only directory names
                continue
            if ignore_label_marker is not None:
                if ignore_label_marker in label.stem:
                    continue
            # ignores hidden directories
            if label.stem[0] == '.':
                continue
            labels.append(label.stem)
        labels = set(labels)
    
        # create encoding and decoding dictionaries of labels:
        dict_encode, dict_decode = sp.datasets.create_dicts_labelsencoded(
            labels,
            add_extra_label = label_silence,
            extra_label = 'silence')
    
        # save labels and their encodings
        dict_encode_path = dataset_path.joinpath('dict_encode.csv')
        dict_decode_path = dataset_path.joinpath('dict_decode.csv')
        sp.utils.save_dict(dict2save = dict_encode,
                            filename = dict_encode_path,
                            overwrite=True)
        dict_decode_path = sp.utils.save_dict(dict2save = dict_decode,
                                                filename = dict_decode_path,
                                                overwrite=True)

        # get audio pathways and assign them their encoded labels:
        paths_list = sp.files.collect_audiofiles(data_dir, recursive=True)
        paths_list = sorted(paths_list)

        dict_encodedlabel2audio = sp.datasets.create_encodedlabel2audio_dict(
            dict_encode,
            paths_list)
        # path for saving dict for which audio paths are assigned to which labels:
        dict_encdodedlabel2audio_path = dataset_path.joinpath(
            'dict_encdodedlabel2audio.csv')

        sp.utils.save_dict(dict2save = dict_encodedlabel2audio,
                            filename = dict_encdodedlabel2audio_path,
                            overwrite=True)

        # assign audio files int train, validation, and test datasets
        train, val, test = sp.datasets.audio2datasets(
            dict_encdodedlabel2audio_path,
            perc_train=0.8,
            limit=None,
            seed=random_seed)
        
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(train)
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(val)
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(test)

        # save audiofiles for each dataset to dict and save
        # for logging purposes
        dataset_dict = dict([('train', train),
                                ('val', val),
                                ('test', test)])
        dataset_dict_path = dataset_path.joinpath('dataset_audiofiles.csv')
        dataset_dict_path = sp.utils.save_dict(
            dict2save = dataset_dict,
            filename = dataset_dict_path,
            overwrite=True)
        feat_base_shape, shape_with_label = sp.feats.get_feature_matrix_shape(
            labeled_data = labeled_data,
            **kwargs)
        extracted_data_dict = dict([('val',dataset_dict['val']),
                        ('test',dataset_dict['test'])])
        val_path = dataset_path.joinpath('val_data.npy')
        test_path = dataset_path.joinpath('test_data.npy')
        extracted_data_path_dict = dict([('val', val_path),
                            ('test', test_path)])
        # extract test data 
        print('\nExtracting validation data for use in training:')
        extracted_data_dict, extracted_data_path_dict = sp.feats.save_features_datasets(
            extracted_data_dict,
            extracted_data_path_dict,
            labeled_data = labeled_data,
            **kwargs)

        val_data = np.load(extracted_data_path_dict['val'])
        test_data = np.load(extracted_data_path_dict['test'])
    else:
        feat_base_shape = feat_settings_dict['feat_base_shape']
        shape_with_label = feat_settings_dict['feat_model_shape']
        # use pre-collected dataset dict
        dataset_dict_path = dataset_path.joinpath('dataset_audiofiles.csv')
        dataset_dict = sp.utils.load_dict(dataset_dict_path)
        for key, value in dataset_dict.items():
            dataset_dict[key] = sp.utils.restore_dictvalue(value)
        val_data = np.load(val_data)
        test_data = np.load(test_data)
        

    if 'fbank' in kwargs['feature_type'] or 'mfcc' in kwargs['feature_type']:
        kwargs['fmax'] = kwargs['sr'] / 2.0 # Niquist theorem
    # extract validation data (must already be extracted)
    color_dimension = (1,) # our data is in grayscale
    input_shape = feat_base_shape + color_dimension
    num_labels = len(dict_decode)
    # otherwise should arleady be specified

    if augment_dict is None:
        augment_dict = dict()


    # designate where to save model and related files
    model_name += '_' + kwargs['feature_type']
    model_dir = dataset_path.joinpath(model_name)
    model_dir = sp.utils.check_dir(model_dir, make=True, append=append_model_dir) # don't want to overwrite already trained model and logs
    model_path = model_dir.joinpath(model_name)
    
    # setup model 
    envclassifier, settings_dict = spdl.cnn_classifier(
        input_shape = input_shape,
        num_labels = num_labels)
    optimizer = 'adam'
    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
    envclassifier.compile(optimizer = optimizer,
                            loss = loss,
                            metrics = metrics)

    # should randomly apply augmentations in generator

    # items that need to be called with each iteration:
    # save best model for each iteration - don't want to be overwritten
    # with worse model
    best_modelname = str(model_path) + '.h5'
    callbacks = spdl.setup_callbacks(
        patience = patience,
        best_modelname = best_modelname, 
        log_filename = model_dir.joinpath('log.csv'),
        append = True)

    normalize = True
    tensor = (1,)
    train_generator = spdl.GeneratorFeatExtraction(
        datalist = dataset_dict['train'],
        model_name = model_name,
        normalize = normalize,
        apply_log = False,
        randomize = True, # want the data order to be different for each iteration 
        random_seed = None,
        desired_input_shape = tensor + input_shape,
        batch_size = batch_size, 
        gray2color = False,
        visualize = visualize,
        vis_every_n_items = vis_every_n_items,
        visuals_dir = model_dir.joinpath('images'),
        decode_dict = dict_decode,
        dataset = 'train',
        augment_dict = augment_dict,
        label_silence = label_silence,
        **kwargs)
    
    val_generator = spdl.Generator(
        data_matrix1 = val_data,
        desired_input_shape = tensor + input_shape)
    
    test_generator = spdl.Generator(
        data_matrix1 = test_data,
        desired_input_shape = tensor + input_shape)
    

    if 'stft' in kwargs['feature_type'] or 'fbank' in kwargs['feature_type'] \
        or 'powspec' in kwargs['feature_type']:
            energy_scale = 'power_to_db'
    else:
        energy_scale = None
    
    feats_train, label_train = next(train_generator.generator())

    try:
        label_train_vis = dict_decode[label_train[0]]
    except KeyError:
        label_train_vis = dict_decode[str(int(label_train[0]))]

    feats_vis = feats_train.reshape((feats_train.shape[1],feats_train.shape[2]))
    sp.feats.plot(feature_matrix = feats_vis, feature_type=kwargs['feature_type'],
                  title='Train: {} features label "{}"'.format(kwargs['feature_type'], 
                                                      label_train_vis),
                        name4pic='train_feats{}.png'.format(sp.utils.get_date()),
                        subprocess=True,
                        energy_scale = energy_scale)
    
    feats_val, label_val = next(val_generator.generator())

    try:
        label_val_vis = dict_decode[label_val[0]]
    except KeyError:
        label_val_vis = dict_decode[str(int(label_val[0]))]

    feats_vis = feats_val.reshape((feats_val.shape[1],feats_val.shape[2]))
    sp.feats.plot(feature_matrix = feats_vis, feature_type=kwargs['feature_type'],
                  title='Val: {} features label "{}"'.format(kwargs['feature_type'], 
                                                      label_val_vis),
                        name4pic='val_feats{}.png'.format(sp.utils.get_date()),
                        subprocess=True,
                        energy_scale = energy_scale)
    
    feats_test, label_test = next(test_generator.generator())
    try:
        label_test_vis = dict_decode[label_test[0]]
    except KeyError:
        label_test_vis = dict_decode[str(int(label_test[0]))]

    feats_vis = feats_test.reshape((feats_test.shape[1],feats_test.shape[2]))
    sp.feats.plot(feature_matrix = feats_vis, feature_type=kwargs['feature_type'],
                  title='Test: {} features label "{}"'.format(kwargs['feature_type'], 
                                                      label_test_vis),
                        name4pic='test_feats{}.png'.format(sp.utils.get_date()),
                        subprocess=True,
                        energy_scale = energy_scale)

    ds_train = tf.data.Dataset.from_generator(
        spdl.make_gen_callable(train_generator.generator()),
        output_types=(feats_train.dtype, label_train.dtype), 
        output_shapes=(feats_train.shape, 
                        label_train.shape))
    ds_val = tf.data.Dataset.from_generator(
        spdl.make_gen_callable(val_generator.generator()),
        output_types=(feats_val.dtype, label_val.dtype), 
        output_shapes=(feats_val.shape, 
                        label_val.shape))
    ds_test = tf.data.Dataset.from_generator(
        spdl.make_gen_callable(test_generator.generator()),
        output_types=(feats_test.dtype, label_test.dtype), 
        output_shapes=(feats_test.shape, 
                        label_test.shape))
        
    print('\nShapes of X and y data from the train, val, and test generators:')
    print(ds_train)
    print(ds_val)
    print(ds_test)
    print()
    
    print('-'*79)
    if augment_dict:
        print('\nAugmentation(s) applied (at random): \n')
        for key, value in augment_dict.items():
            if value == True:
                print('{}'.format(key).upper())
                try:
                    settings = augment_dict['augment_settings_dict'][key]
                    print('- Settings: {}'.format(settings))
                except KeyError:
                    pass
        print()
    else:
        print('\nNo augmentations applied.\n')
    print('-'*79)
    
    # start training
    start = time.time()
    history = envclassifier.fit(
        ds_train,
        steps_per_epoch = len(dataset_dict['train']),
        callbacks = callbacks,
        epochs = epochs,
        validation_data = ds_val,
        validation_steps = val_data.shape[0]
        )

    model_features_dict = dict(model_path = model_path,
                            dataset_dict = dataset_dict,
                            augment_dict = augment_dict)
    model_features_dict.update(settings_dict)
    model_features_dict.update(augment_dict)
    end = time.time()
    total_duration_seconds = round(end-start,2)
    time_dict = dict(total_duration_seconds=total_duration_seconds)
    model_features_dict.update(time_dict)

    model_features_dict_path = model_dir.joinpath('info_{}.csv'.format(
        model_name))
    model_features_dict_path = sp.utils.save_dict(
        filename = model_features_dict_path,
        dict2save = model_features_dict)
    print('\nFinished training the model. The model and associated files can be '+\
        'found here: \n{}'.format(model_dir))

    
    score = envclassifier.evaluate(ds_test, steps=1000) 
    print('Test loss:', score[0]) 
    print('Test accuracy:', score[1])
       
    finished_time = time.time()
    total_total_duration = finished_time - start
    time_new_units, units = sp.utils.adjust_time_units(total_total_duration)
    print('\nEntire program took {} {}.\n\n'.format(time_new_units, units))
    print('-'*79)
    
    return model_dir, history    

def cnnlstm_extract_train(
    model_name = 'cnnlstm_classifier',
    dataset_dict = None,
    num_labels = None,
    augment_dict = None,
    audiodata_path = None,
    save_new_files_dir = None,
    labeled_data = True,
    ignore_label_marker = None,
    context_window = 5,
    batch_size = 10,
    epochs = 5,
    patience = 15,
    callbacks = None,
    random_seed = None,
    visualize = False,
    vis_every_n_items = 50,
    label_silence = False,
    **kwargs):
    '''Extract and augment features during training of a scene/environment/speech classifier
    
    Parameters
    ----------
    model_name : str 
        Name of the model. No extension (will save as .h5 file)
        
    dataset_dict : dict, optional
        A dictionary including datasets as keys, and audio file lists (with or without
        labels) as values. If None, will be created based on `audiodata_path`.
        (default None)
        
    augment_dict : dict, optional
        Dictionary containing keys (e.g. 'add_white_noise'). See 
        `soundpy.augment.list_augmentations`and corresponding True or False
        values. If the value is True, the key / augmentation gets implemented
        at random, each epoch.
        (default None)
    
    audiodata_path : str, pathlib.PosixPath
        Where audio data can be found, if no `dataset_dict` provided.
        (default None)
        
    save_new_files_dir : str, pathlib.PosixPath
        Where new files (logging, model(s), etc.) will be saved. If None, will be 
        set in a unique directory within the current working directory.
        (default None)
        
    **kwargs : additional keyword arguments 
        Keyword arguments for `soundpy.feats.get_feats`.
    
    '''
    # require 'feature_type' to be indicated
    if 'feature_type' not in kwargs:
        raise ValueError('Function `envclassifier_extract_train` expects the '+ \
            'parameter `feature_type` to be set as one of the following:\n'+ \
                '- signal\n- stft\n- powspec\n- fbank\n- mfcc\n') 
    
    #if 'stft' not in kwargs['feature_type'] and 'powspec' not in kwargs['feature_type']:
        #raise ValueError('Function `envclassifier_extract_train` can only reliably '+\
            #'work if `feature_type` parameter is set to "stft" or "powspec".'+\
                #' In future versions the other feature types will be made available.')
    
    # ensure defaults are set if not included in kwargs:
    if 'win_size_ms' not in kwargs:
        kwargs['win_size_ms'] = 20
    if 'percent_overlap' not in kwargs:
        kwargs['percent_overlap'] = 0.5
    if 'rate_of_change' not in kwargs:
        kwargs['rate_of_change'] = False
    if 'rate_of_acceleration' not in kwargs:
        kwargs['rate_of_acceleration'] = False
    if 'dur_sec' not in kwargs:
        raise ValueError('Function `envclassifier_extract_train``requires ' +\
            'the keyword argument `dur_sec` to be set. How many seconds of audio '+\
                'from each audio file would you like to use for training?')
    if 'sr' not in kwargs:
        kwargs['sr'] = 22050
    if 'fft_bins' not in kwargs:
        import warnings
        fft_bins = int(kwargs['win_size_ms'] * kwargs['sr'] // 1000)
        msg = '\nWARNING: `fft_bins` was not set. Setting it to {}'.format(fft_bins)
        warnings.warn(msg)
        kwargs['fft_bins'] = fft_bins
    if 'real_signal' not in kwargs:
        kwargs['real_signal'] = True
    if 'window' not in kwargs:
        kwargs['window'] = 'hann'
    if 'zeropad' not in kwargs:
        kwargs['zeropad'] = True
    if 'num_filters' not in kwargs:
        kwargs['num_filters'] = 40
    if 'num_mfcc' not in kwargs:
        kwargs['num_mfcc'] = 40
        
    # training will fail if patience set to a non-integer type
    if patience is None:
        patience = epochs
    
    # Set up directory to save new files:
    # will not raise error if not exists: instead makes the directory
    if save_new_files_dir is None:
        save_new_files_dir = './example_feats_models/envclassifer/'
    dataset_path = sp.check_dir(save_new_files_dir, make = True)
    # create unique timestamped directory to save new files
    # to avoid overwriting issues:
    dataset_path = dataset_path.joinpath(
        'features_{}_{}'.format(kwargs['feature_type'], sp.utils.get_date()))
    # create that new directory as well
    dataset_path = sp.check_dir(dataset_path, make=True)
    
    # set up datasets if no dataset_dict provided:
    if dataset_dict is None:
        if audiodata_path is None:
            raise ValueError('Function `cnnlstm_extract_train` expects either:\n'+\
                '1) a `dataset_dict` with audiofile pathways assigned to datasets OR'+\
                    '\n2) a `audiodata_path` indicating where audiofiles for'+\
                        'training are located.\n**Both cannot be None.')
        
        # sp.check_dir:
        # raises error if this path doesn't exist (make = False)
        # if does exist, returns path as pathlib.PosixPath object
        data_dir = sp.check_dir(audiodata_path, make = False)
        
        # collect labels
        labels = []
        for label in data_dir.glob('*/'):
            if label.suffix:
                # avoid adding unwanted files in the directory
                # want only directory names
                continue
            if ignore_label_marker is not None:
                if ignore_label_marker in label.stem:
                    continue
            # ignores hidden directories
            if label.stem[0] == '.':
                continue
            labels.append(label.stem)
        labels = set(labels)
    
        # create encoding and decoding dictionaries of labels:
        dict_encode, dict_decode = sp.datasets.create_dicts_labelsencoded(
            labels,
            add_extra_label = label_silence,
            extra_label = 'silence')
    
        # save labels and their encodings
        dict_encode_path = dataset_path.joinpath('dict_encode.csv')
        dict_decode_path = dataset_path.joinpath('dict_decode.csv')
        sp.utils.save_dict(dict2save = dict_encode,
                            filename = dict_encode_path,
                            overwrite=True)
        dict_decode_path = sp.utils.save_dict(dict2save = dict_decode,
                                                filename = dict_decode_path,
                                                overwrite=True)

        # get audio pathways and assign them their encoded labels:
        paths_list = sp.files.collect_audiofiles(data_dir, recursive=True)
        paths_list = sorted(paths_list)

        dict_encodedlabel2audio = sp.datasets.create_encodedlabel2audio_dict(
            dict_encode,
            paths_list)
        # path for saving dict for which audio paths are assigned to which labels:
        dict_encdodedlabel2audio_path = dataset_path.joinpath(
            'dict_encdodedlabel2audio.csv')

        sp.utils.save_dict(dict2save = dict_encodedlabel2audio,
                            filename = dict_encdodedlabel2audio_path,
                            overwrite=True)

        # assign audio files int train, validation, and test datasets
        train, val, test = sp.datasets.audio2datasets(
            dict_encdodedlabel2audio_path,
            perc_train=0.8,
            limit=None,
            seed=random_seed)
        
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(train)
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(val)
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(test)

        # save audiofiles for each dataset to dict and save
        # for logging purposes
        dataset_dict = dict([('train', train),
                                ('val', val),
                                ('test', test)])
        dataset_dict_path = dataset_path.joinpath('dataset_audiofiles.csv')
        dataset_dict_path = sp.utils.save_dict(
            dict2save = dataset_dict,
            filename = dataset_dict_path,
            overwrite=True)
        
    else:
        if num_labels is None:
            raise ValueError('Function `cnnlstm_extract_train` requires '+\
                '`num_labels` to be provided if a pre-made `dataset_dict` is provided.')
        # use pre-collected dataset dict
        dataset_dict = sp.utils.load_dict(dataset_dict)
        # don't have the label data available
        dict_encode, dict_decode = None, None
        
    feat_base_shape, shape_with_label = sp.feats.get_feature_matrix_shape(
        labeled_data = labeled_data,
        **kwargs)
    
    color_dimension = (1,) # our data is in grayscale
    if context_window:
        feat_base_shape = sp.feats.featshape_new_subframe(feat_base_shape,
                                                          context_window,
                                                          zeropad=True,
                                                          axis=0,
                                                          include_dim_size_1=True)

    input_shape = feat_base_shape + color_dimension

    if 'fbank' in kwargs['feature_type'] or 'mfcc' in kwargs['feature_type']:
        kwargs['fmax'] = kwargs['sr'] / 2.0 # Niquist theorem
    # extract validation data (must already be extracted)
    extracted_data_dict = dict([('val',dataset_dict['val']),
                     ('test',dataset_dict['test'])])
    val_path = dataset_path.joinpath('val_data.npy')
    test_path = dataset_path.joinpath('test_data.npy')
    extracted_data_path_dict = dict([('val', val_path),
                          ('test', test_path)])
    # extract test data 
    print('\nExtracting validation data for use in training:')
    extracted_data_dict, extracted_data_path_dict = sp.feats.save_features_datasets(
        extracted_data_dict,
        extracted_data_path_dict,
        labeled_data = labeled_data,
        **kwargs)

    val_data = np.load(extracted_data_path_dict['val'])
    test_data = np.load(extracted_data_path_dict['test'])


    # start training
    start = time.time()

    if dict_encode is not None:
        num_labels = len(dict_encode) 
    # otherwise should arleady be specified

    if augment_dict is None:
        augment_dict = dict()


    # designate where to save model and related files
    model_name = 'audioaugment_' + kwargs['feature_type']
    model_dir = dataset_path.joinpath(model_name)
    model_dir = sp.utils.check_dir(model_dir, make=True)
    model_path = model_dir.joinpath(model_name)
    
    # setup model 
    envclassifier, settings_dict = spdl.cnnlstm_classifier(
        input_shape = input_shape,
        num_labels = num_labels,
        lstm_cells = 40) # need to fix for other kinds of features
    optimizer = 'adam'
    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
    envclassifier.compile(optimizer = optimizer,
                            loss = loss,
                            metrics = metrics)

    # should randomly apply augmentations in generator

    # items that need to be called with each iteration:
    # save best model for each iteration - don't want to be overwritten
    # with worse model
    best_modelname = str(model_path) + '.h5'
    callbacks = spdl.setup_callbacks(
        patience = patience,
        best_modelname = best_modelname, 
        log_filename = model_dir.joinpath('log.csv'),
        append = True)

    normalize = True
    tensor = (1,)
    train_generator = spdl.GeneratorFeatExtraction(
        datalist = dataset_dict['train'],
        model_name = model_name,
        normalize = normalize,
        apply_log = False,
        randomize = True, # want the data order to be different for each iteration 
        random_seed = None,
        desired_input_shape = tensor + input_shape,
        batch_size = batch_size, 
        gray2color = False,
        visualize = visualize,
        vis_every_n_items = vis_every_n_items,
        visuals_dir = model_dir.joinpath('images'),
        decode_dict = dict_decode,
        dataset = 'train',
        augment_dict = augment_dict,
        label_silence = label_silence,
        context_window = context_window,
        **kwargs)
    
    val_generator = spdl.Generator(
        data_matrix1 = val_data,
        desired_input_shape = tensor + input_shape,
        context_window = context_window)
    
    test_generator = spdl.Generator(
        data_matrix1 = test_data,
        desired_input_shape = tensor + input_shape,
        context_window = context_window)
    

    if 'stft' in kwargs['feature_type'] or 'fbank' in kwargs['feature_type'] \
        or 'powspec' in kwargs['feature_type']:
            energy_scale = 'power_to_db'
    else:
        energy_scale = None
    
    feats_train, label_train = next(train_generator.generator())

    #feats_vis = feats_train.reshape((feats_train.shape[1],feats_train.shape[2]))
    #sp.feats.plot(feature_matrix = feats_vis, feature_type=kwargs['feature_type'],
                  #title='Train: {} features label "{}"'.format(kwargs['feature_type'], 
                                                      #dict_decode[label_train[0]]),
                        #name4pic='train_feats{}.png'.format(sp.utils.get_date()),
                        #subprocess=True,
                        #energy_scale = energy_scale)
    
    feats_val, label_val = next(val_generator.generator())

    #feats_vis = feats_val.reshape((feats_val.shape[1],feats_val.shape[2]))
    #sp.feats.plot(feature_matrix = feats_vis, feature_type=kwargs['feature_type'],
                  #title='Val: {} features label "{}"'.format(kwargs['feature_type'], 
                                                      #dict_decode[label_val[0]]),
                        #name4pic='val_feats{}.png'.format(sp.utils.get_date()),
                        #subprocess=True,
                        #energy_scale = energy_scale)
    
    feats_test, label_test = next(test_generator.generator())

    #feats_vis = feats_test.reshape((feats_test.shape[1],feats_test.shape[2]))
    #sp.feats.plot(feature_matrix = feats_vis, feature_type=kwargs['feature_type'],
                  #title='Test: {} features label "{}"'.format(kwargs['feature_type'], 
                                                      #dict_decode[label_test[0]]),
                        #name4pic='test_feats{}.png'.format(sp.utils.get_date()),
                        #subprocess=True,
                        #energy_scale = energy_scale)

    ds_train = tf.data.Dataset.from_generator(
        spdl.make_gen_callable(train_generator.generator()),
        output_types=(feats_train.dtype, label_train.dtype), 
        output_shapes=(feats_train.shape, 
                        label_train.shape))
    ds_val = tf.data.Dataset.from_generator(
        spdl.make_gen_callable(val_generator.generator()),
        output_types=(feats_val.dtype, label_val.dtype), 
        output_shapes=(feats_val.shape, 
                        label_val.shape))
    ds_test = tf.data.Dataset.from_generator(
        spdl.make_gen_callable(test_generator.generator()),
        output_types=(feats_test.dtype, label_test.dtype), 
        output_shapes=(feats_test.shape, 
                        label_test.shape))
        
    print('\nShapes of X and y data from the train, val, and test generators:')
    print(ds_train)
    print(ds_val)
    print(ds_test)
    print()
    
    print('-'*79)
    if augment_dict:
        print('\nAugmentation(s) applied (at random): \n')
        for key, value in augment_dict.items():
            if value == True:
                print('{}'.format(key).upper())
                try:
                    settings = augment_dict['augment_settings_dict'][key]
                    print('- Settings: {}'.format(settings))
                except KeyError:
                    pass
        print()
    else:
        print('\nNo augmentations applied.\n')
    print('-'*79)
    
    history = envclassifier.fit(
        ds_train,
        steps_per_epoch = len(dataset_dict['train']),
        callbacks = callbacks,
        epochs = epochs,
        validation_data = ds_val,
        validation_steps = val_data.shape[0]
        )

    model_features_dict = dict(model_path = model_path,
                            dataset_dict = dataset_dict,
                            augment_dict = augment_dict)
    model_features_dict.update(settings_dict)
    model_features_dict.update(augment_dict)
    end = time.time()
    total_duration_seconds = round(end-start,2)
    time_dict = dict(total_duration_seconds=total_duration_seconds)
    model_features_dict.update(time_dict)

    model_features_dict_path = model_dir.joinpath('info_{}.csv'.format(
        model_name))
    model_features_dict_path = sp.utils.save_dict(
        filename = model_features_dict_path,
        dict2save = model_features_dict)
    print('\nFinished training the model. The model and associated files can be '+\
        'found here: \n{}'.format(model_dir))

    
    score = envclassifier.evaluate(ds_test, steps=1000) 
    print('Test loss:', score[0]) 
    print('Test accuracy:', score[1])
       
    finished_time = time.time()
    total_total_duration = finished_time - start
    time_new_units, units = sp.utils.adjust_time_units(total_total_duration)
    print('\nEntire program took {} {}.\n\n'.format(time_new_units, units))
    print('-'*79)
    
    return model_dir, history    


# TODO cleanup
# TODO test
# TODO continue docstrings
def denoiser_extract_train(
    model_name = 'denoiser',
    augment_dict = None,
    audiodata_clean_path = None,
    audiodata_noisy_path = None,
    features_dir = None,
    save_new_files_dir = None,
    labeled_data = False,
    ignore_label_marker = None,
    batch_size = 10,
    epochs = 5,
    patience = 15,
    callbacks = None,
    random_seed = 20,
    visualize = False,
    vis_every_n_items = 50,
    label_silence = False,
    val_data = None,
    test_data = None,
    append_model_dir = False,
    **kwargs):
    '''Extract and augment features during training of a scene/environment/speech classifier
    
    Parameters
    ----------
    model_name : str 
        Name of the model. No extension (will save as .h5 file) (default 'env_classifier')
        
    augment_dict : dict, optional
        Dictionary containing keys (e.g. 'add_white_noise'). See 
        `soundpy.augment.list_augmentations`and corresponding True or False
        values. If the value is True, the key / augmentation gets implemented
        at random, each epoch.
        (default None)
    
    audiodata_path : str, pathlib.PosixPath
        Where audio data can be found, if no `features_dir` where previously extracted and prepared files are located.
        (default None)
        
    features_dir : str, pathlib.PosixPath
        The feature directory where previously extracted validation and test data 
        are located, as well as the relevant log files.
        
    save_new_files_dir : str, pathlib.PosixPath
        Where new files (logging, model(s), etc.) will be saved. If None, will be 
        set in a unique directory within the current working directory.
        (default None)
        
    labeled_data : bool 
        Useful in determining shape of data. If True, expected label column to exist 
        at the end of the feature column of feature data. Note: this may be removed in 
        future versions. 
        
    ignore_label_marker : str 
        When collecting labels from subdirectory names, this allows a subfolder name to be
        ignored. For example, if `ignore_label_marker` is set as '__', the folder name
        '__test__' will not be included as a label while a folder name 'dog_barking' will.
        
    **kwargs : additional keyword arguments 
        Keyword arguments for `soundpy.feats.get_feats`.
    
    '''

    if features_dir is not None:
        features_dir = sp.utils.string2pathlib(features_dir)
        feat_settings_file = features_dir.joinpath('log_extraction_settings.csv')
        feat_settings_dict = sp.utils.load_dict(feat_settings_file)
        # should be a dict
        feat_kwargs = sp.utils.restore_dictvalue(feat_settings_dict['kwargs'])
        print(feat_kwargs)
        # load decode dictionary for labeled data
        dict_decode_path = features_dir.joinpath('dict_decode.csv')
        dict_decode = sp.utils.load_dict(dict_decode_path)
        dict_encode = None
        # ensure items in dictionaries original type
        for key, value in feat_kwargs.items():
            feat_kwargs[key] = sp.utils.restore_dictvalue(value)
        for key, value in feat_settings_dict.items():
            feat_settings_dict[key] = sp.utils.restore_dictvalue(value)
        for key, value in dict_decode.items():
            # expects key to be integer
            dict_decode[key] = sp.utils.restore_dictvalue(value)
        # update kwargs with loaded feature kwargs
        kwargs = dict(feat_kwargs)
    # require 'feature_type' to be indicated
    if 'feature_type' not in kwargs:
        raise ValueError('Function `denoiser_extract_train` expects the '+ \
            'parameter `feature_type` to be set as one of the following:\n'+ \
                '- signal\n- stft\n- powspec\n- fbank\n- mfcc\n') 
    
    #if 'stft' not in kwargs['feature_type'] and 'powspec' not in kwargs['feature_type']:
        #raise ValueError('Function `denoiser_extract_train` can only reliably '+\
            #'work if `feature_type` parameter is set to "stft" or "powspec".'+\
                #' In future versions the other feature types will be made available.')
    
    # ensure defaults are set if not included in kwargs:
    if 'win_size_ms' not in kwargs:
        kwargs['win_size_ms'] = 20
    if 'percent_overlap' not in kwargs:
        kwargs['percent_overlap'] = 0.5
    if 'rate_of_change' not in kwargs:
        kwargs['rate_of_change'] = False
    if 'rate_of_acceleration' not in kwargs:
        kwargs['rate_of_acceleration'] = False
    if 'dur_sec' not in kwargs:
        raise ValueError('Function `denoiser_extract_train``requires ' +\
            'the keyword argument `dur_sec` to be set. How many seconds of audio '+\
                'from each audio file would you like to use for training?')
    if 'sr' not in kwargs:
        kwargs['sr'] = 22050
    if 'fft_bins' not in kwargs:
        import warnings
        fft_bins = int(kwargs['win_size_ms'] * kwargs['sr'] // 1000)
        msg = '\nWARNING: `fft_bins` was not set. Setting it to {}'.format(fft_bins)
        warnings.warn(msg)
        kwargs['fft_bins'] = fft_bins
    if 'real_signal' not in kwargs:
        kwargs['real_signal'] = True
    if 'window' not in kwargs:
        kwargs['window'] = 'hann'
    if 'zeropad' not in kwargs:
        kwargs['zeropad'] = True
    if 'num_filters' not in kwargs:
        kwargs['num_filters'] = 40
    if 'num_mfcc' not in kwargs:
        kwargs['num_mfcc'] = 40
        
    # training will fail if patience set to a non-integer type
    if patience is None:
        patience = epochs
    
    if features_dir is None:
        # Set up directory to save new files:
        # will not raise error if not exists: instead makes the directory
        if save_new_files_dir is None:
            save_new_files_dir = './example_feats_models/denoiser/'
        dataset_path = sp.check_dir(save_new_files_dir, make = True)
        # create unique timestamped directory to save new files
        # to avoid overwriting issues:
        dataset_path = dataset_path.joinpath(
            'features_{}_{}'.format(kwargs['feature_type'], sp.utils.get_date()))
        # create that new directory as well
        dataset_path = sp.check_dir(dataset_path, make=True)
    else:
        dataset_path = features_dir

    # designate where to save model and related files
    model_name += '_' + kwargs['feature_type']
    model_dir = dataset_path.joinpath(model_name)
    model_dir = sp.utils.check_dir(model_dir, make=True,
                                   append=append_model_dir) # don't want to overwrite already trained model and logs
    model_path = model_dir.joinpath(model_name+'.h5')
    
    

    if features_dir is None:
        if audiodata_clean_path is None:
            raise ValueError('Function `denoiser_extract_train` expects either:\n'+\
                '1) a `dataset_dict` with audiofile pathways assigned to datasets OR'+\
                    '\n2) `audiodata_clean_path` and `audiodata_noisy_path` indicating where audiofiles for'+\
                        'training are located.\n**Both cannot be None.')
        
        # sp.check_dir:
        # raises error if this path doesn't exist (make = False)
        # if does exist, returns path as pathlib.PosixPath object
        data_clean_dir = sp.check_dir(audiodata_clean_path, make = False)
        data_noisy_dir = sp.check_dir(audiodata_noisy_path, make = False)

        paths_list_clean = sp.files.collect_audiofiles(data_clean_dir,
                                                       recursive=False)
        paths_list_clean = sorted(paths_list_clean)
        paths_list_noisy = sp.files.collect_audiofiles(data_noisy_dir,
                                                       recursive=False)
        paths_list_noisy = sorted(paths_list_noisy)
    
        # for now not using any test data: too small a dataset
        # can test from greater dataset
        train_clean, test_clean, __ = sp.datasets.waves2dataset(
            audiolist = paths_list_clean, 
            perc_train=1, 
            seed=40, 
            train=True, 
            val=False, 
            test=False)
        train_noisy, test_noisy, __ = sp.datasets.waves2dataset(
            audiolist = paths_list_noisy, 
            perc_train=1, 
            seed=40, 
            train=True, 
            val=False, 
            test=False)

        # save filenames not used in training
        #doc_dir = model_path.parent
        #sp.utils.save_dict(doc_dir.joinpath('test_noisy_files.csv'), 
                           #dict(test_noisy = test_noisy))
        #sp.utils.save_dict(doc_dir.joinpath('test_clean_files.csv'), 
                           #dict(test_clean = test_clean))
        
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(train_clean)
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(train_noisy)
        
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(test_clean)
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(test_noisy)

        for i in range(10):
            try:
                print(train_clean[i])
                print()
            except IndexError:
                pass
            try:
                print(train_noisy[i])
                print()
                print()
            except IndexError:
                pass
            try:
                print(test_clean[i])
            except IndexError:
                pass
            try:
                print(test_noisy[i])
            except IndexError:
                pass

        labeled_data = False
        feat_base_shape, shape_for_model = sp.feats.get_feature_matrix_shape(
            labeled_data = labeled_data,
            **kwargs)

    if 'fbank' in kwargs['feature_type'] or 'mfcc' in kwargs['feature_type']:
        kwargs['fmax'] = kwargs['sr'] / 2.0 # Niquist theorem
    # extract validation data (must already be extracted)
    color_dimension = (1,) # our data is in grayscale
    input_shape = feat_base_shape + color_dimension

    if augment_dict is None:
        augment_dict = dict()



    
    # setup model 
    denoiser, settings_dict = spdl.autoencoder_denoise(
        input_shape = input_shape)
    adm = tf.keras.optimizers.Adam(learning_rate=0.0001)
    denoiser.compile(optimizer=adm, loss='binary_crossentropy')

    # should randomly apply augmentations in generator

    # items that need to be called with each iteration:
    # save best model for each iteration - don't want to be overwritten
    # with worse model
    best_modelname = str(model_path) + '.h5'
    callbacks = spdl.setup_callbacks(
        patience = patience,
        early_stop = False, # don't have validation data
        save_bestmodel = False,
        best_modelname = best_modelname, # won't be used (no validation data) 
        log_filename = model_dir.joinpath('log.csv'),
        append = True)

    normalize = True
    tensor = (1,)
    train_generator = spdl.GeneratorFeatExtraction(
        datalist = train_noisy,
        datalist2 = train_clean,
        model_name = model_name,
        normalize = normalize,
        apply_log = False,
        randomize = True, # want the data order to be different for each iteration 
        random_seed = 50,
        desired_input_shape = tensor + input_shape,
        batch_size = batch_size, 
        gray2color = False,
        visualize = visualize,
        vis_every_n_items = vis_every_n_items,
        visuals_dir = model_dir.joinpath('images'),
        decode_dict = None,
        dataset = 'train',
        augment_dict = augment_dict,
        label_silence = label_silence,
        **kwargs)
    


    if 'stft' in kwargs['feature_type'] or 'fbank' in kwargs['feature_type'] \
        or 'powspec' in kwargs['feature_type']:
            energy_scale = 'power_to_db'
    else:
        energy_scale = None
    
    feats_noisy, feats_clean = next(train_generator.generator())

    # visualize the features
    feats_vis_noisy = feats_noisy.reshape((feats_noisy.shape[1],feats_noisy.shape[2]))
    sp.feats.plot(feature_matrix = feats_vis_noisy, 
                  feature_type=kwargs['feature_type'],
                  title='Train: {} features label "{}"'.format(kwargs['feature_type'], 
                                                      'noisy'),
                        name4pic='feats_noisy{}.png'.format(sp.utils.get_date()),
                        subprocess=True,
                        energy_scale = energy_scale)
                  
    feats_vis_clean = feats_clean.reshape((feats_clean.shape[1],feats_clean.shape[2]))
    sp.feats.plot(feature_matrix = feats_vis_clean, 
                  feature_type=kwargs['feature_type'],
                  title='Train: {} features label "{}"'.format(kwargs['feature_type'], 
                                                      'clean'),
                  name4pic='feats_clean{}.png'.format(sp.utils.get_date()),
                        
                        subprocess=True,
                        energy_scale = energy_scale)


    ds_train = tf.data.Dataset.from_generator(
        spdl.make_gen_callable(train_generator.generator()),
        output_types=(feats_noisy.dtype, feats_clean.dtype), 
        output_shapes=(feats_noisy.shape, 
                        feats_clean.shape))

    print('\nShapes of X and y data from the train generator:')
    print(ds_train)
    
    print('-'*79)
    if augment_dict:
        print('\nAugmentation(s) applied (at random): \n')
        for key, value in augment_dict.items():
            if value == True:
                print('{}'.format(key).upper())
                try:
                    settings = augment_dict['augment_settings_dict'][key]
                    print('- Settings: {}'.format(settings))
                except KeyError:
                    pass
        print()
    else:
        print('\nNo augmentations applied.\n')
    print('-'*79)
    
    # start training
    start = time.time()
    history = denoiser.fit(
        ds_train,
        steps_per_epoch = len(train_noisy),
        callbacks = callbacks,
        epochs = epochs)

    denoiser.save(model_path)

    # save this info for when implementing model
    kwargs['input_shape'] = input_shape
    sp.utils.save_dict(model_dir.joinpath('log_extraction_settings.csv'), kwargs)
    model_features_dict = dict(model_path = model_path,
                            augment_dict = augment_dict)
    model_features_dict.update(settings_dict)
    model_features_dict.update(augment_dict)
    #model_features_dict.update(kwargs)
    end = time.time()
    total_duration_seconds = round(end-start,2)
    time_dict = dict(total_duration_seconds = total_duration_seconds)
    model_features_dict.update(time_dict)

    model_features_dict_path = model_dir.joinpath('info_{}.csv'.format(
        model_name))
    model_features_dict_path = sp.utils.save_dict(
        filename = model_features_dict_path,
        dict2save = model_features_dict)
    print('\nFinished training the model. The model and associated files can be '+\
        'found here: \n{}'.format(model_dir))

       
    finished_time = time.time()
    total_total_duration = finished_time - start
    time_new_units, units = sp.utils.adjust_time_units(total_total_duration)
    print('\nEntire program took {} {}.\n\n'.format(time_new_units, units))
    print('-'*79)
    
    return model_dir, history    
