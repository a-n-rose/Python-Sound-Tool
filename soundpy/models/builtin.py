'''The soundpy.models.builtin module includes example functions that train neural
networks on sound data.
''' 
import time
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

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
                   normalized = False,
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
        
    normalized : bool 
        If False, the data will be normalized before feeding to the model.
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
    input_shape = data_val_noisy.shape[2:] + (1,)
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

        if use_generator:
            train_generator = spdl.Generator(
                data_matrix1 = data_train_noisy, 
                data_matrix2 = data_train_clean,
                normalized = normalized,
                add_tensor_last = True)
            val_generator = spdl.Generator(
                data_matrix1 = data_val_noisy,
                data_matrix2 = data_val_clean,
                normalized = normalized,
                add_tensor_last = True)

            train_generator.generator()
            val_generator.generator()
            try:
                history = denoiser.fit(
                    train_generator.generator(),
                    steps_per_epoch = data_train_noisy.shape[0],
                    callbacks = callbacks,
                    validation_data = val_generator.generator(),
                    validation_steps = data_val_noisy.shape[0], 
                    **kwargs)
            except ValueError as e:
                print('\nValueError: ', e)
                raise ValueError('Try setting changing the parameter '+\
                    '`add_tensor_last` (in function '+\
                        '`soundpy.models.dataprep.Generator`)'+\
                        ' to either True, False, or None.')

        else:
            #reshape to mix samples and batchsizes:
            train_shape = (data_train_noisy.shape[0]*data_train_noisy.shape[1],)+ data_train_noisy.shape[2:] + (1,)
            val_shape = (data_val_noisy.shape[0]*data_val_noisy.shape[1],)+ data_val_noisy.shape[2:] + (1,)
            
            if not normalized:
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
                        normalized = False,
                        patience = 15,
                        add_tensor_last = True,
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
        
    normalized : bool 
        If False, the data will be normalized before feeding to the model.
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
    # expect shape (num_audiofiles, num_frames, num_features + label_column)
    # subtract the label column and add dimension for 'color scale' 
    input_shape = (data_val.shape[1], data_val.shape[2] - 1, 1) 
    # remove unneeded variable
    del data_val
    
    # setup model 
    envclassifier, settings_dict = spdl.cnn_classifier(
        input_shape = input_shape,
        num_labels = num_labels)
    
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

        if use_generator:
            train_generator = spdl.Generator(data_matrix1 = data_train, 
                                                    data_matrix2 = None,
                                                    normalized = normalized,
                                                    add_tensor_last = add_tensor_last)
            val_generator = spdl.Generator(data_matrix1 = data_val,
                                                data_matrix2 = None,
                                                normalized = normalized,
                                                    add_tensor_last = add_tensor_last)
            test_generator = spdl.Generator(data_matrix1 = data_test,
                                                  data_matrix2 = None,
                                                  normalized = normalized,
                                                    add_tensor_last = add_tensor_last)

            train_generator.generator()
            val_generator.generator()
            test_generator.generator()
            history = envclassifier.fit(
                train_generator.generator(),
                steps_per_epoch = data_train.shape[0],
                callbacks = callbacks,
                validation_data = val_generator.generator(),
                validation_steps = data_val.shape[0],
                **kwargs)
            
            # TODO test how well prediction works. use simple predict instead?
            # need to define `y_test`
            X_test, y_test = sp.feats.separate_dependent_var(data_test)
            y_predicted = envclassifier.predict(
                test_generator.generator(),
                steps = data_test.shape[0])

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
            
            envclassifier.evaluate(X_test, y_test)
            y_predicted = envclassifier.predict(X_test)
            # which category did the model predict?
            
    
        y_pred = np.argmax(y_predicted, axis=1)
        if len(y_pred.shape) > len(y_test.shape):
            y_test = np.expand_dims(y_test, axis=1)
        elif len(y_pred.shape) < len(y_test.shape):
            y_pred = np.expand_dims(y_pred, axis=1)
        try:
            assert y_pred.shape == y_test.shape
        except AssertionError:
            raise ValueError('The shape of prediction data {}'.format(y_pred.shape) +\
                ' does not match the `y_test` dataset {}'.format(y_test.shape) +\
                    '\nThe shapes much match in order to measure accuracy.')
                
        match = sum(y_test == y_pred)
        if len(match.shape) == 1:
            match = match[0]
        test_accuracy = round(match/len(y_test),4)
        print('\nModel reached accuracy of {}%'.format(test_accuracy*100))
        
        end_session = time.time()
        total_dur_sec_session = round(end_session-start_session,2)
        model_features_dict = dict(model_path = model_path,
                                data_train_path = data_train_path,
                                data_val_path = data_val_path, 
                                data_test_path = data_test_path, 
                                total_dur_sec_session = total_dur_sec_session,
                                use_generator = use_generator,
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
        How features are transformed back tino audio samples.
    '''
    # if values saved as strings, restore them to original type
    feature_type = sp.utils.restore_dictvalue(
        feat_settings_dict['feature_type'])
    win_size_ms = sp.utils.restore_dictvalue(
        feat_settings_dict['win_size_ms'])
    sr = sp.utils.restore_dictvalue(
        feat_settings_dict['sr'])
    percent_overlap = sp.utils.restore_dictvalue(
        feat_settings_dict['percent_overlap'])
    try:
        window = sp.utils.restore_dictvalue(feat_settings_dict['window'])
    except KeyError:
        window = None
    frames_per_sample = sp.utils.restore_dictvalue(
        feat_settings_dict['frames_per_sample'])
    input_shape = sp.utils.restore_dictvalue(
        feat_settings_dict['input_shape'])
    dur_sec = sp.utils.restore_dictvalue(
        feat_settings_dict['dur_sec'])
    num_feats = sp.utils.restore_dictvalue(
        feat_settings_dict['num_feats'])
    desired_shape = sp.utils.restore_dictvalue(
        feat_settings_dict['desired_shape'])
    
    feats = sp.feats.get_feats(new_audio, sr=sr, 
                                feature_type = feature_type,
                                win_size_ms = win_size_ms,
                                percent_overlap = percent_overlap,
                                window = window, 
                                dur_sec = dur_sec,
                                num_filters = num_feats)
    # are phase data still present? (only in stft features)
    if feats.dtype == np.complex and np.min(feats) < 0:
        original_phase = sp.dsp.calc_phase(feats,
                                               radians=False)
    elif 'stft' in feature_type or 'powspec' in feature_type:
        feats_stft = sp.feats.get_feats(new_audio, 
                                          sr=sr, 
                                          feature_type = 'stft',
                                          win_size_ms = win_size_ms,
                                          percent_overlap = percent_overlap,
                                          window = window, 
                                          dur_sec = dur_sec,
                                          num_filters = num_feats)
        original_phase = sp.dsp.calc_phase(feats_stft,
                                               radians = False)
    else:
        original_phase = None
    
    if 'signal' in feature_type:
        feats_zeropadded = np.zeros(desired_shape)
        feats_zeropadded = feats_zeropadded.flatten()
        if len(feats.shape) > 1:
            feats_zeropadded = feats_zeropadded.reshape(feats_zeropadded.shape[0],
                                                        feats.shape[1])
        if len(feats) > len(feats_zeropadded):
            feats = feats[:len(feats_zeropadded)]
        feats_zeropadded[:len(feats)] += feats
        # reshape here to avoid memory issues if total # samples is large
        feats = feats_zeropadded.reshape(desired_shape)
    
    feats = sp.feats.prep_new_audiofeats(feats,
                                           desired_shape,
                                           input_shape)
    # ensure same shape as feats
    if original_phase is not None:
        original_phase = sp.feats.prep_new_audiofeats(original_phase,
                                                        desired_shape,
                                                        input_shape)
    
    
    feats_normed = sp.feats.normalize(feats)
    denoiser = load_model(model)
    cleaned_feats = denoiser.predict(feats_normed, batch_size = frames_per_sample)
    
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
    cleaned_audio = sp.feats.feats2audio(cleaned_feats, 
                                           feature_type = feature_type,
                                           sr = sr, 
                                           win_size_ms = win_size_ms,
                                           percent_overlap = percent_overlap,
                                           phase = original_phase)
    if not isinstance(new_audio, np.ndarray):
        noisy_audio, __ = sp.loadsound(new_audio, sr=sr, remove_dc=remove_dc)
    else:
        noisy_audio = new_audio
    if len(cleaned_audio) > len(noisy_audio):
        cleaned_audio = cleaned_audio[:len(noisy_audio)]
    
    max_energy_original = np.max(noisy_audio)
    # match the scale of the original audio:
    cleaned_audio = sp.dsp.scalesound(cleaned_audio, max_val = max_energy_original)
    return cleaned_audio, sr


    
def collect_classifier_settings(feature_extraction_dir):
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
    num_feats = sp.utils.restore_dictvalue(settings_dict['num_feats'])
    # should the shape include the label column or not?
    # currently not
    feat_shape = sp.utils.restore_dictvalue(settings_dict['desired_shape'])
    feature_type = settings_dict['feat_type']
    return datasets, num_labels, feat_shape, num_feats, feature_type

def cnnlstm_train(feature_extraction_dir,
                  model_name = 'model_cnnlstm_classifier',
                  use_generator = True,
                  normalized = False,
                  patience = 15,
                  timesteps = 10,
                  context_window = 5,
                  colorscale = 1,
                  total_training_sessions = None,
                  add_tensor_last = False,
                  **kwargs):
    '''Many settings followed by the paper below.
    
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
    
    frame_width = context_window * 2 + 1 # context window w central frame
    input_shape = (timesteps, frame_width, num_feats, colorscale)
    model, settings = spdl.cnnlstm_classifier(num_labels = num_labels, 
                                                    input_shape = input_shape, 
                                                    lstm_cells = num_feats)
    

    # create callbacks variable if not in kwargs
    # allow users to use different callbacks if desired
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
                                                    normalized = normalized,
                                                    adjust_shape = input_shape,
                                                    add_tensor_last = add_tensor_last)
            val_generator = spdl.Generator(data_matrix1 = data_val,
                                                data_matrix2 = None,
                                                normalized = normalized,
                                                adjust_shape = input_shape,
                                                    add_tensor_last = add_tensor_last)
            test_generator = spdl.Generator(data_matrix1 = data_test,
                                                  data_matrix2 = None,
                                                  normalized = normalized,
                                                  adjust_shape = input_shape,
                                                  add_tensor_last = add_tensor_last)

            train_generator.generator()
            val_generator.generator()
            test_generator.generator()
            history = model.fit(
                train_generator.generator(),
                steps_per_epoch = data_train.shape[0],
                callbacks = callbacks,
                validation_data = val_generator.generator(),
                validation_steps = data_val.shape[0],
                **kwargs)
            
            # TODO test how well prediction works. use simple predict instead?
            # need to define `y_test`
            X_test, y_test = sp.feats.separate_dependent_var(data_test)
            y_predicted = model.predict(
                test_generator.generator(),
                steps = data_test.shape[0])

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
            
            model.evaluate(X_test, y_test)
            y_predicted = model.predict(X_test)
            # which category did the model predict?
            
    
        y_pred = np.argmax(y_predicted, axis=1)
        if len(y_pred.shape) > len(y_test.shape):
            y_test = np.expand_dims(y_test, axis=1)
        elif len(y_pred.shape) < len(y_test.shape):
            y_pred = np.expand_dims(y_pred, axis=1)
        try:
            assert y_pred.shape == y_test.shape
        except AssertionError:
            raise ValueError('The shape of prediction data {}'.format(y_pred.shape) +\
                ' does not match the `y_test` dataset {}'.format(y_test.shape) +\
                    '\nThe shapes much match in order to measure accuracy.')
                
        match = sum(y_test == y_pred)
        if len(match.shape) == 1:
            match = match[0]
        test_accuracy = round(match/len(y_test),4)
        print('\nModel reached accuracy of {}%'.format(test_accuracy*100))
        
        end_session = time.time()
        total_dur_sec_session = round(end_session-start_session,2)
        model_features_dict = dict(model_path = model_path,
                                data_train_path = data_train_path,
                                data_val_path = data_val_path, 
                                data_test_path = data_test_path, 
                                total_dur_sec_session = total_dur_sec_session,
                                use_generator = use_generator,
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

def resnet50_train(feature_extraction_dir,
                   model_name = 'model_resnet50_classifier',
                   use_generator = False,
                   normalized = False,
                   patience = 15,
                   colorscale = 3,
                   total_training_sessions = None,
                   add_tensor_last = None,
                   **kwargs):
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

        if use_generator:
            train_generator = spdl.Generator(data_matrix1 = data_train, 
                                                    data_matrix2 = None,
                                                    normalized = normalized,
                                                    adjust_shape = input_shape,
                                                    add_tensor_last = add_tensor_last,
                                                    gray2color = True)
            val_generator = spdl.Generator(data_matrix1 = data_val,
                                                data_matrix2 = None,
                                                normalized = normalized,
                                                adjust_shape = input_shape,
                                                    add_tensor_last = add_tensor_last,
                                                    gray2color = True)
            test_generator = spdl.Generator(data_matrix1 = data_test,
                                                  data_matrix2 = None,
                                                  normalized = normalized,
                                                  adjust_shape = input_shape,
                                                  add_tensor_last = add_tensor_last,
                                                    gray2color = True)

            train_generator.generator()
            val_generator.generator()
            test_generator.generator()
            history = model.fit(
                train_generator.generator(),
                steps_per_epoch = data_train.shape[0],
                callbacks = callbacks,
                validation_data = val_generator.generator(),
                validation_steps = data_val.shape[0],
                **kwargs)
            
            # TODO test how well prediction works. use simple predict instead?
            # need to define `y_test`
            X_test, y_test = sp.feats.separate_dependent_var(data_test)
            y_predicted = model.predict(
                test_generator.generator(),
                steps = data_test.shape[0])

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
            
            model.evaluate(X_test, y_test)
            y_predicted = model.predict(X_test)
            # which category did the model predict?
            
    
        y_pred = np.argmax(y_predicted, axis=1)
        if len(y_pred.shape) > len(y_test.shape):
            y_test = np.expand_dims(y_test, axis=1)
        elif len(y_pred.shape) < len(y_test.shape):
            y_pred = np.expand_dims(y_pred, axis=1)
        try:
            assert y_pred.shape == y_test.shape
        except AssertionError:
            raise ValueError('The shape of prediction data {}'.format(y_pred.shape) +\
                ' does not match the `y_test` dataset {}'.format(y_test.shape) +\
                    '\nThe shapes much match in order to measure accuracy.')
                
        match = sum(y_test == y_pred)
        if len(match.shape) == 1:
            match = match[0]
        test_accuracy = round(match/len(y_test),4)
        print('\nModel reached accuracy of {}%'.format(test_accuracy*100))
        
        end_session = time.time()
        total_dur_sec_session = round(end_session-start_session,2)
        model_features_dict = dict(model_path = model_path,
                                data_train_path = data_train_path,
                                data_val_path = data_val_path, 
                                data_test_path = data_test_path, 
                                total_dur_sec_session = total_dur_sec_session,
                                use_generator = use_generator,
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

def denoiser_extract_train(
    dataset_dict = None,
    audiodata_path = None,
    feature_type = 'fbank',
    num_feats = None,
    mono = True,
    rate_of_change = False,
    rate_of_acceleration = False,
    subtract_mean = False,
    use_scipy = False,
    dur_sec = 1,
    win_size_ms = 25,
    percent_overlap = 0.5,
    sr = 22050,
    fft_bins = None,
    frames_per_sample = None, 
    labeled_data = False, 
    batch_size = 10,
    use_librosa = True, 
    center = True, 
    mode = 'reflect', 
    log_settings = True, 
    model_name = 'env_classifier',
    epoch = 5,
    patience = 15,
    callbacks = None,
    use_generator = True,
    augmentation_dict = None):
    '''Extract and augment features during training of a denoising model.
    '''
    if dataset_dict is None:
        # set up datasets
        if audiodata_path is None:
            raise ValueError('Function `denoiser_extract_train` expects either:\n'+\
                '1) a `dataset_dict` with audiofile pathways assigned to datasets OR'+\
                    '\n2) a `audiodata_path` indicating where audiofiles for'+\
                        'training are located.\n**Both cannot be None.')
        
        # sp.check_dir:
        # raises error if this path doesn't exist (make = False)
        # if does exist, returns path as pathlib.PosixPath object
        data_dir = sp.check_dir(audiodata_path, make = False)
    
    else:
        # use pre-collected dataset dict
        dataset_dict = dataset_dict
    pass


def envclassifier_extract_train(
    model_name = 'env_classifier',
    dataset_dict = None,
    num_labels = None,
    augment_dict_list = None,
    audiodata_path = None,
    save_new_files_dir = None,
    frames_per_sample = None, # images_per_sample, sections_per_sample..? 
    labeled_data = True,
    batch_size = 10,
    use_librosa = True, 
    center = True, 
    mode = 'reflect', 
    epochs = 5,
    patience = 15,
    callbacks = None,
    random_seed = None,
    visualize = False,
    vis_every_n_items = 50,
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
        
    augment_dict_list : list of dicts, optional
        List of dictionaries containing keys (e.g. 'add_white_noise'). See 
        `soundpy.augment.list_augmentations`and corresponding True or False
        values. If the value is True, the key / augmentation gets implemented. 
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
    
    if 'stft' not in kwargs['feature_type'] and 'powspec' not in kwargs['feature_type']:
        raise ValueError('Function `envclassifier_extract_train` can only reliably '+\
            'work if `feature_type` parameter is set to "stft" or "powspec".'+\
                ' In future versions the other feature types will be made available.')
    
    # ensure defaults are set if not included in kwargs:
    if 'win_size_ms' not in kwargs:
        kwargs['win_size_ms'] = 25
    if 'percent_overlap' not in kwargs:
        kwargs['percent_overlap'] = 0.5
    if 'mono' not in kwargs:
        kwargs['mono'] = True
    if 'rate_of_change' not in kwargs:
        kwargs['rate_of_change'] = False
    if 'rate_of_acceleration' not in kwargs:
        kwargs['rate_of_acceleration'] = False
    if 'subtract_mean' not in kwargs:
        kwargs['subtract_mean'] = False
    if 'dur_sec' not in kwargs:
        raise ValueError('Function `envclassifier_extract_train``requires ' +\
            'the keyword argument `dur_sec` to be set. How many seconds of audio '+\
                'from each audio file would you like to use for training?')
    if 'sr' not in kwargs:
        kwargs['sr'] = 48000
    if 'fft_bins' not in kwargs:
        kwargs['fft_bins'] = None
    if 'real_signal' not in kwargs:
        kwargs['real_signal'] = True
    if 'window' not in kwargs:
        kwargs['window'] = 'hann'
    if 'zeropad' not in kwargs:
        kwargs['zeropad'] = True
    if 'num_filters' not in kwargs:
        kwargs['num_filters'] = 40
        
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
            raise ValueError('Function `denoiser_extract_train` expects either:\n'+\
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
            labels.append(label.stem)
        labels = set(labels)
    
        # create encoding and decoding dictionaries of labels:
        dict_encode, dict_decode = sp.datasets.create_dicts_labelsencoded(
            labels,
            add_extra_label = True,
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
            raise ValueError('Function `denoiser_extract_train` requires '+\
                '`num_labels` to be provided if a pre-made `dataset_dict` is provided.')
        # use pre-collected dataset dict
        dataset_dict = sp.utils.load_dict(dataset_dict)
        # don't have the label data available
        dict_encode, dict_decode = None, None
        

    input_shape = spdl.dataprep.get_input_shape(kwargs, labeled_data = labeled_data,
                                  frames_per_sample = frames_per_sample,
                                  use_librosa = use_librosa)
    
    # update num_fft_bins to input_shape's last column, expecting it to be freq bins  / feats: 
    if kwargs['real_signal']:
        kwargs['fft_bins'] = input_shape[-1] 
    else:
        kwargs['fft_bins'] = input_shape[-1] * 2 -1
        
    # currently unnecessary as 'fbank' and 'mfcc' are not supported yet.
    if 'fbank' in kwargs['feature_type'] or 'mfcc' in kwargs['feature_type']:
        if not kwargs['use_scipy']:
            kwargs['fmax'] = kwargs['sr'] * 2.0
    # extract validation data (must already be extracted)
    val_dict = dict([('val',dataset_dict['val'])])
    val_path = dataset_path.joinpath('val_data.npy')
    val_path_dict = dict([('val', val_path)])

    print('\nExtracting validation data for use in training:')
    val_dict, val_path_dict = sp.feats.save_features_datasets(
        val_dict,
        val_path_dict,
        labeled_data = labeled_data,
        **kwargs)

    val_data = np.load(val_path_dict['val'])


    # start training
    start = time.time()

    input_shape = input_shape + (1,)
    if dict_encode is not None:
        num_labels = len(dict_encode) 
    # otherwise should arleady be specified

    if augment_dict_list is None:
        augment_dict_list = [dict()]


    # designate where to save model and related files
    model_name = 'audioaugment_' + kwargs['feature_type']
    model_dir = dataset_path.joinpath(model_name)
    model_dir = sp.utils.check_dir(model_dir, make=True)
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

    for i, augment_dict in enumerate(augment_dict_list):

        # items that need to be called with each iteration:
        # save best model for each iteration - don't want to be overwritten
        # with worse model
        best_modelname = str(model_path) + '_session{}.h5'.format(i)
        callbacks = spdl.setup_callbacks(
            patience = patience,
            best_modelname = best_modelname, 
            log_filename = model_dir.joinpath('log.csv'),
            append = True)

        normalize = True
        add_tensor_last = False
        add_tensor_first = True
        
        train_generator = spdl.GeneratorFeatExtraction(
            datalist = dataset_dict['train'],
            model_name = model_name,
            normalize = normalize,
            apply_log = False,
            randomize = True, # want the data order to be different for each iteration 
            random_seed = None,
            input_shape = input_shape,
            batch_size = batch_size, 
            add_tensor_last = add_tensor_last, 
            add_tensor_first = add_tensor_first,
            gray2color = False,
            visualize = visualize,
            vis_every_n_items = vis_every_n_items,
            visuals_dir = model_dir.joinpath('images'),
            decode_dict = dict_decode,
            dataset = 'train',
            augment_dict = augment_dict,
            label_silence = False,
            **kwargs)
        
        val_generator = spdl.Generator(
            data_matrix1 = val_data,
            add_tensor_last = True,
            adjust_shape = input_shape[:-1])
        
        if i == 0:
            # Print how many epochs possible if several augmentations
            if len(augment_dict_list) > 1:
                print('~'*79)
                print('\nNOTE: due to several augmentations, total epochs possible:' + \
                    '\n{} epochs\n'.format(len(augment_dict_list * epochs)))
                print('~'*79)
                print()
        print('-'*79)
        print('\nTRAINING SESSION ',i+1, ' out of ', len(augment_dict_list))
        if augment_dict:
            print('\nAugmentation(s) applied: \n')
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
            train_generator.generator(),
            steps_per_epoch = len(dataset_dict['train']),
            callbacks = callbacks,
            epochs = epochs,
            validation_data = val_generator.generator(),
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

        model_features_dict_path = model_dir.joinpath('info_{}_session{}.csv'.format(
            model_name, i))
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
