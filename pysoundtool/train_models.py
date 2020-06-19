import numpy as np
import pathlib
import keras
import time

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)
import pysoundtool as pyst
import pysoundtool.models as pystmodels

    
###############################################################################

# TODO include example extraction data in feature_extraction_dir?
def denoiser_train(model_name = 'model_autoencoder_denoise',
                   feature_type = None,
                   feature_extraction_dir = None,
                   use_generator = True,
                   normalized = False,
                   patience = 10, 
                   **kwargs):
    '''Collects training features and train autoencoder denoiser.
    
    Parameters
    ----------
    model_name : str
        The name for the model. This can be quite generic as the date up to 
        the millisecond will be added to ensure a unique name for each trained model.
        (default 'model_autoencoder_denoise')
        
    feature_type : str, optional
        The type of features that will be used to train the model. This is 
        only for the purposes of naming the model. If set to None, it will 
        not be included in the model name.
        
    feature_extraction_dir : str or pathlib.PosixPath
        Directory where extracted feature files are located (format .npy).
        
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
        The keyword arguments for keras.fit() and keras.fit_generator(). Note, 
        the keyword arguments differ for validation data so be sure to use the 
        correct keyword arguments, depending on if you use the generator or not.
        TODO: add link to keras.fit() and keras.fit_generator()
        
    Returns
    -------
    model_dir : pathlib.PosixPath
        The directory where the model and associated files can be found.
        
    See Also
    --------
    pysoundtool.data.separate_train_val_test_files
        Generates paths lists for train, validation, and test files. Useful
        for noisy vs clean datasets and also for multiple training files.
    
    pysoundtool.models.generator
        The generator function that feeds data to the model.
        
    pysoundtool.models.modelsetup.setup_callbacks
        The function that sets up callbacks (e.g. logging, save best model, early
        stopping, etc.)
    '''
    # ensure feature_extraction_folder exists:
    dataset_path = pyst.utils.check_dir(feature_extraction_dir, make=False)
    
    # designate where to save model and related files
    if feature_type:
        model_name += '_'+feature_type + '_' + pyst.utils.get_date() 
    else:
        model_name += '_' + pyst.utils.get_date() 
    model_dir = dataset_path.joinpath(model_name)
    model_dir = pyst.utils.check_dir(model_dir, make=True)
    model_name += '.h5'
    model_path = model_dir.joinpath(model_name)
    
    # prepare features files to load for training
    features_files = dataset_path.glob('*.npy')
    train_paths, val_paths, test_paths = pyst.data.separate_train_val_test_files(
        features_files)
    
    if not train_paths:
        # perhaps data files located in subdirectories 
        features_files = dataset_path.glob('**/*.npy')
        train_paths, val_paths, test_paths = pyst.data.separate_train_val_test_files(
            features_files)
        if not train_paths:
            raise FileNotFoundError('Could not locate train, validation, or test '+\
                '.npy files in the provided directory: \n{}'.format(dataset_path) +\
                    '\nThis program expects "train", "val", or "test" to be '+\
                        'included in each filename (not parent directory/ies) names.')
    
    # only need train and val feature data for autoencoder 
    train_paths_noisy, train_paths_clean = train_paths
    val_paths_noisy, val_paths_clean = val_paths
    
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
    denoiser, settings_dict = pystmodels.autoencoder_denoise(
        input_shape = input_shape)
    
    # create callbacks variable if not in kwargs
    # allow users to use different callbacks if desired
    if 'callbacks' not in kwargs:
        callbacks = pystmodels.setup_callbacks(patience = patience,
                                                best_modelname = model_path, 
                                                log_filename = model_dir.joinpath('log.csv'))
    adm = keras.optimizers.Adam(learning_rate=0.0001)
    denoiser.compile(optimizer=adm, loss='binary_crossentropy')

    # TODO remove?
    # save variables that are not too large:
    local_variables = locals()
    global_variables = globals()
    pyst.utils.save_dict(local_variables, 
                        model_dir.joinpath('local_variables_{}.csv'.format(
                            model_name)),
                        overwrite=True)
    pyst.utils.save_dict(global_variables,
                        model_dir.joinpath('global_variables_{}.csv'.format(
                            model_name)),
                        overwrite = True)
        
    # start training
    start = time.time()

    for i, train_path in enumerate(train_paths_noisy):
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

        if use_generator:
            train_generator = pystmodels.Generator(data_matrix1 = data_train_noisy, 
                                                    data_matrix2 = data_train_clean,
                                                    normalized = normalized)
            val_generator = pystmodels.Generator(data_matrix1 = data_val_noisy,
                                                data_matrix2 = data_val_clean,
                                                normalized = normalized)

            train_generator.generator()
            val_generator.generator()
            history = denoiser.fit_generator(train_generator.generator(),
                                                    steps_per_epoch = data_train_noisy.shape[0],
                                                    callbacks = callbacks,
                                                    validation_data = val_generator.generator(),
                                                    validation_steps = data_val_noisy.shape[0], 
                                                    **kwargs)

        else:
            #reshape to mix samples and batchsizes:
            train_shape = (data_train_noisy.shape[0]*data_train_noisy.shape[1],)+ data_train_noisy.shape[2:] + (1,)
            val_shape = (data_val_noisy.shape[0]*data_val_noisy.shape[1],)+ data_val_noisy.shape[2:] + (1,)
            
            if not normalized:
                data_train_noisy = pyst.feats.normalize(data_train_noisy)
                data_train_clean = pyst.feats.normalize(data_train_clean)
                data_val_noisy = pyst.feats.normalize(data_val_noisy)
                data_val_clean = pyst.feats.normalize(data_val_clean)
                
            X_train = data_train_noisy.reshape(train_shape)
            y_train = data_train_clean.reshape(train_shape)
            X_val = data_val_noisy.reshape(val_shape)
            y_val = data_val_clean.reshape(val_shape)
            
            denoiser.fit(X_train, y_train,
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
        model_features_dict_path = pyst.utils.save_dict(model_features_dict,
                                                        model_features_dict_path)
    print('\nFinished training the model. The model and associated files can be '+\
        'found here: \n{}'.format(model_dir))
    
    return model_dir


    
###############################################################################

# TODO include example extraction data in feature_extraction_dir?
def envclassifier_train(model_name = 'model_cnn_classifier',
                   feature_type = None,
                   feature_extraction_dir = None,
                   use_generator = True,
                   normalized = False,
                   patience = 10,
                   **kwargs):
    '''Collects training features and trains cnn environment classifier.
    
    This model may be applied to any speech and label scenario, for example, 
    male vs female speech, clinical vs healthy speech, simple speech / word
    recognition, as well as noise / scene / environment classification.
    
    Parameters
    ----------
    model_name : str
        The name for the model. This can be quite generic as the date up to 
        the millisecond will be added to ensure a unique name for each trained model.
        (default 'model_cnn_classifier')
        
    feature_type : str, optional
        The type of features that will be used to train the model. This is 
        only for the purposes of naming the model. If set to None, it will 
        not be included in the model name.
        
    feature_extraction_dir : str or pathlib.PosixPath
        Directory where extracted feature files are located (format .npy).
        
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
        The keyword arguments for keras.fit() and keras.fit_generator(). Note, 
        the keyword arguments differ for validation data so be sure to use the 
        correct keyword arguments, depending on if you use the generator or not.
        TODO: add link to keras.fit() and keras.fit_generator()
        
    Returns
    -------
    model_dir : pathlib.PosixPath
        The directory where the model and associated files can be found.
        
    See Also
    --------
    pysoundtool.data.separate_train_val_test_files
        Generates paths lists for train, validation, and test files. Useful
        for noisy vs clean datasets and also for multiple training files.
    
    pysoundtool.models.generator
        The generator function that feeds data to the model.
        
    pysoundtool.models.modelsetup.setup_callbacks
        The function that sets up callbacks (e.g. logging, save best model, early
        stopping, etc.)
    '''
    # ensure feature_extraction_folder exists:
    dataset_path = pyst.utils.check_dir(feature_extraction_dir, make=False)
    
    # designate where to save model and related files
    if feature_type:
        model_name += '_'+feature_type + '_' + pyst.utils.get_date() 
    else:
        model_name += '_' + pyst.utils.get_date() 
    model_dir = dataset_path.joinpath(model_name)
    model_dir = pyst.utils.check_dir(model_dir, make=True)
    model_name += '.h5'
    model_path = model_dir.joinpath(model_name)
    
    # prepare features files to load for training
    features_files = dataset_path.glob('*.npy')
    train_paths, val_paths, test_paths = pyst.data.separate_train_val_test_files(
        features_files)
    
    if not train_paths:
        # perhaps data files located in subdirectories 
        features_files = dataset_path.glob('**/*.npy')
        train_paths, val_paths, test_paths = pyst.data.separate_train_val_test_files(
            features_files)
        if not train_paths:
            raise FileNotFoundError('Could not locate train, validation, or test '+\
                '.npy files in the provided directory: \n{}'.format(dataset_path) +\
                    '\nThis program expects "train", "val", or "test" to be '+\
                        'included in each filename (not parent directory/ies) names.')
    
    # need dictionary for decoding labels:
    dict_decode_path = dataset_path.joinpath('dict_decode.csv')
    if not os.path.exists(dict_decode_path):
        raise FileNotFoundError('Could not find {}.'.format(dict_decode_path))
    dict_decode = pyst.utils.load_dict(dict_decode_path)
    num_labels = len(dict_decode)
    
    # load smaller dataset to determine input size:
    data_val = np.load(val_paths[0])
    # expect shape (num_audiofiles, num_frames, num_features + label_column)
    # subtract the label column and add dimension for 'color scale' 
    input_shape = (data_val.shape[1], data_val.shape[2] - 1, 1) 
    # remove unneeded variable
    del data_val
    
    # setup model 
    envclassifier, settings_dict = pystmodels.cnn_classifier(
        input_shape = input_shape,
        num_labels = num_labels)
    
    # create callbacks variable if not in kwargs
    # allow users to use different callbacks if desired
    if 'callbacks' not in kwargs:
        callbacks = pystmodels.setup_callbacks(patience = patience,
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
    pyst.utils.save_dict(local_variables, 
                        model_dir.joinpath('local_variables_{}.csv'.format(
                            model_name)),
                        overwrite=True)
    pyst.utils.save_dict(global_variables,
                        model_dir.joinpath('global_variables_{}.csv'.format(
                            model_name)),
                        overwrite = True)
        
    # start training
    start = time.time()

    for i, train_path in enumerate(train_paths):
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

        if use_generator:
            train_generator = pystmodels.Generator(data_matrix1 = data_train, 
                                                    data_matrix2 = None,
                                                    normalized = normalized)
            val_generator = pystmodels.Generator(data_matrix1 = data_val,
                                                data_matrix2 = None,
                                                normalized = normalized)
            test_generator = pystmodels.Generator(data_matrix1 = data_test,
                                                  data_matrix2 = None,
                                                  normalized = normalized)

            train_generator.generator()
            val_generator.generator()
            test_generator.generator()
            history = envclassifier.fit_generator(
                train_generator.generator(),
                steps_per_epoch = data_train.shape[0],
                callbacks = callbacks,
                validation_data = val_generator.generator(),
                validation_steps = data_val.shape[0],
                **kwargs)
            
            # TODO test how well prediction works. use simple predict instead?
            # need to define `y_test`
            X_test, y_test = pyst.feats.separate_dependent_var(data_test)
            y_predicted = envclassifier.predict_generator(
                test_generator.generator(),
                steps = data_test.shape[0])

        else:
            # TODO make scaling data optional?
            # data is separated and shaped for this classifier in scale_X_y..
            X_train, y_train, scalars = pyst.feats.scale_X_y(data_train,
                                                                is_train=True)
            X_val, y_val, __ = pyst.feats.scale_X_y(data_val,
                                                    is_train=False, 
                                                    scalars=scalars)
            X_test, y_test, __ = pyst.feats.scale_X_y(data_test,
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
        model_features_dict_path = pyst.utils.save_dict(model_features_dict,
                                                        model_features_dict_path)
    print('\nFinished training the model. The model and associated files can be '+\
        'found here: \n{}'.format(model_dir))
    
    return model_dir 

