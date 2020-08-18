'''The models.modelsetup module containes functionality for preparing for training a model
'''

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)
import tensorflow
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger,\
    ModelCheckpoint, TensorBoard
import numpy as np
import soundpy as sp

###############################################################################

def setup_layers(num_features, num_layers, kernel_shape = (3,3), max_feature_map = 64):
    '''Sets up `feature_maps` and `kernels` for 1 or more layered convolutional neural networks.
    
    Parameters
    ----------
    num_features : int 
        The number of features used to train the model. This will be used to set the number 
        of `feature_maps` for each layer. 
        
    num_layers : int 
        The number of layers desired 
        
    kernel_shape : tuple or int 
        The shape of the desired kernel 
        
    max_feature_map : int 
        The maximum size of feature map / filter. This depends on the system and is relevant for
        processing higher definition features, such as STFT features. If this is set too large 
        given memory restraints, training may be 'killed'.
        
    Returns
    -------
    feature_maps : list 
        List of feature maps or filters that will be applied to each layer of the network.
        
    kernels : list 
        List of kernels that will be applied to each layer of the network. Matches length of 
        `feature_maps`
        
    Warning
    -------
    If `num_features` is larger than the `max_feature_map`. The `num_features` is usually used to
    set the first feature map, but if too large, will be reduced to be lower than `max_feature_map`.
        
    '''
    if num_features > max_feature_map:
        num_features_orig = num_features
        while num_features > max_feature_map:
            num_features //= 1.1
        import warnings
        msg = '\nWARNING: feature maps will not be set according to `num_features` '+\
            'as `num_features` is too high ({})'.format(num_features_orig)+\
                ' The first feature map will be reduced to {}'.format(num_features)
        warnings.warn(msg)
    feature_maps = []
    kernels = []
    for i in range(num_layers):
        if i == 0:
            feature_maps.append(num_features)
            kernels.append(kernel_shape)
        else:
            feature_maps.append(feature_maps[i-1]//2)
            kernels.append(kernel_shape)
    return feature_maps, kernels

def setup_callbacks(early_stop=True, patience=15, log=True,
                        log_filename=None, append=True, save_bestmodel=True, 
                        best_modelname=None,monitor='val_loss', verbose=1, save_best_only=True, mode='min', tensorboard=True,
                        write_images=False, x_test=None, y_test=None,
                        batch_size = None, embedded_layer_name = None):
    '''Easy set up of early stopping, model logging, and saving best model.
    
    Parameters
    ----------
    early_stop : bool 
        Whether or not the model should stop if training is not improving 
        (default True)
    patience : int 
        The number of epochs the model should complete without improvement
        before stopping training. (default 15)
    log : bool 
        If true, the accuracy, loss, and (if possible) the val_accuracy and 
        val_loss for each epoch will be saved in a .csv file. (default True)
    log_filename : str or pathlib.PosixPath, optional
        The filename the logging information will be stored. If None, 
        the date will be used as a unique .csv filename in a subfolder 
        'model_logs' in the local directory.
    save_bestmodel : bool
        If True, the best performing model will be saved.
    best_modelname : str or pathlib.PosixPath
        The name to save the best model version under. If None, the date
        will be used to create a unique .h5 filename and it will be saved
        in a subfolder 'best_models' in the local directory.
    monitor : str 
        The metric to be used to measure model performance. 
        (default 'val_loss'
    verbose : bool
        If True, the state of the model will be printed. (default True)
    save_best_only : bool 
        If True, the best performing model will overwrite any previously 
        saved 'best model'.
    mode : str 
        If `monitor` is set to 'val_loss', this should be set to 'min'.
        If `monitor``is set to 'val_acc', this should be set to 'max'. If 
        `mode` is set to 'auto', the direction will be inferred. 
        (default 'min')
    tensorboard : bool 
        If True, logs for TensorBoard will be made.
        
    Returns
    -------
    callbacks : # TODO what data type is this?
        The callbacks ready to be applied to Keras model training.
     '''
    callbacks = []
    if early_stop:
        early_stopping_callback = EarlyStopping(
            monitor=monitor,
            patience=patience)
        callbacks.append(early_stopping_callback)
    if log or log_filename:
        if log_filename is None:
            # create directory to store log files:
            log_dir = sp.utils.check_dir('./model_logs/', 
                                           make=True, 
                                           write_into=True)
            log_filename = '{}log_'.format(log_dir)+\
                sp.utils.get_date()+'.csv'
        # TODO test if pathlib.PosixPath won't work
        # if pathlib object, turn into string
        # TODO do some sort of check to ensure path is valid?
        if not isinstance(log_filename, str):
            log_filename = str(log_filename)
        csv_logging = CSVLogger(filename=log_filename, append=append)
        callbacks.append(csv_logging)
    if save_bestmodel:
        if best_modelname is None:
            model_dir = sp.utils.check_dir('./best_models/', 
                                             make=True, 
                                             write_into=True)
            best_modelname = '{}model_'.format(model_dir)+\
                sp.utils.get_date()+'.h5'
        # TODO test if pathlib.PosixPath won't work
        # if pathlib object, turn into string
        # TODO do some sort of check to ensure path is valid?
        if not isinstance(best_modelname, str):
            best_modelname = str(best_modelname)
        checkpoint_callback = ModelCheckpoint(best_modelname, monitor=monitor,
                                                verbose=verbose,
                                                save_best_only=save_best_only,
                                                mode=mode)
        callbacks.append(checkpoint_callback)
    if tensorboard:
        if log_filename is not None:
            log_filename = sp.utils.string2pathlib(log_filename)
            log_dir = log_filename.parent
            log_dir = log_dir.joinpath('tb_logs/')
        else:
            log_dir = sp.utils.string2pathlib('./model_logs/tb_logs/')
        log_dir = sp.utils.check_dir(log_dir, make=True)
        if x_test is not None and y_test is not None:
            with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
                np.savetxt(f, y_test)
            embeddings_freq = 1
            if embedded_layer_name is None:
                embedded_layer_name = 'features'
            embeddings_layer_names = [embedded_layer_name]
            embeddings_metadata = 'metadata.tsv'
            embeddings_data = x_test
        else:
            embeddings_freq = 0
            embeddings_layer_names = None
            embeddings_metadata = None
            embeddings_data = None
        tb = TensorBoard(log_dir, batch_size = batch_size, write_images=write_images, 
                         embeddings_freq = embeddings_freq, 
                         embeddings_layer_names = embeddings_layer_names, 
                         embeddings_metadata = embeddings_metadata,
                         embeddings_data = embeddings_data)
        callbacks.append(tb)
    if callbacks:
        return callbacks
    return None
