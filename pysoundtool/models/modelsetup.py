'''
TODO explore using just weights in ModelCheckpoint
save_weights_only=True, save_freq='epoch', and save_best_only
'''

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
import numpy as np
import pysoundtool as pyst

###############################################################################

def setup_callbacks(early_stop=True, patience=15, log=True,
                        log_filename=None, append=True, save_bestmodel=True, 
                        best_modelname=None,monitor='val_loss', verbose=1, save_best_only=True, mode='min'):
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
            log_dir = pyst.utils.check_dir('./model_logs/', 
                                           make=True, 
                                           write_into=True)
            log_filename = '{}log_'.format(log_dir)+\
                pyst.utils.get_date()+'.csv'
        # TODO test if pathlib.PosixPath won't work
        # if pathlib object, turn into string
        # TODO do some sort of check to ensure path is valid?
        if not isinstance(log_filename, str):
            log_filename = str(log_filename)
        csv_logging = CSVLogger(filename=log_filename, append=append)
        callbacks.append(csv_logging)
    if save_bestmodel:
        if best_modelname is None:
            model_dir = pyst.utils.check_dir('./best_models/', 
                                             make=True, 
                                             write_into=True)
            best_modelname = '{}model_'.format(model_dir)+\
                pyst.utils.get_date()+'.h5'
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
    if callbacks:
        return callbacks
    return None


