'''The models.template_models module contains functions for building (ideally research-based) models.
'''

import os, sys
import inspect
import pathlib
import numpy as np
# for building and training models
import tensorflow
from tensorflow.keras import applications
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Conv2DTranspose, \
    LSTM, MaxPooling2D, TimeDistributed
from tensorflow.keras.constraints import max_norm
 
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)

import soundpy as pyst

###############################################################################

            
def adjust_layers_cnn(**kwargs):
    '''Reduces layers of CNN until the model can be built.

    If the number of filters for 'mfcc' or 'fbank' is in the lower range
    (i.e. 13 or so), this causes issues with the default settings of
    the cnn architecture. The architecture was built with at least 40
    filters being applied during feature extraction. To deal with this
    problem, the number of CNN layers are reduced. 
    
    Parameters
    ----------
    **kwargs : Keyword arguments 
        Keyword arguments for soundpy.models.template_models.cnn_classifier
        
    Returns
    -------
    settings : dict 
        Updated dictionary with relevant settings for model.
        
    References
    ----------
    https://github.com/pgys/NoIze
    '''
    
    settings = dict(kwargs)
    try: 
        assert len(settings['feature_maps']) == len(settings['kernel_size'])
    except AssertionError:
        raise ValueError('Expect `feature_maps` and `kernel_size` to have same length.')
    if len(settings['feature_maps']) > 1:
        settings['feature_maps'] = kwargs['feature_maps'][:-1]
    else:
        raise ValueError('CNN Model cannot be trained with set number of '+\
            'features and feature maps.')
    if len(settings['kernel_size']) > 1:
        settings['kernel_size'] = kwargs['kernel_size'][:-1]
    return settings

def cnn_classifier(feature_maps = [40, 20, 10], 
                   kernel_size =  [(3, 3), (3, 3), (3, 3)],
                   strides = 2,
                   activation_layer = 'relu', 
                 activation_output = 'softmax', 
                 input_shape = (79, 40, 1), 
                 num_labels = 3, 
                 dense_hidden_units=100, dropout=0.25):
    '''Build a single or multilayer convolutional neural network.
    
    Parameters
    ----------
    feature_maps : int or list 
        The filter or feature map applied to the data. One feature map per
        convolutional neural layer required. For example, a list of length
        3 will result in a three-layer convolutional neural network.
    kernel_size : tuple or list of tuples
        Must match the number of feature_maps. The size of each corresponding feature map.
    strides : int
    
    activation_layer : str 
        (default 'relu')
    activation_outpu : str 
        (default 'softmax')
    input_shape : tuple
        The shape of the input 
    dense_hidden_units : int, optional
    
    dropout : float, optional
        Reduces overfitting
        
    Returns
    -------
    model : tf.keras.Model
        Model ready to be compiled.
        
    settings : dict 
        Dictionary with relevant settings for model.
        
    Warning
    -------
    If number features are not compatible with number of layers, warning raised and 
    layers adjusted. E.g. For lower number of MFCC features this will likely be applied if number
    of layers is greater than 1.
        
    References
    ----------
    A. Sehgal and N. Kehtarnavaz, "A Convolutional Neural Network 
    Smartphone App for Real-Time Voice Activity Detection," in IEEE Access, 
    vol. 6, pp. 9017-9026, 2018. 
    '''
    try:
        settings = dict(feature_maps = feature_maps, 
                        kernel_size = kernel_size,
                        strides = strides,
                        activation_layer = activation_layer,
                        activation_output = activation_output,
                        input_shape = input_shape,
                        num_labels = num_labels,
                        dense_hidden_units = dense_hidden_units,
                        dropout = dropout)
        model = Sequential()
        if not isinstance(feature_maps, list):
            feature_maps = list(feature_maps)
        if not isinstance(kernel_size, list):
            kernel_size = list(kernel_size)
        assert len(feature_maps) == len(kernel_size)
        for i, featmap in enumerate(feature_maps):
            if i == 0:
                model.add(Conv2D(featmap,
                        kernel_size[i],
                        strides = strides,
                        activation = activation_layer,
                        input_shape = input_shape))
            else:
                model.add(Conv2D(featmap,
                        kernel_size[i],
                        strides = strides,
                        activation = activation_layer))
        if dense_hidden_units is not None:
            model.add(Dense(dense_hidden_units))
        if dropout is not None:
            model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(num_labels, activation = activation_output))   
    except ValueError:
        import warnings
        msg = '\nWARNING: number of layers ({}) incompatible with number'.format(len(feature_maps))+\
            ' of features. Reducing number of layers until model and number features is compatible.'
        warnings.warn(msg)
        num_layers_orig = len(feature_maps)
        try:
            updated_settings = adjust_layers_cnn(**settings)
        except ValueError as e:
            print(e)
            return None, settings
        model, settings = cnn_classifier(**updated_settings)
        print('Updated number of layers: {}'.format(len(settings['feature_maps'])))
        
    return model, settings

# TODO: update model based on research
def autoencoder_denoise(input_shape,
                        kernel_size = (3,3),
                        max_norm_value = 2.0,
                        activation_function_layer = 'relu',
                        activation_function_output = 'sigmoid',
                        padding = 'same',
                        kernel_initializer = 'he_uniform'):
    '''Build a simple autoencoder denoiser.
    
    Parameters
    ----------
    input_shape : tuple
        Shape of the input data.
    max_norm_value : int or float
        
    Returns
    -------
    autoencoder : tf.keras.Model
        Model ready to be compiled
        
    References
    ----------
    Versloot, Christian (2019, December 19). Creating a Signal Noise Removal 
    Autoencoder with Keras. MachineCurve. https://www.machinecurve.com
    '''
    autoencoder = Sequential()
    autoencoder.add(Conv2D(128, kernel_size = kernel_size,
                            kernel_constraint = max_norm(max_norm_value),
                            activation = activation_function_layer,
                            kernel_initializer = kernel_initializer,
                            input_shape=input_shape))
    autoencoder.add(Conv2D(32, kernel_size = kernel_size,
                            kernel_constraint = max_norm(max_norm_value),
                            activation = activation_function_layer,
                            kernel_initializer = kernel_initializer))
    autoencoder.add(Conv2DTranspose(32, kernel_size = kernel_size,
                            kernel_constraint=max_norm(max_norm_value),
                            activation = activation_function_layer,
                            kernel_initializer = kernel_initializer))
    autoencoder.add(Conv2DTranspose(128, kernel_size = kernel_size,
                            kernel_constraint = max_norm(max_norm_value),
                            activation = activation_function_layer,
                            kernel_initializer = kernel_initializer))
    autoencoder.add(Conv2D(1, kernel_size = kernel_size,
                            kernel_constraint = max_norm(max_norm_value),
                            activation = activation_function_output, 
                            padding = padding))
    settings = dict(input_shape = input_shape,
                    kernel_size = kernel_size,
                    max_norm_value = max_norm_value,
                    activation_function_layer = activation_function_layer,
                    activation_function_output = activation_function_output,
                    padding = padding,
                    kernel_initializer = kernel_initializer)
    return autoencoder, settings

def resnet50_classifier(input_shape, num_labels, activation = 'softmax',
                        final_layer_name = 'features'):
    '''Simple image classifier built ontop of a pretrained ResNet50 model.
    
    References
    ----------
    Revay, S. & Teschke, M. (2019). Multiclass Language Identification using Deep 
    Learning on Spectral Images of Audio Signals. arXiv:1905.04348 [cs.SD]
    '''
    if len(input_shape) != 3:
        raise ValueError('ResNet50 expects 3D feature data, not {}D.'.format(
            len(input_shape)))
    if input_shape[-1] != 3:
        raise ValueError('ResNet50 expects 3 channels for RGB values, not {}.'.format(
            input_shape[-1]))
    base_model = applications.resnet50.ResNet50(weights=None, include_top=False,
                                                input_shape = input_shape)
    x = base_model.output 
    x = Flatten()(x)
    predictions = Dense(num_labels,
                        activation = activation,
                        name = final_layer_name)(x) # add name to this layer for TensorBoard visuals
    model = Model(inputs = base_model.input,
                  outputs = predictions)
    settings = dict(input_shape = input_shape,
                    num_labels = num_labels,
                    activation = activation,
                    final_layer_name = final_layer_name)
    return model, settings

def cnnlstm_classifier(num_labels, input_shape, lstm_cells, 
                       feature_map_filters = 32, kernel_size = (8,4),
                       pool_size = (3,3), dense_hidden_units = 60, 
                       activation_layer = 'relu', activation_output = 'softmax', 
                       dropout = 0.25):
    '''Model architecture inpsired from the paper below.
    
    References
    ----------
    Kim, Myungjong & Cao, Beiming & An, Kwanghoon & Wang, Jun. (2018). Dysarthric Speech Recognition Using Convolutional LSTM Neural Network. 10.21437/interspeech.2018-2250. 
    '''
    cnn = Sequential()
    cnn.add(Conv2D(feature_map_filters, kernel_size = kernel_size, activation = activation_layer))
    # non-overlapping pool_size 
    cnn.add(MaxPooling2D(pool_size = pool_size))
    cnn.add(Dropout(dropout))
    cnn.add(Flatten())

    # prepare stacked LSTM
    model = Sequential()
    model.add(TimeDistributed(cnn,input_shape = input_shape))
    model.add(LSTM(lstm_cells, return_sequences = True))
    model.add(LSTM(lstm_cells, return_sequences = True))
    model.add(Flatten())
    model.add(Dense(num_labels,activation = activation_output)) 
    settings = dict(input_shape = input_shape,
                    num_labels = num_labels,
                    kernel_size = kernel_size,
                    pool_size = pool_size,
                    activation_layer = activation_layer,
                    activation_output = activation_output,
                    lstm_cells = lstm_cells,
                    feature_map_filters = feature_map_filters,
                    dense_hidden_units = dense_hidden_units,
                    dropout = dropout)
    return model, settings
