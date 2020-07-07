'''The models.template_models module contains functions for building (ideally research-based) models.
'''

import os, sys
import inspect
import pathlib
import numpy as np
# for building and training models
from keras import applications
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, Conv2DTranspose, \
    LSTM, MaxPooling2D, TimeDistributed
from keras.constraints import max_norm
 
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)

import pysoundtool as pyst

###############################################################################

# TODO add maxpooling layer
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
        
    References
    ----------
    A. Sehgal and N. Kehtarnavaz, "A Convolutional Neural Network 
    Smartphone App for Real-Time Voice Activity Detection," in IEEE Access, 
    vol. 6, pp. 9017-9026, 2018. 
    '''
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
    settings = dict(feature_maps = feature_maps, 
                    kernel_size = kernel_size,
                    strides = strides,
                    activation_layer = activation_layer,
                    activation_output = activation_output,
                    input_shape = input_shape,
                    num_labels = num_labels,
                    dense_hidden_units = dense_hidden_units,
                    dropout = dropout)
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
