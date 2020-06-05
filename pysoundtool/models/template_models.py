import os, sys
import inspect
import pathlib
import numpy as np
# for building and training models
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, Conv2DTranspose
from keras.constraints import max_norm
from sklearn.preprocessing import StandardScaler, normalize
 
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
                 dense_hidden_units=None, dropout=None):
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
    return model

# TODO: update model based on research
def autoencoder_denoise(input_shape,
                        max_norm_value=2.0):
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
    https://www.machinecurve.com/index.php/2019/12/19/creating-a-signal-noise-removal-autoencoder-with-keras/#creating-the-autoencoder
    '''
    autoencoder = Sequential()
    autoencoder.add(Conv2D(128, kernel_size=(3, 3),
                            kernel_constraint=max_norm(max_norm_value),
                            activation='relu',
                            kernel_initializer='he_uniform',
                            input_shape=input_shape))
    autoencoder.add(Conv2D(32, kernel_size=(3, 3),
                            kernel_constraint=max_norm(max_norm_value),
                            activation='relu',
                            kernel_initializer='he_uniform'))
    autoencoder.add(Conv2DTranspose(32, kernel_size=(3,3),
                            kernel_constraint=max_norm(max_norm_value),
                            activation='relu',
                            kernel_initializer='he_uniform'))
    autoencoder.add(Conv2DTranspose(128, kernel_size=(3,3),
                            kernel_constraint=max_norm(max_norm_value),
                            activation='relu',
                            kernel_initializer='he_uniform'))
    autoencoder.add(Conv2D(1, kernel_size=(3, 3),
                            kernel_constraint=max_norm(max_norm_value),
                            activation='sigmoid', padding='same'))
    return autoencoder
