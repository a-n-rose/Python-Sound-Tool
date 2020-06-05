import os, sys
import inspect
import pathlib
import numpy as np
# for building and training models
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.preprocessing import StandardScaler, normalize
 
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)

import pysoundtool as pyst

###############################################################################

# TODO add maxpooling layer
def cnn_classifier(feature_maps, kernel_size, strides, activation_layer, 
                 activation_output, input_shape, num_labels, 
                 dense_hidden_units=None, dropout=None):
    '''Build a single or multilayer convolutional neural network.
    
    Parameters
    ----------
    feature_maps : list 
        The filter or feature map applied to the data. One feature map per
        convolutional neural layer required. For example, a list of length
        3 will result in a three-layer convolutional neural network.
    kernel_size : int or list 
        Must match the size of feature_maps. Determines the size of the
        correspondign feature map.
    strides : int
    
    activation_layer : 
    
    input_shape : tuple
        The shape of the input 
    dense_hidden_units : optional
    
    dropout : int, optional
        If dropout desired
    '''
    model = Sequential()
    if not isinstance(feature_maps, list):
        feature_maps = list(feature_maps)
    if not isinstance(kernel_size, list):
        kernel_size = list(kernel_size)
    assert len(feature_maps) == len(kernel_size)
    for i, featmap in enumerate(feature_maps):
        if i == 0:
            model.add(Conv2D(featmap),
                      kernel_size[i],
                      strides = strides,
                      activation = activation_layer,
                      input_shape = input_shape)
        else:
            model.add(Conv2D(featmap),
                      kernel_size[i],
                      strides = strides,
                      activation = activation_layer)
    if dense_hidden_units is not None:
        model.add(Dense(dense_hidden_units))
    if dropout is not None:
        model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(num_labels, activation = activation_output))    
    return model

