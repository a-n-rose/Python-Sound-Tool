import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)

import pysoundtool as pyst
import pysoundtool.models as pystmodels

cnn_classifier = pystmodels.cnn_classifier(feature_maps = [40, 20, 10],
                                  kernel_size = [(3, 3), (3, 3), (3, 3)],
                                  strides = 2,
                                  activation_layer = 'relu',
                                  activation_output='softmax',
                                  input_shape = (79, 40, 1),
                                  num_labels = 3,
                                  dense_hidden_units = 100,
                                  dropout = 0.25)

autoencoder_denoise = pystmodels.autoencoder_denoise(input_shape = (10,11,40),
                                             max_norm_value=2.0)


