from keras.models import Model
from keras.models import load_model 
import numpy as np
import matplotlib.pyplot as plt

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)
import pysoundtool_online as pyst


def featuremaps(features, model, image_dir='./feature_maps/'):
    '''Saves the feature maps of each convolutional layer as .png file.
    
    References
    ----------
    Brownlee, Jason (2019, May, 6). How to Visualize Filters and Feature
    Maps in Convolutional Neural Networks. Machine Learning Mastery.
    https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
    '''
    conv_idx = []
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if 'conv' in layer.name:
            conv_idx.append(i)
    for idx in conv_idx:
        model_featmaps = Model(inputs = model.inputs,
                    outputs = model.layers[idx].output)
        featuremaps = model_featmaps.predict(features)
        for i in range(featuremaps.shape[-1]):
            plt.clf()
            plt.imshow(featuremaps[0,:,:,i], cmap='gray')
            image_dir = pyst.utils.check_dir(image_dir, make=True)
            image_path = image_dir.joinpath('layer_{}'.format(idx),
                                            'featmap_{}.png'.format(i))
            image_par = pyst.utils.check_dir(image_path.parent, make=True)
            plt.savefig(image_path)
