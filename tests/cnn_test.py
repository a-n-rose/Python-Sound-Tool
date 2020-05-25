
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pysoundtool as pyst
import pysoundtool.models as sound_models
import numpy as np
import pytest

###############################################################################

def test_scale_X_y_3d_train():
    np.random.seed(seed=0)
    data = np.random.random_sample(size=(2,1,3))
    X, y, scalars = sound_models.cnn.scale_X_y(data)
    expected1 = np.array([[[[ 1.],[ 1.]]], [[[-1.],[-1.]]]])
    expected2 = np.array([[0.60276338],[0.64589411]])
    assert np.allclose(expected1, X)
    assert np.allclose(expected2, y)
    assert isinstance(scalars, dict)
    
def test_scale_X_y_2d_train():
    np.random.seed(seed=0)
    data = np.random.random_sample(size=(2,3))
    with pytest.raises(ValueError):
        X, y, scalars = sound_models.cnn.scale_X_y(data)
        
def test_scale_X_y_3d_test():
    np.random.seed(seed=0)
    data1 = np.random.random_sample(size=(2,1,3))
    np.random.seed(seed=40)
    data2 = np.random.random_sample(size=(2,1,3))
    X, y, scalars = sound_models.cnn.scale_X_y(data1)
    X, y, scalars = sound_models.cnn.scale_X_y(data2, is_train=False, scalars=scalars)
    expected1 = np.array([[[[-1.],[-1.]]],[[[-1.],[-1.]]]])
    expected2 = np.array([[0.78853488],[0.30391231]])
    assert np.allclose(expected1, X)
    assert np.allclose(expected2, y)
    assert isinstance(scalars, dict)
    
def test_scale_X_y_3d_test_typeerror():
    np.random.seed(seed=0)
    data1 = np.random.random_sample(size=(2,1,3))
    np.random.seed(seed=40)
    data2 = np.random.random_sample(size=(2,1,3))
    X, y, scalars = sound_models.cnn.scale_X_y(data1)
    with pytest.raises(TypeError):
        X, y, scalars = sound_models.cnn.scale_X_y(data2, is_train=False)
        
def test_scale_X_y_3d_train_1feature_valueerror():
    np.random.seed(seed=0)
    data = np.random.random_sample(size=(2,3,1))
    with pytest.raises(ValueError):
        X, y, scalars = sound_models.cnn.scale_X_y(data)
        


