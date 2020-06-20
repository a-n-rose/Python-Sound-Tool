
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pysoundtool as pyst
import numpy as np
import pytest


        
def test_separate_dependent_var_2d():
    data = np.array(range(18)).reshape(-1,1)
    with pytest.raises(ValueError):
        X, y = pyst.feats.separate_dependent_var(data)
    
    
def test_separate_dependent_var_3d():
    data = np.array(range(12)).reshape(2,2,3)
    X, y = pyst.feats.separate_dependent_var(data)
    expected1 = np.array([[[ 0,  1],[ 3,  4]],[[ 6,  7],[ 9, 10]]])
    expected2 = np.array([2, 8])
    assert np.array_equal(expected1, X)
    assert np.array_equal(expected2, y)
    
def test_separate_dependent_var_3d_1feature_valueerror():
    data = np.array(range(12)).reshape(2,6,1)
    with pytest.raises(ValueError):
        X, y = pyst.feats.separate_dependent_var(data)
        
def test_separate_dependent_var_3d_2feats():
    data = np.array(range(12)).reshape(2,3,2)
    X, y = pyst.feats.separate_dependent_var(data)
    expected1 = np.array([[[ 0],[ 2],[ 4]],[[ 6],[ 8],[10]]])
    expected2 = np.array([1, 7])
    assert np.array_equal(expected1, X)
    assert np.array_equal(expected2, y)
    
def test_scale_X_y_3d_train():
    np.random.seed(seed=0)
    data = np.random.random_sample(size=(2,1,3))
    X, y, scalars = pyst.feats.scale_X_y(data)
    expected1 = np.array([[[[ 1.],[ 1.]]], [[[-1.],[-1.]]]])
    expected2 = np.array([[0.60276338],[0.64589411]])
    assert np.allclose(expected1, X)
    assert np.allclose(expected2, y)
    assert isinstance(scalars, dict)
    
def test_scale_X_y_2d_train():
    np.random.seed(seed=0)
    data = np.random.random_sample(size=(2,3))
    with pytest.raises(ValueError):
        X, y, scalars = pyst.feats.scale_X_y(data)
        
def test_scale_X_y_3d_test():
    np.random.seed(seed=0)
    data1 = np.random.random_sample(size=(2,1,3))
    np.random.seed(seed=40)
    data2 = np.random.random_sample(size=(2,1,3))
    X, y, scalars = pyst.feats.scale_X_y(data1)
    X, y, scalars = pyst.feats.scale_X_y(data2, is_train=False, 
                                               scalars=scalars)
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
    X, y, scalars = pyst.feats.scale_X_y(data1)
    with pytest.raises(TypeError):
        X, y, scalars = pyst.feats.scale_X_y(data2, is_train=False)
        
def test_scale_X_y_3d_train_1feature_valueerror():
    np.random.seed(seed=0)
    data = np.random.random_sample(size=(2,3,1))
    with pytest.raises(ValueError):
        X, y, scalars = pyst.feats.scale_X_y(data)
