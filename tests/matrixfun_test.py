
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from pysoundtool.mathfun import matrixfun
import numpy as np
import pytest


def test_overlap_add():
    enhanced_matrix = np.ones((4, 4))
    frame_length = 4
    overlap = 2
    sig = matrixfun.overlap_add(enhanced_matrix, frame_length, overlap)
    value1 = np.array([1., 1., 2., 2., 2., 2., 2., 2., 1., 1.])
    assert np.array_equal(value1, sig)
    
def test_overlap_add():
    enhanced_matrix = np.ones((4, 4))
    frame_length = 4
    overlap = 1
    sig = matrixfun.overlap_add(enhanced_matrix, frame_length, overlap)
    v1 = np.array([1., 1., 1., 2., 1., 1., 2., 1., 1., 2., 1., 1., 1.])
    assert np.array_equal(v1, sig)
    
def test_overlap_add_framelength_mismatch():
    enhanced_matrix = np.ones((4, 4))
    frame_length = 3
    overlap = 1
    with pytest.raises(TypeError):
        sig = matrixfun.overlap_add(enhanced_matrix, 
                                    frame_length, 
                                    overlap)
