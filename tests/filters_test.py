
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pysoundtool as pyst
import numpy as np
import pytest


def test_setup_bands():
    fil = pyst.Filter(num_bands=6)
    fil.setup_bands()
    band_start_index = fil.band_start_index
    band_end_index = fil.band_end_index
    print(low_bins)
    print(band_end_index)
    value1 = np.array([  0., 160., 320., 480., 640., 800.])
    value2 = np.array([160., 320., 480., 640., 800., 960.])
    #assert np.allclose(value1, band_start_index)
    #assert np.allclose(value2, band_end_index)
    assert np.array_equal(value1, band_start_index)
    assert np.array_equal(value2, band_end_index)

