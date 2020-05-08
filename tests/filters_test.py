
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pysoundtool as pyst
import numpy as np
import pytest


def test_setup_bands_default():
    fil = pyst.Filter()
    fil.setup_bands()
    band_start_freq = fil.band_start_freq
    band_end_freq = fil.band_end_freq
    value1 = np.array([  0., 160., 320., 480., 640., 800.])
    value2 = np.array([160., 320., 480., 640., 800., 960.])
    assert np.array_equal(value1, band_start_freq)
    assert np.array_equal(value2, band_end_freq)
    
def test_setup_bands_8():
    fil = pyst.Filter(num_bands = 8)
    fil.setup_bands()
    band_start_freq = fil.band_start_freq
    band_end_freq = fil.band_end_freq
    value1 = np.array([  0., 120., 240., 360., 480., 600., 720., 840.])
    value2 = np.array([120., 240., 360., 480., 600., 720., 840., 960.])
    assert np.array_equal(value1, band_start_freq)
    assert np.array_equal(value2, band_end_freq)

def test_setup_bands_frameduration16ms():
    fil = pyst.Filter(frame_duration = 16)
    fil.setup_bands()
    band_start_freq = fil.band_start_freq
    band_end_freq = fil.band_end_freq
    value1 = np.array([  0., 128., 256., 384., 512., 640.])
    value2 = np.array([128., 256., 384., 512., 640., 768.])
    assert np.array_equal(value1, band_start_freq)
    assert np.array_equal(value2, band_end_freq)
    
def test_setup_bands_frameduration500ms():
    fil = pyst.Filter(frame_duration = 500)
    fil.setup_bands()
    band_start_freq = fil.band_start_freq
    band_end_freq = fil.band_end_freq
    value1 = np.array([    0.,  4000.,  8000., 12000., 16000., 20000.])
    value2 = np.array([ 4000.,  8000., 12000., 16000., 20000., 24000.])
    assert np.array_equal(value1, band_start_freq)
    assert np.array_equal(value2, band_end_freq)
