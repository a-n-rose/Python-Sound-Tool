
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pysoundtool as pyst
from pysoundtool.mathfun import dsp
import numpy as np
import pytest


def test_calc_phase():
    np.random.seed(seed=0)
    rand_fft = np.random.random(2) + np.random.random(2) * 1j
    phase = dsp.calc_phase(rand_fft)
    value1 = np.array([0.67324134+0.73942281j, 0.79544405+0.60602703j])
    assert np.allclose(value1, phase)
    

def test_calc_phase_framelength10_default():
    frame_length = 10
    time = np.arange(0, 10, 0.1)
    signal = np.sin(time)[:frame_length]
    fft_vals = np.fft.fft(signal)
    phase = dsp.calc_phase(fft_vals)
    value1 = np.array([ 1.        +0.j,         -0.37872566+0.92550898j])
    assert np.allclose(value1, phase[:2])
    
def test_calc_phase_framelength10_radiansTrue():
    frame_length = 10
    time = np.arange(0, 10, 0.1)
    signal = np.sin(time)[:frame_length]
    fft_vals = np.fft.fft(signal)
    phase = dsp.calc_phase(fft_vals, radians = True)
    value1 = np.array([ 0.,         1.95921533])
    assert np.allclose(value1, phase[:2])
    
def test_reconstruct_whole_spectrum():
    x = np.array([3.,2.,1.,0.,0.,0.,0.])
    x_reconstructed = dsp.reconstruct_whole_spectrum(x)
    val1 = np.array([3., 2., 1., 0., 1., 2., 3.])
    assert np.array_equal(val1, x_reconstructed)
    assert len(x_reconstructed) == len(x)
    
def test_reconstruct_whole_spectrum_input4_nfft7():
    x = np.array([3.,2.,1.,0.])
    n_fft = 7
    x_reconstructed = dsp.reconstruct_whole_spectrum(x, n_fft=n_fft)
    val1 = np.array([3., 2., 1., 0., 1., 2., 3.])
    assert np.array_equal(val1, x_reconstructed)
    assert len(x_reconstructed) == n_fft
    
def test_reconstruct_whole_spectrum_input4_nfft6():
    x = np.array([3.,2.,1.,0.])
    n_fft= 6
    x_reconstructed = dsp.reconstruct_whole_spectrum(x, n_fft=n_fft)
    print(x_reconstructed)
    val1 = np.array([3., 2., 1., 0., 2., 3.])
    assert np.array_equal(val1, x_reconstructed)
    assert len(x_reconstructed) == n_fft
    
def test_reconstruct_whole_spectrum_input4_nfft5():
    x = np.array([3.,2.,1.,0.])
    n_fft = 5
    x_reconstructed = dsp.reconstruct_whole_spectrum(x, n_fft=n_fft)
    print(x_reconstructed)
    val1 = np.array([3., 2., 1., 2., 3.])
    assert np.array_equal(val1, x_reconstructed)
    assert len(x_reconstructed) == n_fft

def test_reconstruct_whole_spectrum_input4_nfft14():
    x = np.array([3.,2.,1.,0.])
    n_fft = 14
    x_reconstructed = dsp.reconstruct_whole_spectrum(x, n_fft=n_fft)
    print(x_reconstructed)
    val1 = np.array([3., 2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3.])
    assert np.array_equal(val1, x_reconstructed)
    assert len(x_reconstructed) == n_fft    
