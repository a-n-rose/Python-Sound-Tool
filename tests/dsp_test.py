
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pysoundtool as pyst
import numpy as np
import pytest


def test_calc_phase():
    np.random.seed(seed=0)
    rand_fft = np.random.random(2) + np.random.random(2) * 1j
    phase = pyst.dsp.calc_phase(rand_fft)
    value1 = np.array([0.67324134+0.73942281j, 0.79544405+0.60602703j])
    assert np.allclose(value1, phase)
    

def test_calc_phase_framelength10_default():
    frame_length = 10
    time = np.arange(0, 10, 0.1)
    signal = np.sin(time)[:frame_length]
    fft_vals = np.fft.fft(signal)
    phase = pyst.dsp.calc_phase(fft_vals)
    value1 = np.array([ 1.        +0.j,         -0.37872566+0.92550898j])
    assert np.allclose(value1, phase[:2])
    
def test_calc_phase_framelength10_radiansTrue():
    frame_length = 10
    time = np.arange(0, 10, 0.1)
    signal = np.sin(time)[:frame_length]
    fft_vals = np.fft.fft(signal)
    phase = pyst.dsp.calc_phase(fft_vals, radians = True)
    value1 = np.array([ 0.,         1.95921533])
    assert np.allclose(value1, phase[:2])
    
def test_reconstruct_whole_spectrum():
    x = np.array([3.,2.,1.,0.,0.,0.,0.])
    x_reconstructed = pyst.dsp.reconstruct_whole_spectrum(x)
    expected = np.array([3., 2., 1., 0., 1., 2., 3.])
    assert np.array_equal(expected, x_reconstructed)
    assert len(x_reconstructed) == len(x)
    
def test_reconstruct_whole_spectrum_input4_nfft7():
    x = np.array([3.,2.,1.,0.])
    n_fft = 7
    x_reconstructed = pyst.dsp.reconstruct_whole_spectrum(x, n_fft=n_fft)
    expected = np.array([3., 2., 1., 0., 1., 2., 3.])
    assert np.array_equal(expected, x_reconstructed)
    assert len(x_reconstructed) == n_fft
    
def test_reconstruct_whole_spectrum_input4_nfft6():
    x = np.array([3.,2.,1.,0.])
    n_fft= 6
    x_reconstructed = pyst.dsp.reconstruct_whole_spectrum(x, n_fft=n_fft)
    print(x_reconstructed)
    expected = np.array([3., 2., 1., 0., 2., 3.])
    assert np.array_equal(expected, x_reconstructed)
    assert len(x_reconstructed) == n_fft
    
def test_reconstruct_whole_spectrum_input4_nfft5():
    x = np.array([3.,2.,1.,0.])
    n_fft = 5
    x_reconstructed = pyst.dsp.reconstruct_whole_spectrum(x, n_fft=n_fft)
    print(x_reconstructed)
    expected = np.array([3., 2., 1., 2., 3.])
    assert np.array_equal(expected, x_reconstructed)
    assert len(x_reconstructed) == n_fft

def test_reconstruct_whole_spectrum_input4_nfft14():
    x = np.array([3.,2.,1.,0.])
    n_fft = 14
    x_reconstructed = pyst.dsp.reconstruct_whole_spectrum(x, n_fft=n_fft)
    print(x_reconstructed)
    expected = np.array([3., 2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3.])
    assert np.array_equal(expected, x_reconstructed)
    assert len(x_reconstructed) == n_fft    
    
def test_reconstruct_whole_spectrum_complexvals():
    np.random.seed(seed=0)
    x_complex = np.random.random(2) + np.random.random(2) * 1j
    n_fft = int(2*len(x_complex))
    x_reconstructed = pyst.dsp.reconstruct_whole_spectrum(x_complex,
                                                     n_fft = n_fft)
    expected = np.array([0.5488135 +0.60276338j, 0.71518937+0.54488318j, 0.        +0.j, 0.5488135 +0.60276338j])
    print(x_reconstructed)
    assert np.allclose(expected, x_reconstructed)
    assert len(x_reconstructed) == n_fft    
    
    
def test_overlap_add():
    enhanced_matrix = np.ones((4, 4))
    frame_length = 4
    overlap = 2
    sig = pyst.dsp.overlap_add(enhanced_matrix, frame_length, overlap)
    expected = np.array([1., 1., 2., 2., 2., 2., 2., 2., 1., 1.])
    assert np.array_equal(expected, sig)
    
def test_overlap_add():
    enhanced_matrix = np.ones((4, 4))
    frame_length = 4
    overlap = 1
    sig = pyst.dsp.overlap_add(enhanced_matrix, frame_length, overlap)
    expected = np.array([1., 1., 1., 2., 1., 1., 2., 1., 1., 2., 1., 1., 1.])
    assert np.array_equal(expected, sig)
    
def test_overlap_add_framelength_mismatch():
    enhanced_matrix = np.ones((4, 4))
    frame_length = 3
    overlap = 1
    with pytest.raises(TypeError):
        sig = pyst.dsp.overlap_add(enhanced_matrix, 
                                    frame_length, 
                                    overlap)
        

    
