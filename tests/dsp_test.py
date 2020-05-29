
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
        
def test_generate_sound_default():
    data, sr = pyst.dsp.generate_sound()
    expected1 = np.array([0., 0.06260483, 0.12366658, 0.18168021, 0.2352158 ])
    expected2 = 2000
    expected3 = 8000
    assert np.allclose(expected1, data[:5])
    assert len(data) == expected2
    assert sr == expected3
    
def test_generate_sound_freq5():
    sound, sr = pyst.dsp.generate_sound(freq=5, amplitude=0.5, sr=5, dur_sec=1)
    expected1 = np.array([ 0.000000e+00,  5.000000e-01,  
                         3.061617e-16, -5.000000e-01, -6.123234e-16])
    expected_sr = 5
    expected_len = expected_sr * 1
    assert np.allclose(expected1, sound)
    assert sr == expected_sr
    assert len(sound) == expected_len
    
def test_get_time_points():
    time = pyst.dsp.get_time_points(dur_sec = 0.1, sr=50)
    expected = np.array([0.    ,0.025 ,0.05 , 0.075, 0.1  ])
    assert np.allclose(time, expected)
    
def test_generate_noise():
    noise = pyst.dsp.generate_noise(5, random_seed=0)
    expected = np.array([0.04410131, 0.01000393, 0.02446845, 0.05602233, 0.04668895])
    assert np.allclose(expected, noise)
    
def test_set_signal_length_longer():
    input_samples = np.array([1,2,3,4,5])
    samples = pyst.dsp.set_signal_length(input_samples, numsamps = 8)
    expected = np.array([1,2,3,4,5,0,0,0])
    assert len(samples) == 8
    assert np.array_equal(samples, expected)
    
def test_set_signal_length_shorter():
    input_samples = np.array([1,2,3,4,5])
    samples = pyst.dsp.set_signal_length(input_samples, numsamps = 4)
    expected = np.array([1,2,3,4])
    assert len(samples) == 4
    assert np.array_equal(samples, expected)
    
def test_scalesound_default():
    np.random.seed(0)
    input_samples = np.random.random_sample((5,))
    output_samples = pyst.dsp.scalesound(input_samples)
    expected = np.array([-0.14138 ,1., 0.22872961, -0.16834299, -1.])
    assert np.allclose(output_samples, expected)
    
def test_scalesound_min_minus100_max100():
    np.random.seed(0)
    input_samples = np.random.random_sample((5,))
    output_samples = pyst.dsp.scalesound(input_samples, min_val=-100, max_val=100)
    expected = np.array([ -14.13800026,100., 22.87296052,-16.83429866,-100.])
    assert np.allclose(output_samples, expected)
    
def test_scalesound_min_minuspoint25_maxpoint25():
    np.random.seed(0)
    input_samples = np.random.random_sample((5,))
    output_samples = pyst.dsp.scalesound(input_samples, min_val=-.25, max_val=.25)
    expected = np.array([-0.035345, 0.25, 0.0571824, -0.04208575, -0.25])
    assert np.allclose(output_samples, expected)
