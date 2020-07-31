
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pysoundtool as pyso
import numpy as np
import pytest

test_dir = 'test_audio/'
test_audiofile = '{}audio2channels.wav'.format(test_dir)
test_traffic = '{}traffic.wav'.format(test_dir)
test_python = '{}python.wav'.format(test_dir)
test_horn = '{}car_horn.wav'.format(test_dir)
        
def test_separate_dependent_var_2d():
    data = np.array(range(18)).reshape(-1,1)
    with pytest.raises(ValueError):
        X, y = pyso.feats.separate_dependent_var(data)
    
    
def test_separate_dependent_var_3d():
    data = np.array(range(12)).reshape(2,2,3)
    X, y = pyso.feats.separate_dependent_var(data)
    expected1 = np.array([[[ 0,  1],[ 3,  4]],[[ 6,  7],[ 9, 10]]])
    expected2 = np.array([2, 8])
    assert np.array_equal(expected1, X)
    assert np.array_equal(expected2, y)
    
def test_separate_dependent_var_3d_1feature_valueerror():
    data = np.array(range(12)).reshape(2,6,1)
    with pytest.raises(ValueError):
        X, y = pyso.feats.separate_dependent_var(data)
        
def test_separate_dependent_var_3d_2feats():
    data = np.array(range(12)).reshape(2,3,2)
    X, y = pyso.feats.separate_dependent_var(data)
    expected1 = np.array([[[ 0],[ 2],[ 4]],[[ 6],[ 8],[10]]])
    expected2 = np.array([1, 7])
    assert np.array_equal(expected1, X)
    assert np.array_equal(expected2, y)
    
def test_scale_X_y_3d_train():
    np.random.seed(seed=0)
    data = np.random.random_sample(size=(2,1,3))
    X, y, scalars = pyso.feats.scale_X_y(data)
    expected1 = np.array([[[[ 1.],[ 1.]]], [[[-1.],[-1.]]]])
    expected2 = np.array([[0.60276338],[0.64589411]])
    assert np.allclose(expected1, X)
    assert np.allclose(expected2, y)
    assert isinstance(scalars, dict)
    
def test_scale_X_y_2d_train():
    np.random.seed(seed=0)
    data = np.random.random_sample(size=(2,3))
    with pytest.raises(ValueError):
        X, y, scalars = pyso.feats.scale_X_y(data)
        
def test_scale_X_y_3d_test():
    np.random.seed(seed=0)
    data1 = np.random.random_sample(size=(2,1,3))
    np.random.seed(seed=40)
    data2 = np.random.random_sample(size=(2,1,3))
    X, y, scalars = pyso.feats.scale_X_y(data1)
    X, y, scalars = pyso.feats.scale_X_y(data2, is_train=False, 
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
    X, y, scalars = pyso.feats.scale_X_y(data1)
    with pytest.raises(TypeError):
        X, y, scalars = pyso.feats.scale_X_y(data2, is_train=False)
        
def test_scale_X_y_3d_train_1feature_valueerror():
    np.random.seed(seed=0)
    data = np.random.random_sample(size=(2,3,1))
    with pytest.raises(ValueError):
        X, y, scalars = pyso.feats.scale_X_y(data)
        
def test_get_vad_samples_stft_consistency():
    sr = 48000
    win_size_ms = 50
    percent_overlap = 0
    snr = 20
    speech, sr = pyso.loadsound(test_python, sr = sr)
    noise = pyso.generate_noise(len(speech), random_seed = 40)
    speech_snr, snr_measured = pyso.dsp.add_backgroundsound(speech,
                                                noise,
                                                sr = sr,
                                                snr = snr,
                                                delay_mainsound_sec = 1,
                                                total_len_sec = 3,
                                                random_seed = 40)
    vad_samples, sr = pyso.feats.get_vad_samples(speech_snr, 
                                           sr = sr,
                                           win_size_ms = win_size_ms,
                                           percent_overlap = percent_overlap)
    vad_stft, sr = pyso.feats.get_vad_stft(speech_snr, 
                                     sr = sr,
                                     win_size_ms = win_size_ms,
                                     percent_overlap = percent_overlap)
    vad_samples_ms = len(vad_samples) / sr / 0.001
    vad_stft_ms = len(vad_stft) * win_size_ms
    assert vad_samples_ms == vad_stft_ms
    
def test_get_vad_samples_40SNR_50percentVAD_length_threshold():
    sr = 48000
    win_size_ms = 50
    percent_overlap = 0
    snr = 40
    speech, sr = pyso.loadsound(test_python, sr = sr, remove_dc=False)
    # get just speech segment, no surrounding silence:
    # goal VAD
    speech_only, sr = pyso.loadsound('{}python_speech_only.wav'.format(test_dir), 
                                     sr = sr, remove_dc=False)
    noise = pyso.generate_noise(len(speech), random_seed = 40)
    speech_snr, snr_measured = pyso.dsp.add_backgroundsound(speech,
                                                noise,
                                                sr = sr,
                                                snr = snr,
                                                delay_mainsound_sec = 1,
                                                total_len_sec = 3,
                                                random_seed = snr,
                                                remove_dc = False)
    vad_samples, sr = pyso.feats.get_vad_samples(speech_snr, 
                                           sr = sr,
                                           win_size_ms = win_size_ms,
                                           percent_overlap = percent_overlap)
    vad_samples_ms = len(vad_samples) / sr / 0.001
    example_speech_ms = len(speech_only) / sr / 0.001
    if example_speech_ms >= vad_samples_ms:
        assert (example_speech_ms / vad_samples_ms) < 2
    elif example_speech_ms <= vad_samples_ms:
        assert (vad_samples_ms / example_speech_ms) < 2
