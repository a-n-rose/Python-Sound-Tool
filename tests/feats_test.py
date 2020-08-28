
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import soundpy as sp
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
        X, y = sp.feats.separate_dependent_var(data)
    
def test_separate_dependent_var_3d():
    data = np.array(range(12)).reshape(2,2,3)
    X, y = sp.feats.separate_dependent_var(data)
    expected1 = np.array([[[ 0,  1],[ 3,  4]],[[ 6,  7],[ 9, 10]]])
    expected2 = np.array([2, 8])
    assert np.array_equal(expected1, X)
    assert np.array_equal(expected2, y)
    
def test_separate_dependent_var_3d_1feature_valueerror():
    data = np.array(range(12)).reshape(2,6,1)
    with pytest.raises(ValueError):
        X, y = sp.feats.separate_dependent_var(data)
        
def test_separate_dependent_var_3d_2feats():
    data = np.array(range(12)).reshape(2,3,2)
    X, y = sp.feats.separate_dependent_var(data)
    expected1 = np.array([[[ 0],[ 2],[ 4]],[[ 6],[ 8],[10]]])
    expected2 = np.array([1, 7])
    assert np.array_equal(expected1, X)
    assert np.array_equal(expected2, y)
    
def test_scale_X_y_3d_train():
    np.random.seed(seed=0)
    data = np.random.random_sample(size=(2,1,3))
    X, y, scalars = sp.feats.scale_X_y(data)
    expected1 = np.array([[[[ 1.],[ 1.]]], [[[-1.],[-1.]]]])
    expected2 = np.array([[0.60276338],[0.64589411]])
    assert np.allclose(expected1, X)
    assert np.allclose(expected2, y)
    assert isinstance(scalars, dict)
    
def test_scale_X_y_2d_train():
    np.random.seed(seed=0)
    data = np.random.random_sample(size=(2,3))
    with pytest.raises(ValueError):
        X, y, scalars = sp.feats.scale_X_y(data)
        
def test_scale_X_y_3d_test():
    np.random.seed(seed=0)
    data1 = np.random.random_sample(size=(2,1,3))
    np.random.seed(seed=40)
    data2 = np.random.random_sample(size=(2,1,3))
    X, y, scalars = sp.feats.scale_X_y(data1)
    X, y, scalars = sp.feats.scale_X_y(data2, is_train=False, 
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
    X, y, scalars = sp.feats.scale_X_y(data1)
    with pytest.raises(TypeError):
        X, y, scalars = sp.feats.scale_X_y(data2, is_train=False)
        
def test_scale_X_y_3d_train_1feature_valueerror():
    np.random.seed(seed=0)
    data = np.random.random_sample(size=(2,3,1))
    with pytest.raises(ValueError):
        X, y, scalars = sp.feats.scale_X_y(data)
        
def test_get_vad_samples_stft_consistency():
    sr = 48000
    win_size_ms = 50
    percent_overlap = 0
    snr = 20
    speech, sr = sp.loadsound(test_python, sr = sr)
    noise = sp.generate_noise(len(speech), random_seed = 40)
    speech_snr, snr_measured = sp.dsp.add_backgroundsound(speech,
                                                noise,
                                                sr = sr,
                                                snr = snr,
                                                delay_mainsound_sec = 1,
                                                total_len_sec = 3,
                                                random_seed = 40)
    vad_samples, vad_matrix1 = sp.feats.get_vad_samples(speech_snr, 
                                           sr = sr,
                                           win_size_ms = win_size_ms,
                                           percent_overlap = percent_overlap)
    vad_stft, vad_matrix2 = sp.feats.get_vad_stft(speech_snr, 
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
    speech, sr = sp.loadsound(test_python, sr = sr, remove_dc=False)
    # get just speech segment, no surrounding silence:
    # goal VAD
    speech_only, sr = sp.loadsound('{}python_speech_only.wav'.format(test_dir), 
                                     sr = sr, remove_dc=False)
    noise = sp.generate_noise(len(speech), random_seed = 40)
    speech_snr, snr_measured = sp.dsp.add_backgroundsound(speech,
                                                noise,
                                                sr = sr,
                                                snr = snr,
                                                delay_mainsound_sec = 1,
                                                total_len_sec = 3,
                                                random_seed = snr,
                                                remove_dc = False)
    vad_samples, vad_vector = sp.feats.get_vad_samples(speech_snr, 
                                           sr = sr,
                                           win_size_ms = win_size_ms,
                                           percent_overlap = percent_overlap)
    vad_samples_ms = len(vad_samples) / sr / 0.001
    example_speech_ms = len(speech_only) / sr / 0.001
    if example_speech_ms >= vad_samples_ms:
        assert (example_speech_ms / vad_samples_ms) < 2
    elif example_speech_ms <= vad_samples_ms:
        assert (vad_samples_ms / example_speech_ms) < 2
        
def test_get_samples_clipped_40SNR_50percentVAD_length_threshold():
    sr = 48000
    win_size_ms = 50
    percent_overlap = 0
    snr = 40
    speech, sr = sp.loadsound(test_python, sr = sr, remove_dc=False)
    # get just speech segment, no surrounding silence:
    # goal VAD
    speech_only, sr = sp.loadsound('{}python_speech_only.wav'.format(test_dir), 
                                     sr = sr, remove_dc=False)
    noise = sp.generate_noise(len(speech), random_seed = 40)
    speech_snr, snr_measured = sp.dsp.add_backgroundsound(speech,
                                                noise,
                                                sr = sr,
                                                snr = snr,
                                                delay_mainsound_sec = 1,
                                                total_len_sec = 3,
                                                random_seed = snr,
                                                remove_dc = False)
    vad_samples, vad_vector = sp.feats.get_samples_clipped(speech_snr, 
                                           sr = sr,
                                           win_size_ms = win_size_ms,
                                           percent_overlap = percent_overlap)
    vad_samples_ms = len(vad_samples) / sr / 0.001
    example_speech_ms = len(speech_only) / sr / 0.001
    if example_speech_ms >= vad_samples_ms:
        assert (example_speech_ms / vad_samples_ms) < 2
    elif example_speech_ms <= vad_samples_ms:
        assert (vad_samples_ms / example_speech_ms) < 2
        
def test_apply_new_subframe_defaults():
    matrix_input = np.arange(15).reshape(5,3,1)
    got = sp.feats.apply_new_subframe(
        matrix_input, 
        new_frame_size = 2)
    expected_shape = (2, 3, 3, 1)
    expected = np.array([[[[ 0],
         [ 1],
         [ 2]],

        [[ 3],
         [ 4],
         [ 5]],

        [[ 6],
         [ 7],
         [ 8]]],


       [[[ 9],
         [10],
         [11]],

        [[12],
         [13],
         [14]],

        [[ 0],
         [ 0],
         [ 0]]]])
    assert expected_shape == got.shape
    assert np.array_equal(expected, got)


def test_apply_new_subframe_axis1():
    matrix_input = np.arange(15).reshape(1,5,3,1)
    got = sp.feats.apply_new_subframe(
        matrix_input, 
        new_frame_size = 2,
        axis = 1)
    expected_shape = (1, 2, 3, 3, 1)
    expected = np.array([[[[[ 0],
          [ 1],
          [ 2]],

         [[ 3],
          [ 4],
          [ 5]],

         [[ 6],
          [ 7],
          [ 8]]],


        [[[ 9],
          [10],
          [11]],

         [[12],
          [13],
          [14]],

         [[ 0],
          [ 0],
          [ 0]]]]])
    assert expected_shape == got.shape
    assert np.array_equal(expected, got)


def test_apply_new_subframe_no_zeropad():
    matrix_input = np.arange(15).reshape(1,5,3,1)
    got = sp.feats.apply_new_subframe(
        matrix_input, 
        new_frame_size = 2,
        axis = 1,
        zeropad = False)
    expected_shape = (1, 2, 2, 3, 1)
    expected = np.array([[[[[ 0],
          [ 1],
          [ 2]],

         [[ 3],
          [ 4],
          [ 5]]],


        [[[ 6],
          [ 7],
          [ 8]],

         [[ 9],
          [10],
          [11]]]]])
    assert expected_shape == got.shape
    assert np.array_equal(expected, got)
    
def test_apply_new_subframe_last_axis():
    matrix_input = np.arange(15).reshape(5,3)
    got = sp.feats.apply_new_subframe(
            matrix_input, 
            new_frame_size = 2,
            axis = 1,
            zeropad = True)
    expected_shape = (5, 2, 2)
    expected = np.array([[[ 0,  1],
        [ 2,  0]],

       [[ 3,  4],
        [ 5,  0]],

       [[ 6,  7],
        [ 8,  0]],

       [[ 9, 10],
        [11,  0]],

       [[12, 13],
        [14,  0]]])
    assert expected_shape == got.shape
    assert np.array_equal(expected, got)
    
def test_apply_new_subframe_last_axis_no_zeropad():
    matrix_input = np.arange(15).reshape(5,3)
    got = sp.feats.apply_new_subframe(
            matrix_input, 
            new_frame_size = 2,
            axis = 1,
            zeropad = False)
    expected_shape = (5, 2)
    expected = np.array([[ 0,  1],
       [ 3,  4],
       [ 6,  7],
       [ 9, 10],
       [12, 13]])
    assert expected_shape == got.shape
    assert np.array_equal(expected, got)
    
def test_apply_new_subframe_2D_negative_axis():
    matrix_input = np.arange(15).reshape(3,5)
    got = sp.feats.apply_new_subframe(
        matrix_input, 
        new_frame_size = 2,
        axis = -2,
        zeropad = False)
    expected_shape = (2, 5)
    expected = np.array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
    got2 = sp.feats.apply_new_subframe(
        matrix_input, 
        new_frame_size = 2,
        axis = 0,
        zeropad = False)
    assert expected_shape == got.shape
    assert np.array_equal(expected, got)
    assert np.array_equal(got, got2)
    
def test_apply_new_subframe_4D_axis3_zeropad():
    matrix_input = np.arange(32).reshape(2,2,4,2)
    got = sp.feats.apply_new_subframe(
        matrix_input, 
        new_frame_size = 3,
        axis = -2,
        zeropad = True)
    expected_shape = (2, 2, 3, 2, 2)
    expected = np.array([[[[[ 0,  1],
          [ 2,  3]],

         [[ 4,  5],
          [ 6,  7]],

         [[ 0,  0],
          [ 0,  0]]],


        [[[ 8,  9],
          [10, 11]],

         [[12, 13],
          [14, 15]],

         [[ 0,  0],
          [ 0,  0]]]],



       [[[[16, 17],
          [18, 19]],

         [[20, 21],
          [22, 23]],

         [[ 0,  0],
          [ 0,  0]]],


        [[[24, 25],
          [26, 27]],

         [[28, 29],
          [30, 31]],

         [[ 0,  0],
          [ 0,  0]]]]])
    got2 = sp.feats.apply_new_subframe(
        matrix_input, 
        new_frame_size = 3,
        axis = 2,
        zeropad = True)
    assert expected_shape == got.shape
    assert np.array_equal(expected, got)
    assert np.array_equal(got, got2)


def test_apply_new_subframe_4D_axis3_NO_zeropad():
    matrix_input = np.arange(32).reshape(2,2,4,2)
    got = sp.feats.apply_new_subframe(
        matrix_input, 
        new_frame_size = 3,
        axis = -2,
        zeropad = False)
    expected_shape = (2, 2, 3, 2)
    expected = np.array([[[[ 0,  1],
         [ 2,  3],
         [ 4,  5]],

        [[ 8,  9],
         [10, 11],
         [12, 13]]],


       [[[16, 17],
         [18, 19],
         [20, 21]],

        [[24, 25],
         [26, 27],
         [28, 29]]]])
    got2 = sp.feats.apply_new_subframe(
        matrix_input, 
        new_frame_size = 3,
        axis = 2,
        zeropad = False)
    assert expected_shape == got.shape
    assert np.array_equal(expected, got)
    assert np.array_equal(got, got2)
    
    
def test_apply_new_subframe_6D_valueerror():
    matrix_input = np.arange(160).reshape(2,2,5,2,2,2)
    with pytest.raises(ValueError):
        sp.feats.apply_new_subframe(matrix_input, 
                                      new_frame_size = 2)
        
def test_apply_new_subframe_1D_valueerror():
    matrix_input = np.array(5)
    with pytest.raises(ValueError):
        sp.feats.apply_new_subframe(matrix_input, 
                                      new_frame_size = 2)
    
def test_get_feature_matrix_shape():
    feature = 'signal'
    sr = 22050
    dur_sec = 1
    win_size_ms = 20
    percent_overlap = 0.5
    labeled_data = False
    got_base_shape, got_model_shape = sp.feats.get_feature_matrix_shape(
        feature_type = feature,
        sr = sr,
        dur_sec = dur_sec,
        win_size_ms = win_size_ms,
        percent_overlap = percent_overlap,
        labeled_data = labeled_data)
    expected_base_shape = (22050,)
    expected_model_shape = (50, 441)
    assert expected_base_shape == got_base_shape
    assert expected_model_shape == got_model_shape
    
