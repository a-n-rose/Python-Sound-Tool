import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import pytest
import librosa
import pathlib
import soundpy as sp

audio_dir = 'test_audio/'
test_audiofile = '{}audio2channels.wav'.format(audio_dir)



def test_path_or_samples_str():
    item_type = sp.utils.path_or_samples(test_audiofile)
    assert item_type == 'path'
    
def test_path_or_samples_pathlib():
    item_type = sp.utils.path_or_samples(pathlib.Path(test_audiofile))
    assert item_type == 'path'
    
def test_path_or_samples_tuple_librosa():
    item = librosa.load(test_audiofile)
    item_type = sp.utils.path_or_samples(item)
    assert item_type == 'samples'
    
def test_path_or_samples_tuple_not_real_samples():
    item = (np.ndarray([1,2,3]), 4)
    item_type = sp.utils.path_or_samples(item)
    assert item_type == 'samples'
    
def test_path_or_samples_str_not_real_path():
    print('IF TEST FAILES: For now, function does not test for path validity.')
    with pytest.raises(ValueError):
        item_type = sp.utils.path_or_samples('blah')
    
def test_path_or_samples_pathlib_not_real_path():
    print('IF TEST FAILES: For now, function does not test for path validity.')
    with pytest.raises(ValueError):
        item_type = sp.utils.path_or_samples(pathlib.Path('blah'))
    
def test_match_dtype_float2int():
    array_original = np.array([1,2,3,4])
    array_to_change = np.array([1.,2.,3.,4.,5.])
    array_adjusted = sp.utils.match_dtype(array_to_change, array_original)
    assert array_original.dtype == array_adjusted.dtype
    assert len(array_to_change) == len(array_adjusted)
    assert np.array_equal(array_to_change, array_adjusted)
    assert array_to_change.dtype != array_original.dtype
    
def test_match_dtype_int2float():
    array_original = np.array([1.,2.,3.,4.])
    array_to_change = np.array([1,2,3,4,5])
    array_adjusted = sp.utils.match_dtype(array_to_change, array_original)
    assert array_original.dtype == array_adjusted.dtype
    assert len(array_to_change) == len(array_adjusted)
    assert np.array_equal(array_to_change, array_adjusted)
    assert array_to_change.dtype != array_original.dtype
    
def test_shape_samps_channels_too_many_dimensions():
    input_data = np.array([1,2,3,4,5,6,7,8,9,10,11,12]).reshape(2,3,2)
    with pytest.raises(ValueError):
        output_data = sp.dsp.shape_samps_channels(input_data)

def test_check_dir_default_create():
    test_dir = './testtesttest/'
    test_dir = sp.utils.check_dir(test_dir)
    assert isinstance(test_dir, pathlib.PosixPath)
    assert os.path.exists(test_dir)
    os.rmdir(test_dir)
    
def test_check_dir_check_exists():
    test_dir = './testtesttest/'
    test_dir = sp.utils.check_dir(test_dir, make=True)
    test_dir = sp.utils.check_dir(test_dir, make=False)
    assert isinstance(test_dir, pathlib.PosixPath)
    assert os.path.exists(test_dir)
    os.rmdir(test_dir)
    
def test_check_dir_check_exists_raiseerror():
    test_dir = './testtesttest/'
    with pytest.raises(FileNotFoundError):
        test_dir = sp.utils.check_dir(test_dir, make=False)
    
def test_check_dir_check_exists_notwriteinto_raiseerror():
    test_dir = './testtesttest/'
    test_dir = sp.utils.check_dir(test_dir, make=True)
    with pytest.raises(FileExistsError):
        test_dir = sp.utils.check_dir(test_dir, make=False, append=False)
    os.rmdir(test_dir)
    
def test_check_dir_pathwithextension_raiseerror():
    test_dir = './testtesttest.py/'
    with pytest.raises(TypeError):
        test_dir = sp.utils.check_dir(test_dir, make=False)
        
def test_string2list():
    audiofiles = sp.files.collect_audiofiles(audio_dir,wav_only=False,
                                         recursive=False)
    audiofiles_string = str(audiofiles)
    audiofiles_checked = sp.utils.restore_dictvalue(audiofiles_string)
    assert audiofiles ==  audiofiles_checked
    
def test_string2list_loaddict():
    audiofiles = sp.files.collect_audiofiles(audio_dir,wav_only=False,
                                         recursive=False)
    d = dict([(0,audiofiles)])
    test_dict_path = 'testest.csv'
    if os.path.exists(test_dict_path):
        os.remove(test_dict_path)
    d_path = sp.utils.save_dict(
        dict2save = d, 
        filename = test_dict_path)
    d_loaded = sp.utils.load_dict(d_path)
    for i, key in enumerate(d_loaded):
        key = key
    audiofiles_string = d_loaded[key]
    audiofiles_checked = sp.utils.restore_dictvalue(audiofiles_string)
    assert audiofiles ==  audiofiles_checked
    os.remove(test_dict_path)
    
def test_restore_dictvalue_list_of_tuples():
    pass

def test_restore_dictvalue_regular_string():
    expected = 'hann'
    got = sp.utils.restore_dictvalue(expected)
    assert expected == got

def test_restore_dictvalue_None():
    expected = None 
    string_val = str(expected)
    got = sp.utils.restore_dictvalue(string_val)
    assert expected == got

def test_restore_dictvalue_True():
    expected = True
    string_val = str(expected)
    got = sp.utils.restore_dictvalue(string_val)
    assert expected == got
    
def test_restore_dictvalue_False():
    expected = False 
    string_val = str(expected)
    got = sp.utils.restore_dictvalue(string_val)
    assert expected == got

def test_restore_dictvalue_int():
    expected = 1
    string_val = str(expected)
    got = sp.utils.restore_dictvalue(string_val)
    assert expected == got

def test_restore_dictvalue_float():
    expected = 1.0
    string_val = str(expected)
    got = sp.utils.restore_dictvalue(string_val)
    assert expected == got

def test_restore_dictvalue_tuple():
    expected = (3,4)
    string_val = str(expected)
    got = sp.utils.restore_dictvalue(string_val)
    assert expected == got
    
def test_restore_dictvalue_list_of_pathwaystrings():
    expected = ['audio1.wav','audio2.wav','audio3.wav']
    string_list = str(expected)
    got = sp.utils.restore_dictvalue(string_list)
    assert expected == got
    
def test_restore_dictvalue_list_of_pathlib_ojbect_strings():
    expected = [pathlib.Path('audio1.wav'),pathlib.Path('audio2.wav'),pathlib.Path('audio3.wav')]
    string_list = str(expected)
    got = sp.utils.restore_dictvalue(string_list)
    assert expected == got
    
def test_restore_dictvalue_list_of_pathwaystrings_nested():
    expected = [['audio1.wav','audio2.wav'],['audio3.wav']]
    string_list = str(expected)
    got = sp.utils.restore_dictvalue(string_list)
    assert expected == got
    
def test_restore_dictvalue_list_of_pathlib_ojbect_strings_nested():
    expected = [[pathlib.Path('audio1.wav'),pathlib.Path('audio2.wav')],[pathlib.Path('audio3.wav')]]
    string_list = str(expected)
    with pytest.raises(ValueError):
        got = sp.utils.restore_dictvalue(string_list)
    
def test_restore_dictvalue_tuple_labeledpaths():
    expected = [(1, 'audio1.wav'),(2, 'audio2.wav'),(3, 'audio3.wav')]
    string_list = str(expected)
    got = sp.utils.restore_dictvalue(string_list)
    assert expected == got
    
def test_restore_dictvalue_tuple_labeled_pathlibojbects():
    expected = [(1, pathlib.Path('audio1.wav')),(2, pathlib.Path('audio2.wav')),(3, pathlib.Path('audio3.wav'))]
    string_list = str(expected)
    got = sp.utils.restore_dictvalue(string_list)
    assert expected == got
