
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import pytest
import soundfile as sf
import pysoundtool as pyst


###############################################################################

audiodir = 'test_audio/'
example_dir = '{}examps/'.format(audiodir)
example_dir = pyst.utils.check_dir(example_dir, make=True)
test_wav_stereo = '{}audio2channels.wav'.format(audiodir)
test_wav_mono = '{}traffic.wav'.format(audiodir)
test_aiff = '{}traffic.aiff'.format(audiodir)
test_flac = '{}259672__nooc__this-is-not-right.flac'.format(audiodir)
test_m4a = '{}505803__skennison__new-recording.m4a'.format(audiodir)
test_mp3 = '{}244287__kleinhirn2000__toast-glas-langsam.mp3'.format(audiodir)
test_ogg = '{}240674__zajo__you-have-been-denied.ogg'.format(audiodir)

def test_loadsound_mono_uselibrosa_False():
    samples, sr = pyst.loadsound(test_wav_stereo,use_scipy=True)
    expected = np.array([0.06140351, 0.06140351, 0.06140351, 0.06140351,
                         0.06140351])
    expected_shape = (len(expected),)
    expected_sr = 16000 # sr of the audiofile (no default)
    assert np.allclose(samples[:5], expected)
    assert expected_shape == samples[:5].shape 
    assert expected_sr == sr
    
def test_loadsound_mono_dur1_uselibrosa_False():
    samples, sr = pyst.loadsound(test_wav_stereo, dur_sec=1,use_scipy=True)
    expected = np.array([0.06140351, 0.06140351, 0.06140351, 0.06140351,
                         0.06140351])
    expected_shape = (len(expected),)
    expected_sr = 16000 # sr of the audiofile (no default)
    assert np.allclose(samples[:5], expected)
    assert expected_shape == samples[:5].shape 
    assert len(samples) == expected_sr
    
def test_loadsound_stereo_uselibrosa_False():
    samples, sr = pyst.loadsound(test_wav_stereo, mono=False,use_scipy=True)
    expected = np.array([[0.06140351, 0.06140351],[0.06140351, 0.06140351],
                         [0.06140351, 0.06140351]])
    expected_shape = expected.shape
    expected_sr = 16000 # sr of the audiofile (no default)
    assert np.allclose(samples[:3], expected)
    assert expected_shape == samples[:3].shape 
    assert expected_sr == sr
    
def test_loadsound_stereo_dur1_uselibrosa_False():
    samples, sr = pyst.loadsound(test_wav_stereo, mono=False, dur_sec=1,use_scipy=True)
    expected = np.array([[0.06140351, 0.06140351],[0.06140351, 0.06140351],
                         [0.06140351, 0.06140351]])
    expected_shape = expected.shape
    expected_sr = 16000 # sr of the audiofile (no default)
    assert np.allclose(samples[:3], expected)
    assert expected_shape == samples[:3].shape 
    assert expected_sr == sr
    assert len(samples) == expected_sr
    
def test_loadsound_mono_sr48000_uselibrosa_False():
    samples, sr = pyst.loadsound(test_wav_stereo, mono=True, sr=48000,use_scipy=True)
    expected = np.array([0.07632732, 0.07633357, 0.07633357, 0.07632732,
                         0.07632107])
    expected_sr = 48000
    assert np.allclose(samples[:5], expected)
    assert sr == expected_sr
    
def test_loadsound_stereo_sr48000_uselibrosa_False():
    samples, sr = pyst.loadsound(test_wav_stereo, sr=48000, mono=False,use_scipy=True)
    expected = np.array([[0.07632732, 0.07632732],[0.07633357, 0.07628564],
                         [0.07633357, 0.07628563]])
    expected_shape = expected.shape
    expected_sr = 48000 
    assert np.allclose(samples[:3], expected)
    assert expected_shape == samples[:3].shape 
    assert expected_sr == sr
    
def test_loadsound_aiff2wav_sr22050():
    samples, sr = pyst.loadsound(test_aiff, sr=22050)
    assert samples is not None
    
def test_loadsound_flac2wav_sr22050_uselibrosa_False():
    samples, sr = pyst.loadsound(test_flac, sr=22050,use_scipy=True)
    assert samples is not None
    assert sr == 22050
    
def test_loadsound_m4a2wav_sr22050_uselibrosa_False():
    print('IF TEST FAILS: Now loads with librosa if RunTimeError.. is this good?')
    with pytest.raises(RuntimeError):
        samples, sr = pyst.loadsound(test_m4a, sr=22050, use_scipy=True)
    
def test_loadsound_mp32wav_sr22050_error_uselibrosa_False():
    print('IF TEST FAILS: Now loads with librosa if RunTimeError.. is this good?')
    with pytest.raises(RuntimeError):
        samples, sr = pyst.loadsound(test_mp3, sr=22050,use_scipy=True)
    
def test_loadsound_ogg2wav_sr22050_uselibrosa_False():
    samples, sr = pyst.loadsound(test_ogg, sr=22050,use_scipy=True)
    assert samples is not None
    assert sr == 22050
    
def test_loadsound_librosa_wav():
    # use librosa to load file
    samples, sr = pyst.loadsound(test_wav_stereo, use_scipy=False)
    # use scipy.io.wavfile to load the file
    samples2, sr2 = pyst.loadsound(test_wav_stereo)
    assert np.allclose(samples[:5], np.array([0., 0., 0., 0., 0.]))
    assert sr==16000
    print('IF ERROR: Librosa and Scipy.io.wavfile load data a little differently.')
    print('Therefore, slight differences between the data values are expected.')
    assert np.allclose(samples, samples2)
    
def test_loadsound_librosa_sr_None():
    samples, sr = pyst.loadsound(test_wav_stereo, sr=None)
    assert sr == 16000
    
def test_loadsound_scipy_sr_None():
    samples, sr = pyst.loadsound(test_wav_stereo, sr=None, use_scipy=True)
    assert sr == 16000
    
def test_loadsound_librosa_wav_dur1_sr22050():
    # use librosa to load file
    samples, sr = pyst.loadsound(test_wav_stereo, dur_sec=1, sr=22050, use_scipy=False)

    assert np.allclose(samples[:5], np.array([0., 0., 0., 0., 0.]))
    assert sr==22050
    assert len(samples) == sr
    
def test_loadsound_librosa_wav_dur1_sr22050_stereo():
    # use librosa to load file
    samples, sr = pyst.loadsound(test_wav_stereo, mono=False, dur_sec=1, 
                                 sr=22050, use_scipy=False)
    expected = np.array([[0.,0.],[0.,0.],[0.,0.]])
    assert np.allclose(samples[:3], expected)
    assert sr==22050
    assert samples.shape == (22050,2)
    
def test_loadsound_librosa_aiff():
    samples, sr = pyst.loadsound(test_aiff, use_scipy=False)
    expected = np.array([0.09291077, 0.06417847, 0.04179382, 0.02642822, 
                         0.01808167])
    assert np.allclose(samples[:5], expected)
    assert sr==48000
    
def test_loadsound_librosa_aiff_sr16000():
    samples, sr = pyst.loadsound(test_aiff, sr=16000, use_scipy=False)
    expected = np.array([ 0.05152914,0.03653815, -0.0083929,
                         -0.0207656,-0.03038501])
    assert np.allclose(samples[:5], expected)
    assert sr==16000
    
def test_loadsound_librosa_flac():
    samples, sr = pyst.loadsound(test_flac, use_scipy=False)
    expected = np.array([ 0.0000000e+00,0.0000000e+00, 0.0000000e+00,
                         0.0000000e+00,-3.0517578e-05])
    assert np.allclose(samples[:5], expected)
    assert sr==44100
    
def test_loadsound_librosa_ogg():
    samples, sr = pyst.loadsound(test_ogg, use_scipy=False)
    expected = np.array([-0.00639889, -0.00722905, -0.00864992, 
                         -0.00878596, -0.00894831])
    assert np.allclose(samples[:5], expected)
    assert sr==44100
    
def test_loadsound_librosa_m4a():
    samples, sr = pyst.loadsound(test_m4a, use_scipy=False)
    expected = np.array([0. ,0. ,0. ,0. ,0.])
    assert np.allclose(samples[:5], expected)
    assert sr==48000
    
def test_loadsound_librosa_mp3():
    samples, sr = pyst.loadsound(test_mp3, use_scipy=False)
    expected = np.array([ 0.000e+00, -1.5258789e-05,  0.000e+00, 
                         0.00e+00,0.0000000e+00])
    assert np.allclose(samples[:5], expected)
    assert sr==44100
    
def test_savesound_mismatch_format_flac_filename_wav():
    y, sr = pyst.loadsound(test_wav_mono)
    f = pyst.utils.string2pathlib(test_wav_mono)
    format_type = 'FLAC'
    audiofile_new = example_dir.joinpath(f.name)
    audiofile_corrected = pyst.savesound(audiofile_new, y, sr, format=format_type)
    soundobject = sf.SoundFile(audiofile_corrected)
    assert audiofile_corrected.suffix[1:].lower() == format_type.lower() 
    assert soundobject.format == format_type
    os.remove(audiofile_corrected)
    
def test_savesound_filename_wav2flac():
    y, sr = pyst.loadsound(test_wav_mono)
    f = pyst.utils.string2pathlib(test_wav_mono)
    format_type = 'FLAC'
    audiofile_new = example_dir.joinpath(f.stem+'.'+format_type.lower())
    audiofile_corrected = pyst.savesound(audiofile_new, y, sr)
    soundobject = sf.SoundFile(audiofile_corrected)
    assert audiofile_corrected.suffix[1:].lower() == format_type.lower() 
    assert soundobject.format == format_type
    os.remove(audiofile_corrected)
    
def test_savesound_default_FileExistsError():
    y, sr = pyst.loadsound(test_wav_mono)
    with pytest.raises(FileExistsError):
        filename = pyst.savesound(test_wav_mono, y, sr)
        
def test_savesound_default_overwrite():
    y, sr = pyst.loadsound(test_wav_mono)
    soundobject1 = sf.SoundFile(test_wav_mono)
    filename = pyst.savesound(test_wav_mono, y, sr, overwrite=True)
    soundobject2 = sf.SoundFile(filename)
    assert soundobject1.format == soundobject2.format

def test_adjust_shape_last_column():
    desired_shape = (3,3,5)
    input_data = np.ones((3,3,3))
    input_adjusted = pyst.feats.adjust_shape(input_data, 
                                           desired_shape = desired_shape)
    expected = np.array([[[1., 1., 1., 0., 0.],
                          [1., 1., 1., 0., 0.],
                          [1., 1., 1., 0., 0.]],
                        [[1., 1., 1., 0., 0.],
                         [1., 1., 1., 0., 0.],
                         [1., 1., 1., 0., 0.]],
                        [[1., 1., 1., 0., 0.],
                         [1., 1., 1., 0., 0.],
                         [1., 1., 1., 0., 0.]]])
    assert np.array_equal(input_adjusted, expected)
    assert input_adjusted.shape == desired_shape

def test_adjust_feature_shape_several_columns():
    desired_shape = (3,4,5)
    input_data = np.ones((3,3,3))
    input_adjusted = pyst.feats.adjust_shape(input_data, 
                                           desired_shape = desired_shape)
    expected = np.array([[[1., 1., 1., 0., 0.],
                          [1., 1., 1., 0., 0.],
                          [1., 1., 1., 0., 0.],
                          [0., 0., 0., 0., 0.]],
                        [[1., 1., 1., 0., 0.],
                         [1., 1., 1., 0., 0.],
                         [1., 1., 1., 0., 0.],
                         [0., 0., 0., 0., 0.]],
                        [[1., 1., 1., 0., 0.],
                         [1., 1., 1., 0., 0.],
                         [1., 1., 1., 0., 0.],
                         [0., 0., 0., 0., 0.]]])
    assert np.array_equal(input_adjusted, expected)
    assert input_adjusted.shape == desired_shape
    
def test_adjust_feature_shape_smaller():
    desired_shape = (3,3,2)
    input_data = np.ones((3,3,3))
    input_adjusted = pyst.feats.adjust_shape(input_data, 
                                           desired_shape = desired_shape)
    expected = np.array([[[1., 1.],
                          [1., 1.],
                          [1., 1.]],
                        [[1., 1.],
                         [1., 1.],
                         [1., 1.]],
                        [[1., 1.],
                         [1., 1.],
                         [1., 1.]]])
    assert np.array_equal(input_adjusted, expected)
    assert input_adjusted.shape == desired_shape
    
def test_adjust_feature_shape_tuple_smaller():
    desired_shape = (3,2,2)
    input_data = np.ones((3,3,3))
    input_adjusted = pyst.feats.adjust_shape(input_data, 
                                           desired_shape = desired_shape)
    expected = np.array([[[1., 1.],
                          [1., 1.]],
                        [[1., 1.],
                         [1., 1.]],
                        [[1., 1.],
                         [1., 1.]]])
    assert np.array_equal(input_adjusted, expected)
    assert input_adjusted.shape == desired_shape

def test_adjust_feature_shape_mismatch_dims_error():
    desired_shape = (3,2,2)
    input_data = np.ones((3,3,3,3))
    with pytest.raises(ValueError):
        input_adjusted = pyst.feats.adjust_shape(input_data, 
                                            desired_shape = desired_shape)

def test_adjust_feature_shape_2dims():
    desired_shape = (3,4)
    input_data = np.ones((3,3))
    input_adjusted = pyst.feats.adjust_shape(input_data, 
                                            desired_shape = desired_shape)
    expected = np.array([[1., 1., 1., 0.],[1., 1., 1., 0.],[1., 1., 1., 0.]])
    assert np.array_equal(input_adjusted, expected)
    assert input_adjusted.shape == desired_shape 
    
def test_adjust_feature_shape_1dim_zeropad():
    desired_shape = (4,)
    input_data = np.ones((3,))
    input_adjusted = pyst.feats.adjust_shape(input_data, 
                                            desired_shape = desired_shape)
    expected = np.array([1., 1., 1., 0.])
    assert np.array_equal(input_adjusted, expected)
    assert input_adjusted.shape == desired_shape 
    
def test_adjust_feature_shape_1dim_limit():
    desired_shape = (2,)
    input_data = np.ones((3,))
    input_adjusted = pyst.feats.adjust_shape(input_data, 
                                            desired_shape = desired_shape)
    expected = np.array([1., 1.])
    assert np.array_equal(input_adjusted, expected)
    assert input_adjusted.shape == desired_shape 
    
def test_adjust_feature_shape_adding_extra_dimensions():
    desired_shape = (3,2,2,1)
    input_data = np.ones((3,3,3))
    with pytest.raises(ValueError):
        input_adjusted = pyst.feats.adjust_shape(input_data, 
                                            desired_shape = desired_shape)

def test_audio2datasets_seed0_error():
    dict_input = dict([(0,['1.wav','2.wav','3.wav','4.wav','5.wav',
                           '6.wav','7.wav','8.wav','9.wav','10.wav',
                           '11.wav','12.wav','13.wav','14.wav','15.wav',]),
                      (1, ['1.ogg','2.ogg','3.ogg','4.ogg','5.ogg',
                           '6.ogg','7.ogg','8.ogg','9.ogg','10.ogg',
                           '11.ogg','12.ogg','13.ogg','14.ogg','15.ogg',]),
                      (2, ['1.aiff','2.aiff','3.aiff','4.aiff','5.aiff',
                           '6.aiff','7.aiff','8.aiff','9.aiff','10.aiff',
                           '11.aiff','12.aiff','13.aiff','14.aiff',
                           '15.aiff',])])
    with pytest.raises(ValueError):
        dataset_tuple = pyst.datasets.audio2datasets(dict_input, seed=0)

def test_audio2datasets_labeledaudio_dict():
    dict_input = dict([(0,['1.wav','2.wav','3.wav','4.wav','5.wav',
                           '6.wav','7.wav','8.wav','9.wav','10.wav',
                           '11.wav','12.wav','13.wav','14.wav','15.wav',]),
                      (1, ['1.ogg','2.ogg','3.ogg','4.ogg','5.ogg',
                           '6.ogg','7.ogg','8.ogg','9.ogg','10.ogg',
                           '11.ogg','12.ogg','13.ogg','14.ogg','15.ogg',]),
                      (2, ['1.aiff','2.aiff','3.aiff','4.aiff','5.aiff',
                           '6.aiff','7.aiff','8.aiff','9.aiff','10.aiff',
                           '11.aiff','12.aiff','13.aiff','14.aiff',
                           '15.aiff',])])
    dataset_tuple = pyst.datasets.audio2datasets(dict_input, perc_train=0.7, seed=40)
    expected_train = [(1, '7.ogg'), (0, '9.wav'), (1, '11.ogg'), (1, '12.ogg'), 
                      (2, '11.aiff'), (2, '1.aiff'), (2, '7.aiff'), (0, '3.wav'), 
                      (1, '10.ogg'), (1, '13.ogg'), (0, '4.wav'), (2, '3.aiff'), 
                      (2, '5.aiff'), (2, '4.aiff'), (1, '8.ogg'), (2, '12.aiff'), 
                      (2, '9.aiff'), (0, '7.wav'), (2, '8.aiff'), (1, '4.ogg'), 
                      (0, '1.wav'), (1, '9.ogg'), (0, '11.wav'), (1, '1.ogg'), 
                      (0, '12.wav'), (0, '13.wav'), (2, '13.aiff'), (1, '5.ogg'), 
                      (1, '3.ogg'), (0, '5.wav'), (0, '10.wav'), (0, '8.wav'), 
                      (2, '10.aiff')]
    expected_val = [(1, '14.ogg'), (0, '2.wav'), (2, '2.aiff'), (0, '14.wav'), 
                    (2, '14.aiff'), (1, '2.ogg')]
    expected_test = [(1, '6.ogg'), (0, '15.wav'), (2, '15.aiff'), (0, '6.wav'), 
                     (2, '6.aiff'), (1, '15.ogg')]
    for i, dataset in enumerate(dataset_tuple):
        print(i)
        print(dataset)
    assert expected_train == dataset_tuple[0]
    assert expected_val == dataset_tuple[1]
    assert expected_test == dataset_tuple[2]
    
def test_audio2datasets_labeledaudio_loaddict():
    dict_input = dict([(0,['1.wav','2.wav','3.wav','4.wav','5.wav',
                           '6.wav','7.wav','8.wav','9.wav','10.wav',
                           '11.wav','12.wav','13.wav','14.wav','15.wav',]),
                      (1, ['1.ogg','2.ogg','3.ogg','4.ogg','5.ogg',
                           '6.ogg','7.ogg','8.ogg','9.ogg','10.ogg',
                           '11.ogg','12.ogg','13.ogg','14.ogg','15.ogg',]),
                      (2, ['1.aiff','2.aiff','3.aiff','4.aiff','5.aiff',
                           '6.aiff','7.aiff','8.aiff','9.aiff','10.aiff',
                           '11.aiff','12.aiff','13.aiff','14.aiff',
                           '15.aiff',])])
    saved_dict_path = 'testtest.csv'
    if os.path.exists(saved_dict_path):
        os.remove(saved_dict_path)
    saved_dict_path = pyst.utils.save_dict(
        dict2save = dict_input,
        filename = saved_dict_path)
    dataset_tuple = pyst.datasets.audio2datasets(saved_dict_path, perc_train=0.7, seed=40)
    for i, dataset in enumerate(dataset_tuple):
        print(i)
        print(dataset)
    expected_train = [(1, '7.ogg'), (0, '9.wav'), (1, '11.ogg'), (1, '12.ogg'), 
                      (2, '11.aiff'), (2, '1.aiff'), (2, '7.aiff'), (0, '3.wav'), 
                      (1, '10.ogg'), (1, '13.ogg'), (0, '4.wav'), (2, '3.aiff'), 
                      (2, '5.aiff'), (2, '4.aiff'), (1, '8.ogg'), (2, '12.aiff'), 
                      (2, '9.aiff'), (0, '7.wav'), (2, '8.aiff'), (1, '4.ogg'), 
                      (0, '1.wav'), (1, '9.ogg'), (0, '11.wav'), (1, '1.ogg'), 
                      (0, '12.wav'), (0, '13.wav'), (2, '13.aiff'), (1, '5.ogg'), 
                      (1, '3.ogg'), (0, '5.wav'), (0, '10.wav'), (0, '8.wav'), 
                      (2, '10.aiff')]
    expected_val = [(1, '14.ogg'), (0, '2.wav'), (2, '2.aiff'), (0, '14.wav'), 
                    (2, '14.aiff'), (1, '2.ogg')]
    expected_test = [(1, '6.ogg'), (0, '15.wav'), (2, '15.aiff'), (0, '6.wav'), 
                     (2, '6.aiff'), (1, '15.ogg')]
    assert expected_train == dataset_tuple[0]
    assert expected_val == dataset_tuple[1]
    assert expected_test == dataset_tuple[2]
    os.remove(saved_dict_path)
    
def test_audio2datasets_labeledaudio_loaddict_1label():
    dict_input = dict([(0,['1.wav','2.wav','3.wav','4.wav','5.wav',
                           '6.wav','7.wav','8.wav','9.wav','10.wav',
                           '11.wav','12.wav','13.wav','14.wav','15.wav',
                           '1.ogg','2.ogg','3.ogg','4.ogg','5.ogg',
                           '6.ogg','7.ogg','8.ogg','9.ogg','10.ogg',
                           '11.ogg','12.ogg','13.ogg','14.ogg','15.ogg',
                           '1.aiff','2.aiff','3.aiff','4.aiff','5.aiff',
                           '6.aiff','7.aiff','8.aiff','9.aiff','10.aiff',
                           '11.aiff','12.aiff','13.aiff','14.aiff',
                           '15.aiff',])])
    saved_dict_path = 'testtest.csv'
    if os.path.exists(saved_dict_path):
        os.remove(saved_dict_path)
    saved_dict_path = pyst.utils.save_dict(
        dict2save = dict_input,
        filename = saved_dict_path)
    dataset_tuple = pyst.datasets.audio2datasets(saved_dict_path, seed=40)
    for i, dataset in enumerate(dataset_tuple):
        print(i)
        print(dataset)
    expected_train = ['8.ogg', '14.ogg', '1.aiff', '4.ogg', '4.wav', '5.wav', 
                      '5.ogg', '2.wav', '12.aiff', '12.ogg', '11.wav', '8.wav', 
                      '15.wav', '6.aiff', '5.aiff', '15.aiff', '9.wav', '6.wav', 
                      '14.aiff', '8.aiff', '10.aiff', '2.ogg', '14.wav', '13.ogg', 
                      '3.ogg', '1.wav', '3.wav', '12.wav', '6.ogg', '13.wav', 
                      '10.ogg', '3.aiff', '7.wav', '10.wav', '7.aiff', '9.ogg', 
                      '2.aiff']
    expected_val = ['9.aiff', '15.ogg', '13.aiff', '1.ogg']
    expected_test = ['11.ogg', '7.ogg', '4.aiff', '11.aiff']
    assert expected_train == dataset_tuple[0]
    assert expected_val == dataset_tuple[1]
    assert expected_test == dataset_tuple[2]
    os.remove(saved_dict_path)
    
def test_audio2datasets_audio_1label_dict():
    dict_input = dict([(0,['1.wav','2.wav','3.wav','4.wav','5.wav',
                           '6.wav','7.wav','8.wav','9.wav','10.wav',
                           '11.wav','12.wav','13.wav','14.wav','15.wav',
                           '1.ogg','2.ogg','3.ogg','4.ogg','5.ogg',
                           '6.ogg','7.ogg','8.ogg','9.ogg','10.ogg',
                           '11.ogg','12.ogg','13.ogg','14.ogg','15.ogg',
                           '1.aiff','2.aiff','3.aiff','4.aiff','5.aiff',
                           '6.aiff','7.aiff','8.aiff','9.aiff','10.aiff',
                           '11.aiff','12.aiff','13.aiff','14.aiff',
                           '15.aiff',])])
    dataset_tuple = pyst.datasets.audio2datasets(dict_input, seed=40)
    expected_train =['8.ogg', '14.ogg', '1.aiff', '4.ogg', '4.wav', '5.wav', 
                     '5.ogg', '2.wav', '12.aiff', '12.ogg', '11.wav', '8.wav',
                     '15.wav', '6.aiff', '5.aiff', '15.aiff', '9.wav', '6.wav',
                     '14.aiff', '8.aiff', '10.aiff', '2.ogg', '14.wav',
                     '13.ogg', '3.ogg', '1.wav', '3.wav', '12.wav', '6.ogg',
                     '13.wav', '10.ogg', '3.aiff', '7.wav', '10.wav', '7.aiff',
                     '9.ogg', '2.aiff']
    expected_val = ['9.aiff', '15.ogg', '13.aiff', '1.ogg']
    expected_test = ['11.ogg', '7.ogg', '4.aiff', '11.aiff']
    for i, dataset in enumerate(dataset_tuple):
        print(i)
        print(dataset)
    assert expected_train == dataset_tuple[0]
    assert expected_val == dataset_tuple[1]
    assert expected_test == dataset_tuple[2]
    
def test_audio2datasets_audio_list():
    dict_input = ['1.wav','2.wav','3.wav','4.wav','5.wav',
                           '6.wav','7.wav','8.wav','9.wav','10.wav',
                           '11.wav','12.wav','13.wav','14.wav','15.wav',
                           '1.ogg','2.ogg','3.ogg','4.ogg','5.ogg',
                           '6.ogg','7.ogg','8.ogg','9.ogg','10.ogg',
                           '11.ogg','12.ogg','13.ogg','14.ogg','15.ogg',
                           '1.aiff','2.aiff','3.aiff','4.aiff','5.aiff',
                           '6.aiff','7.aiff','8.aiff','9.aiff','10.aiff',
                           '11.aiff','12.aiff','13.aiff','14.aiff','15.aiff']
    dataset_tuple = pyst.datasets.audio2datasets(dict_input, seed=40)
    expected_train = ['8.ogg', '14.ogg', '1.aiff', '4.ogg', '4.wav', '5.wav', 
                      '5.ogg', '2.wav', '12.aiff', '12.ogg', '11.wav', '8.wav',
                      '15.wav', '6.aiff', '5.aiff', '15.aiff', '9.wav',
                      '6.wav', '14.aiff', '8.aiff', '10.aiff', '2.ogg',
                      '14.wav', '13.ogg', '3.ogg', '1.wav', '3.wav', '12.wav',
                      '6.ogg', '13.wav', '10.ogg', '3.aiff', '7.wav', '10.wav',
                      '7.aiff', '9.ogg', '2.aiff']
    for i, dataset in enumerate(dataset_tuple):
        print(i)
        print(dataset)
    expected_val = ['9.aiff', '15.ogg', '13.aiff', '1.ogg']
    expected_test = ['11.ogg', '7.ogg', '4.aiff', '11.aiff']
    assert expected_train == dataset_tuple[0]
    assert expected_val == dataset_tuple[1]
    assert expected_test == dataset_tuple[2]
    
def test_audio2datasets_audio_set():
    dict_input = set(['1.wav','2.wav','3.wav','4.wav','5.wav',
                           '6.wav','7.wav','8.wav','9.wav','10.wav',
                           '11.wav','12.wav','13.wav','14.wav','15.wav',
                           '1.ogg','2.ogg','3.ogg','4.ogg','5.ogg',
                           '6.ogg','7.ogg','8.ogg','9.ogg','10.ogg',
                           '11.ogg','12.ogg','13.ogg','14.ogg','15.ogg',
                           '1.aiff','2.aiff','3.aiff','4.aiff','5.aiff',
                           '6.aiff','7.aiff','8.aiff','9.aiff','10.aiff',
                           '11.aiff','12.aiff','13.aiff','14.aiff','15.aiff'])
    dataset_tuple = pyst.datasets.audio2datasets(dict_input, seed=40)
    expected_train = ['8.ogg', '14.ogg', '1.aiff', '4.ogg', '4.wav', '5.wav',
                      '5.ogg', '2.wav', '12.aiff', '12.ogg', '11.wav', '8.wav',
                      '15.wav', '6.aiff', '5.aiff', '15.aiff', '9.wav',
                      '6.wav', '14.aiff', '8.aiff', '10.aiff', '2.ogg',
                      '14.wav', '13.ogg', '3.ogg', '1.wav', '3.wav', '12.wav',
                      '6.ogg', '13.wav', '10.ogg', '3.aiff', '7.wav', '10.wav',
                      '7.aiff', '9.ogg', '2.aiff']
    expected_val = ['9.aiff', '15.ogg', '13.aiff', '1.ogg']
    expected_test = ['11.ogg', '7.ogg', '4.aiff', '11.aiff']
    for i, dataset in enumerate(dataset_tuple):
        print(i)
        print(dataset)
    assert expected_train == dataset_tuple[0]
    assert expected_val == dataset_tuple[1]
    assert expected_test == dataset_tuple[2]
    
    
def test_audio2datasets_noisy_clean_datasets_match():
    dict_input_clean = ['1_clean.wav','2_clean.wav','3_clean.wav', 
                        '4_clean.wav','5_clean.wav','6_clean.wav',
                        '7_clean.wav','8_clean.wav','9_clean.wav',
                        '10_clean.wav','11_clean.wav','12_clean.wav',
                        '13_clean.wav','14_clean.wav','15_clean.wav',
                           '1_clean.ogg','2_clean.ogg','3_clean.ogg',
                           '4_clean.ogg','5_clean.ogg','6_clean.ogg',
                           '7_clean.ogg','8_clean.ogg','9_clean.ogg',
                           '10_clean.ogg','11_clean.ogg','12_clean.ogg',
                           '13_clean.ogg','14_clean.ogg','15_clean.ogg',
                           '1_clean.aiff','2_clean.aiff','3_clean.aiff',
                           '4_clean.aiff','5_clean.aiff','6_clean.aiff',
                           '7_clean.aiff','8_clean.aiff','9_clean.aiff',
                           '10_clean.aiff','11_clean.aiff','12_clean.aiff',
                           '13_clean.aiff','14_clean.aiff','15_clean.aiff']
    dict_input_noisy = ['1_noisy.wav','2_noisy.wav','3_noisy.wav',
                        '4_noisy.wav','5_noisy.wav','6_noisy.wav',
                        '7_noisy.wav','8_noisy.wav','9_noisy.wav',
                        '10_noisy.wav','11_noisy.wav','12_noisy.wav',
                        '13_noisy.wav','14_noisy.wav','15_noisy.wav',
                        '1_noisy.ogg','2_noisy.ogg','3_noisy.ogg',
                        '4_noisy.ogg','5_noisy.ogg','6_noisy.ogg',
                        '7_noisy.ogg','8_noisy.ogg','9_noisy.ogg',
                        '10_noisy.ogg','11_noisy.ogg','12_noisy.ogg',
                        '13_noisy.ogg','14_noisy.ogg','15_noisy.ogg',
                        '1_noisy.aiff','2_noisy.aiff','3_noisy.aiff',
                        '4_noisy.aiff','5_noisy.aiff','6_noisy.aiff',
                        '7_noisy.aiff','8_noisy.aiff','9_noisy.aiff',
                        '10_noisy.aiff','11_noisy.aiff','12_noisy.aiff',
                        '13_noisy.aiff','14_noisy.aiff','15_noisy.aiff']
    dataset_tuple_clean = pyst.datasets.audio2datasets(dict_input_clean, seed=40)
    dataset_tuple_noisy = pyst.datasets.audio2datasets(dict_input_noisy, seed=40)
    expected_train_clean = ['8_clean.ogg', '15_clean.ogg', '10_clean.aiff', 
                            '4_clean.ogg', '4_clean.wav', '5_clean.wav',
                            '5_clean.ogg', '2_clean.wav', '13_clean.aiff',
                            '13_clean.ogg', '12_clean.wav', '8_clean.wav',
                            '1_clean.wav', '6_clean.aiff', '5_clean.aiff',
                            '1_clean.aiff', '9_clean.wav', '6_clean.wav',
                            '15_clean.aiff', '8_clean.aiff', '11_clean.aiff',
                            '2_clean.ogg', '15_clean.wav', '14_clean.ogg',
                            '3_clean.ogg', '10_clean.wav', '3_clean.wav',
                            '13_clean.wav', '6_clean.ogg', '14_clean.wav',
                            '11_clean.ogg', '3_clean.aiff', '7_clean.wav',
                            '11_clean.wav', '7_clean.aiff', '9_clean.ogg',
                            '2_clean.aiff']
    expected_val_clean = ['9_clean.aiff', '1_clean.ogg', '14_clean.aiff', 
                          '10_clean.ogg']
    expected_test_clean = ['12_clean.ogg', '7_clean.ogg', '4_clean.aiff', 
                           '12_clean.aiff']
    expected_train_noisy = ['8_noisy.ogg', '15_noisy.ogg', '10_noisy.aiff',
                            '4_noisy.ogg', '4_noisy.wav', '5_noisy.wav',
                            '5_noisy.ogg', '2_noisy.wav', '13_noisy.aiff',
                            '13_noisy.ogg', '12_noisy.wav', '8_noisy.wav',
                            '1_noisy.wav', '6_noisy.aiff', '5_noisy.aiff',
                            '1_noisy.aiff', '9_noisy.wav', '6_noisy.wav',
                            '15_noisy.aiff', '8_noisy.aiff', '11_noisy.aiff', 
                            '2_noisy.ogg', '15_noisy.wav', '14_noisy.ogg',
                            '3_noisy.ogg', '10_noisy.wav', '3_noisy.wav',
                            '13_noisy.wav', '6_noisy.ogg', '14_noisy.wav',
                            '11_noisy.ogg', '3_noisy.aiff', '7_noisy.wav',
                            '11_noisy.wav', '7_noisy.aiff', '9_noisy.ogg',
                            '2_noisy.aiff']
    expected_val_noisy = ['9_noisy.aiff', '1_noisy.ogg', '14_noisy.aiff',
                          '10_noisy.ogg']
    expected_test_noisy = ['12_noisy.ogg', '7_noisy.ogg', '4_noisy.aiff',
                           '12_noisy.aiff']
    for i, dataset in enumerate(dataset_tuple_clean):
        print(i)
        print(dataset)
    for i, dataset in enumerate(dataset_tuple_noisy):
        print(i)
        print(dataset)
    assert expected_train_clean == dataset_tuple_clean[0]
    assert expected_val_clean == dataset_tuple_clean[1]
    assert expected_test_clean == dataset_tuple_clean[2]
    assert expected_train_noisy == dataset_tuple_noisy[0]
    assert expected_val_noisy == dataset_tuple_noisy[1]
    assert expected_test_noisy == dataset_tuple_noisy[2]
    

