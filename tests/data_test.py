
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pysoundtool as pyst
import numpy as np
import pytest

test_wav = './audio2channels.wav'
test_aiff = './audiodata/traffic.aiff'
test_flac = './audiodata/259672__nooc__this-is-not-right.flac'
test_m4a = './audiodata/505803__skennison__new-recording.m4a'
test_mp3 = './audiodata/244287__kleinhirn2000__toast-glas-langsam.mp3'
test_ogg = './audiodata/240674__zajo__you-have-been-denied.ogg'

def test_loadsound_default_mono():
    samples, sr = pyst.loadsound(test_wav)
    expected = np.array([0.06140351, 0.06140351, 0.06140351, 0.06140351, 0.06140351])
    expected_shape = (len(expected),)
    expected_sr = 16000 # sr of the audiofile (no default)
    assert np.allclose(samples[:5], expected)
    assert expected_shape == samples[:5].shape 
    assert expected_sr == sr
    
def test_loadsound_default_mono_dur1():
    samples, sr = pyst.loadsound(test_wav, dur_sec=1)
    expected = np.array([0.06140351, 0.06140351, 0.06140351, 0.06140351, 0.06140351])
    expected_shape = (len(expected),)
    expected_sr = 16000 # sr of the audiofile (no default)
    assert np.allclose(samples[:5], expected)
    assert expected_shape == samples[:5].shape 
    assert len(samples) == expected_sr
    
def test_loadsound_stereo():
    samples, sr = pyst.loadsound(test_wav, mono=False)
    expected = np.array([[0.06140351, 0.06140351],[0.06140351, 0.06140351],[0.06140351, 0.06140351]])
    expected_shape = expected.shape
    expected_sr = 16000 # sr of the audiofile (no default)
    assert np.allclose(samples[:3], expected)
    assert expected_shape == samples[:3].shape 
    assert expected_sr == sr
    
def test_loadsound_stereo_dur1():
    samples, sr = pyst.loadsound(test_wav, mono=False, dur_sec=1)
    expected = np.array([[0.06140351, 0.06140351],[0.06140351, 0.06140351],[0.06140351, 0.06140351]])
    expected_shape = expected.shape
    expected_sr = 16000 # sr of the audiofile (no default)
    assert np.allclose(samples[:3], expected)
    assert expected_shape == samples[:3].shape 
    assert expected_sr == sr
    assert len(samples) == expected_sr
    
def test_loadsound_mono_sr48000():
    samples, sr = pyst.loadsound(test_wav, mono=True, sr=48000)
    expected = np.array([0.07632732, 0.07633357, 0.07633357, 0.07632732, 0.07632107])
    expected_sr = 48000
    assert np.allclose(samples[:5], expected)
    assert sr == expected_sr
    
def test_loadsound_stereo_sr48000():
    samples, sr = pyst.loadsound(test_wav, sr=48000, mono=False)
    expected = np.array([[0.07632732, 0.07632732],[0.07633357, 0.07628564],[0.07633357, 0.07628563]])
    expected_shape = expected.shape
    expected_sr = 48000 
    assert np.allclose(samples[:3], expected)
    assert expected_shape == samples[:3].shape 
    assert expected_sr == sr
    
def test_loadsound_aiff2wav_sr22050():
    samples, sr = pyst.loadsound(test_aiff, sr=22050)
    assert samples is not None
    
def test_loadsound_flac2wav_sr22050():
    samples, sr = pyst.loadsound(test_flac, sr=22050)
    assert samples is not None
    assert sr == 22050
    
def test_loadsound_m4a2wav_sr22050_error():
    with pytest.raises(RuntimeError):
        samples, sr = pyst.loadsound(test_m4a, sr=22050)
    
def test_loadsound_mp32wav_sr22050_error():
    with pytest.raises(RuntimeError):
        samples, sr = pyst.loadsound(test_mp3, sr=22050)
    
def test_loadsound_ogg2wav_sr22050():
    samples, sr = pyst.loadsound(test_ogg, sr=22050)
    assert samples is not None
    assert sr == 22050
    
def test_loadsound_librosa_wav():
    # use librosa to load file
    samples, sr = pyst.loadsound(test_wav, use_librosa=True)
    # use scipy.io.wavfile to load the file
    samples2, sr2 = pyst.loadsound(test_wav)
    assert np.allclose(samples[:5], np.array([0., 0., 0., 0., 0.]))
    assert sr==16000
    print('Librosa and Scipy.io.wavfile load data a little differently.')
    print('Therefore, slight differences between the data values are expected.')
    assert np.allclose(samples, samples2)
    
def test_loadsound_librosa_wav_dur1_sr22050():
    # use librosa to load file
    samples, sr = pyst.loadsound(test_wav, dur_sec=1, sr=22050, use_librosa=True)

    assert np.allclose(samples[:5], np.array([0., 0., 0., 0., 0.]))
    assert sr==22050
    assert len(samples) == sr
    
def test_loadsound_librosa_wav_dur1_sr22050_stereo():
    # use librosa to load file
    samples, sr = pyst.loadsound(test_wav, mono=False, dur_sec=1, 
                                 sr=22050, use_librosa=True)
    expected = np.array([[0.,0.],[0.,0.],[0.,0.]])
    assert np.allclose(samples[:3], expected)
    assert sr==22050
    assert samples.shape == (22050,2)
    
def test_loadsound_librosa_aiff():
    samples, sr = pyst.loadsound(test_aiff, use_librosa=True)
    expected = np.array([0.09291077, 0.06417847, 0.04179382, 0.02642822, 0.01808167])
    assert np.allclose(samples[:5], expected)
    assert sr==48000
    
def test_loadsound_librosa_aiff_sr16000():
    samples, sr = pyst.loadsound(test_aiff, sr=16000, use_librosa=True)
    expected = np.array([ 0.05152914,0.03653815, -0.0083929, -0.0207656,-0.03038501])
    assert np.allclose(samples[:5], expected)
    assert sr==16000
    
def test_loadsound_librosa_flac():
    samples, sr = pyst.loadsound(test_flac, use_librosa=True)
    expected = np.array([ 0.0000000e+00,0.0000000e+00, 0.0000000e+00, 0.0000000e+00
,-3.0517578e-05])
    assert np.allclose(samples[:5], expected)
    assert sr==44100
    
def test_loadsound_librosa_ogg():
    samples, sr = pyst.loadsound(test_ogg, use_librosa=True)
    expected = np.array([-0.00639889, -0.00722905, -0.00864992, -0.00878596, -0.00894831])
    assert np.allclose(samples[:5], expected)
    assert sr==44100
    
def test_loadsound_librosa_m4a():
    samples, sr = pyst.loadsound(test_m4a, use_librosa=True)
    expected = np.array([0. ,0. ,0. ,0. ,0.])
    assert np.allclose(samples[:5], expected)
    assert sr==48000
    
def test_loadsound_librosa_mp3():
    samples, sr = pyst.loadsound(test_mp3, use_librosa=True)
    expected = np.array([ 0.000e+00, -1.5258789e-05,  0.000e+00,  0.00e+00,
  0.0000000e+00])
    assert np.allclose(samples[:5], expected)
    assert sr==44100
