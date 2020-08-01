
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import pytest
import librosa
import pysoundtool as pyso


test_dir = 'test_audio/'
test_audiofile = '{}audio2channels.wav'.format(test_dir)
test_traffic = '{}traffic.wav'.format(test_dir)
test_python = '{}python.wav'.format(test_dir)
test_horn = '{}car_horn.wav'.format(test_dir)

samples_48000, sr_48000 = librosa.load(test_audiofile, sr=48000)
samples_44100, sr_44100 = librosa.load(test_audiofile, sr=44100)
samples_22050, sr_22050 = librosa.load(test_audiofile, sr=22050)
samples_16000, sr_16000 = librosa.load(test_audiofile, sr=16000)
samples_8000, sr_8000 = librosa.load(test_audiofile, sr=8000)


def test_shape_samps_channels_mono():
    input_data = np.array([1,2,3,4,5])
    output_data = pyso.dsp.shape_samps_channels(input_data)
    assert np.array_equal(input_data, output_data)

def test_shape_samps_channels_stereo_correct():
    input_data = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(5,2)
    output_data = pyso.dsp.shape_samps_channels(input_data)
    assert np.array_equal(input_data, output_data)

def test_shape_samps_channels_stereo_incorrect():
    input_data = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(2,5)
    output_data = pyso.dsp.shape_samps_channels(input_data)
    assert np.array_equal(input_data.T, output_data)

def test_calc_phase():
    np.random.seed(seed=0)
    rand_fft = np.random.random(2) + np.random.random(2) * 1j
    phase = pyso.dsp.calc_phase(rand_fft)
    value1 = np.array([0.67324134+0.73942281j, 0.79544405+0.60602703j])
    assert np.allclose(value1, phase)
    
def test_calc_phase_framelength10_default():
    frame_length = 10
    time = np.arange(0, 10, 0.1)
    signal = np.sin(time)[:frame_length]
    fft_vals = np.fft.fft(signal)
    phase = pyso.dsp.calc_phase(fft_vals)
    value1 = np.array([ 1.        +0.j,         -0.37872566+0.92550898j])
    assert np.allclose(value1, phase[:2])
    
def test_calc_phase_framelength10_radiansTrue():
    frame_length = 10
    time = np.arange(0, 10, 0.1)
    signal = np.sin(time)[:frame_length]
    fft_vals = np.fft.fft(signal)
    phase = pyso.dsp.calc_phase(fft_vals, radians = True)
    value1 = np.array([ 0.,         1.95921533])
    assert np.allclose(value1, phase[:2])
    
def test_reconstruct_whole_spectrum():
    x = np.array([3.,2.,1.,0.,0.,0.,0.])
    x_reconstructed = pyso.dsp.reconstruct_whole_spectrum(x)
    expected = np.array([3., 2., 1., 0., 1., 2., 3.])
    assert np.array_equal(expected, x_reconstructed)
    assert len(x_reconstructed) == len(x)
    
def test_reconstruct_whole_spectrum_input4_nfft7():
    x = np.array([3.,2.,1.,0.])
    n_fft = 7
    x_reconstructed = pyso.dsp.reconstruct_whole_spectrum(x, n_fft=n_fft)
    expected = np.array([3., 2., 1., 0., 1., 2., 3.])
    assert np.array_equal(expected, x_reconstructed)
    assert len(x_reconstructed) == n_fft
    
def test_reconstruct_whole_spectrum_input4_nfft6():
    x = np.array([3.,2.,1.,0.])
    n_fft= 6
    x_reconstructed = pyso.dsp.reconstruct_whole_spectrum(x, n_fft=n_fft)
    print(x_reconstructed)
    expected = np.array([3., 2., 1., 0., 2., 3.])
    assert np.array_equal(expected, x_reconstructed)
    assert len(x_reconstructed) == n_fft
    
def test_reconstruct_whole_spectrum_input4_nfft5():
    x = np.array([3.,2.,1.,0.])
    n_fft = 5
    x_reconstructed = pyso.dsp.reconstruct_whole_spectrum(x, n_fft=n_fft)
    print(x_reconstructed)
    expected = np.array([3., 2., 1., 2., 3.])
    assert np.array_equal(expected, x_reconstructed)
    assert len(x_reconstructed) == n_fft

def test_reconstruct_whole_spectrum_input4_nfft14():
    x = np.array([3.,2.,1.,0.])
    n_fft = 14
    x_reconstructed = pyso.dsp.reconstruct_whole_spectrum(x, n_fft=n_fft)
    print(x_reconstructed)
    expected = np.array([3., 2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3.])
    assert np.array_equal(expected, x_reconstructed)
    assert len(x_reconstructed) == n_fft    
    
def test_reconstruct_whole_spectrum_complexvals():
    np.random.seed(seed=0)
    x_complex = np.random.random(2) + np.random.random(2) * 1j
    n_fft = int(2*len(x_complex))
    x_reconstructed = pyso.dsp.reconstruct_whole_spectrum(x_complex,
                                                     n_fft = n_fft)
    expected = np.array([0.5488135 +0.60276338j, 0.71518937+0.54488318j, 0.        +0.j, 0.5488135 +0.60276338j])
    print(x_reconstructed)
    assert np.allclose(expected, x_reconstructed)
    assert len(x_reconstructed) == n_fft    
    
def test_overlap_add():
    enhanced_matrix = np.ones((4, 4))
    frame_length = 4
    overlap = 2
    sig = pyso.dsp.overlap_add(enhanced_matrix, frame_length, overlap)
    expected = np.array([1., 1., 2., 2., 2., 2., 2., 2., 1., 1.])
    assert np.array_equal(expected, sig)
    
def test_overlap_add():
    enhanced_matrix = np.ones((4, 4))
    frame_length = 4
    overlap = 1
    sig = pyso.dsp.overlap_add(enhanced_matrix, frame_length, overlap)
    expected = np.array([1., 1., 1., 2., 1., 1., 2., 1., 1., 2., 1., 1., 1.])
    assert np.array_equal(expected, sig)
    
def test_overlap_add_complexvals():
    enhanced_matrix = np.ones((4, 4),dtype=np.complex)
    frame_length = 4
    overlap = 1
    sig = pyso.dsp.overlap_add(enhanced_matrix, frame_length, overlap)
    expected = np.array([1.+0.j, 1.+0.j, 1.+0.j, 2.+0.j, 1.+0.j, 1.+0.j, 
                         2.+0.j, 1.+0.j, 1.+0.j, 2.+0.j,1.+0.j, 1.+0.j, 1.+0.j])
    assert sig.dtype == expected.dtype
    
def test_overlap_add_framelength_mismatch():
    enhanced_matrix = np.ones((4, 4))
    frame_length = 3
    overlap = 1
    with pytest.raises(TypeError):
        sig = pyso.dsp.overlap_add(enhanced_matrix, 
                                    frame_length, 
                                    overlap)
        
def test_calc_num_subframes_fullframes():
    expected = 5
    subframes = pyso.dsp.calc_num_subframes(30,10,5)
    assert expected == subframes
    
def test_calc_num_subframes_mismatchframes():
    expected = 5
    subframes = pyso.dsp.calc_num_subframes(33,10,5)
    print(subframes)
    assert expected == subframes
    
def test_calc_num_subframes_mismatchframes_zeropad():
    expected = 6
    subframes = pyso.dsp.calc_num_subframes(33,10,5, zeropad=True)
    print(subframes)
    assert expected == subframes
        
def test_generate_sound_default():
    data, sr = pyso.dsp.generate_sound()
    expected1 = np.array([0., 0.06260483, 0.12366658, 0.18168021, 0.2352158 ])
    expected2 = 2000
    expected3 = 8000
    assert np.allclose(expected1, data[:5])
    assert len(data) == expected2
    assert sr == expected3
    
def test_generate_sound_freq5():
    sound, sr = pyso.dsp.generate_sound(freq=5, amplitude=0.5, sr=5, dur_sec=1)
    expected1 = np.array([ 0.000000e+00,  5.000000e-01,  
                         3.061617e-16, -5.000000e-01, -6.123234e-16])
    expected_sr = 5
    expected_len = expected_sr * 1
    assert np.allclose(expected1, sound)
    assert sr == expected_sr
    assert len(sound) == expected_len
    
def test_get_time_points():
    time = pyso.dsp.get_time_points(dur_sec = 0.1, sr=50)
    expected = np.array([0.    ,0.025 ,0.05 , 0.075, 0.1  ])
    assert np.allclose(time, expected)
    
def test_generate_noise():
    noise = pyso.dsp.generate_noise(5, random_seed=0)
    expected = np.array([0.04410131, 0.01000393, 0.02446845, 0.05602233, 0.04668895])
    assert np.allclose(expected, noise)
    
def test_set_signal_length_longer():
    input_samples = np.array([1,2,3,4,5])
    samples = pyso.dsp.set_signal_length(input_samples, numsamps = 8)
    expected = np.array([1,2,3,4,5,0,0,0])
    assert len(samples) == 8
    assert np.array_equal(samples, expected)
    
def test_set_signal_length_shorter():
    input_samples = np.array([1,2,3,4,5])
    samples = pyso.dsp.set_signal_length(input_samples, numsamps = 4)
    expected = np.array([1,2,3,4])
    assert len(samples) == 4
    assert np.array_equal(samples, expected)
    
def test_scalesound_default():
    np.random.seed(0)
    input_samples = np.random.random_sample((5,))
    output_samples = pyso.dsp.scalesound(input_samples)
    expected = np.array([-0.14138 ,1., 0.22872961, -0.16834299, -1.])
    assert np.allclose(output_samples, expected)
    
def test_scalesound_min_minus100_max100():
    np.random.seed(0)
    input_samples = np.random.random_sample((5,))
    output_samples = pyso.dsp.scalesound(input_samples, min_val=-100, max_val=100)
    expected = np.array([ -14.13800026,100., 22.87296052,-16.83429866,-100.])
    assert np.allclose(output_samples, expected)
    
def test_scalesound_min_minuspoint25_maxpoint25():
    np.random.seed(0)
    input_samples = np.random.random_sample((5,))
    output_samples = pyso.dsp.scalesound(input_samples, min_val=-.25, max_val=.25)
    expected = np.array([-0.035345, 0.25, 0.0571824, -0.04208575, -0.25])
    assert np.allclose(output_samples, expected)
    
def test_normalize_default():
    np.random.seed(0)
    input_samples = np.random.random_sample((5,))
    output_samples = pyso.feats.normalize(input_samples)
    expected = np.array([0.42931, 1., 0.6143648, 0.41582851, 0.])
    assert np.allclose(output_samples, expected)
    
def test_normalize_min_max():
    np.random.seed(0)
    input_samples = np.random.random_sample((5,))
    np.random.seed(40)
    previous_samples = np.random.random_sample((5,))
    min_val = np.min(previous_samples)
    max_val = np.max(previous_samples)
    output_samples = pyso.feats.normalize(input_samples, 
                                        max_val = max_val, min_val = min_val)
    expected = np.array([0.67303388, 0.89996095, 0.74661839, 0.66767314, 0.50232462])
    assert np.allclose(output_samples, expected)
    
def test_resample_audio_sr22050_to_16000():
    test_audio_1sec, sr = pyso.loadsound(test_audiofile,dur_sec=1, sr=22050)
    assert sr == 22050
    assert len(test_audio_1sec) == 22050
    test_audio_newsr, sr_new = pyso.dsp.resample_audio(test_audio_1sec, 
                                               sr_original = sr,
                                               sr_desired = 16000)
    assert sr_new == 16000
    assert len(test_audio_newsr) == 16000
    
def test_resample_audio_sr100_to_80():
    # signal 5 milliseconds long
    test_audio, sr1 = pyso.dsp.generate_sound(freq=10, sr=100,dur_sec=0.05)
    test_resampled, sr2 = pyso.dsp.resample_audio(test_audio, 
                                             sr_original=sr1,
                                             sr_desired=80)
    expected = np.array([-2.22044605e-17, 3.35408001e-01, 3.72022523e-01, 6.51178161e-02])
    assert np.allclose(test_resampled, expected)
    
def test_stereo2mono_stereo_input():
    # 50 samples:
    data = np.linspace(0,20)
    data_2channel = data.reshape(25,2)
    data_mono = pyso.dsp.stereo2mono(data_2channel)
    expected = np.array([0., 0.81632653, 1.63265306, 2.44897959, 3.26530612])
    assert np.allclose(data_mono[:5],expected)
    
def test_stereo2mono_mono_input():
    # 50 samples mono 
    data = np.expand_dims(np.linspace(0,20),axis=1)
    data_mono = pyso.dsp.stereo2mono(data)
    assert np.array_equal(data, data_mono)

def test_apply_sample_length():
    data = np.array([1,2,3,4,5])
    num_samps = 10
    new_data = pyso.dsp.apply_sample_length(data, num_samps)
    expected = np.array([1,2,3,4,5,1,2,3,4,5])
    assert np.array_equal(new_data, expected)
    assert len(new_data) == num_samps
    assert data.dtype == new_data.dtype
    

def test_apply_sample_length_mirrored():
    data = np.array([1,2,3,4,5])
    num_samps = 10
    new_data = pyso.dsp.apply_sample_length(data, num_samps, mirror_sound=True)
    expected = np.array([1,2,3,4,5,4,3,2,1,2])
    assert np.array_equal(new_data, expected)
    assert len(new_data) == num_samps
    assert data.dtype == new_data.dtype
    
def test_apply_sample_length_tooshort():
    data = np.array([1,2,3,4,5])
    num_samps = 3
    new_data = pyso.dsp.apply_sample_length(data, num_samps)
    print(new_data)
    assert len(new_data) == num_samps
    assert data.dtype == new_data.dtype
    
def test_apply_sample_length_stereo_longer():
    data = np.zeros((3,2))
    data[:,0] = np.array([0,1,2])
    data[:,1] = np.array([1,2,3])
    num_samps = 5
    new_data = pyso.dsp.apply_sample_length(data,num_samps, clip_at_zero=False)
    assert len(new_data) == num_samps
    assert len(new_data.shape) == len(data.shape)
    assert new_data.shape[1] == data.shape[1]
    
def test_apply_sample_length_stereo_shorter():
    data = np.zeros((3,2))
    data[:,0] = np.array([0,1,2])
    data[:,1] = np.array([1,2,3])
    num_samps = 2
    new_data = pyso.dsp.apply_sample_length(data,num_samps)
    assert len(new_data) == num_samps
    assert len(new_data.shape) == len(data.shape)
    assert new_data.shape[1] == data.shape[1]
    assert np.array_equal(data[:2], new_data)
    
def test_apply_sample_length_toomany_dimensions():
    data = np.zeros((2,3,3))
    with pytest.raises(ValueError):
        pyso.dsp.apply_sample_length(data,5)
        
def test_apply_num_channels_1d_to_3d():
    data = np.array([1, 1, 1, 1])
    data3d = pyso.dsp.apply_num_channels(data, 3)
    expected = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1],[1, 1, 1]])
    assert np.array_equal(expected, data3d)
    
def test_apply_num_channels_3d_to_2d():
    data = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1],[1, 1, 1]])
    data2d = pyso.dsp.apply_num_channels(data, 2)
    expected = np.array([[1, 1],[1, 1],[1, 1],[1, 1]])
    assert np.array_equal(expected, data2d)
    
def test_zeropad_sound_mono_targetlen8():
    data = np.array([1,2,3,4])
    data_zeropadded = pyso.dsp.zeropad_sound(data, sr=4, target_len=8)
    expected = np.array([1., 2., 3., 4., 0., 0., 0., 0.])
    assert np.array_equal(data_zeropadded, expected)
    
def test_zeropad_sound_mono_targetlen8_delay1sec():
    data = np.array([1,2,3,4])
    data_zeropadded = pyso.dsp.zeropad_sound(data, sr=4, target_len=8, delay_sec=1)
    expected = np.array([0., 0., 0., 0., 1., 2., 3., 4.])
    assert np.array_equal(data_zeropadded, expected)
    
def test_zeropad_stereo_targetlen_5():
    data = np.zeros((3,2))
    data[:,0] = np.array([0,1,2])
    data[:,1] = np.array([1,2,3])
    data_zeropadded = pyso.dsp.zeropad_sound(data, sr=3, target_len=5)
    expected = np.array([[0., 1.],[1., 2.],[2., 3.],[0., 0.],[0., 0.]])
    assert np.array_equal(data_zeropadded, expected)
    
def test_random_selection_samples_wrap():
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    expected = np.array([ 7,  8,  9, 10,  1,  2,  3])
    got = pyso.dsp.random_selection_samples(x, len_section_samps = 7, 
                                            wrap = True, random_seed = 40)
    assert expected.all() == got.all()
    
def test_random_selection_samples_nowrap():
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    expected = np.array([3, 4, 5, 6, 7, 8, 9])
    got = pyso.dsp.random_selection_samples(x, len_section_samps = 7, 
                                            wrap = False, random_seed = 40)
    assert expected.all() == got.all()

def test_clip_at_zero_negative_to_positive_transition():
    w = np.sin(np.arange(400)+0.7) 
    b = pyso.dsp.clip_at_zero(w)
    assert round(b[0],1) == 0
    assert round(b[-1],1) == 0
    assert b[1] > 0
    assert b[-2] < 0
    
