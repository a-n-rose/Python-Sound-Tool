
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import librosa
import numpy as np
import pytest
import pysoundtool as pyst

test_audiofile = './audio2channels.wav'
test_noisyfile = './audiodata/python_traffic.wav'
test_filtered_wiener = './audiodata/python_traffic_wiener.wav'
test_filtered_wiener_postfilter = './audiodata/python_traffic_pf.wav'
test_filtered_bandsub = './audiodata/python_traffic_bs.wav'


def test_setup_bands_default():
    fil = pyst.BandSubtraction()
    fil.setup_bands()
    band_start_freq = fil.band_start_freq
    band_end_freq = fil.band_end_freq
    expected1 = np.array([  0.,  80., 160., 240., 320., 400.])
    expected2 = np.array([ 80., 160., 240., 320., 400., 480.])
    assert np.array_equal(expected1, band_start_freq)
    assert np.array_equal(expected2, band_end_freq)
    
def test_setup_bands_8():
    fil = pyst.BandSubtraction(num_bands = 8)
    fil.setup_bands()
    band_start_freq = fil.band_start_freq
    band_end_freq = fil.band_end_freq
    expected1 = np.array([  0.,  60., 120., 180., 240., 300., 360., 420.])
    expected2 = np.array([ 60., 120., 180., 240., 300., 360., 420., 480.])
    assert np.array_equal(expected1, band_start_freq)
    assert np.array_equal(expected2, band_end_freq)

def test_setup_bands_winsize16ms():
    fil = pyst.BandSubtraction(win_size_ms = 16)
    fil.setup_bands()
    band_start_freq = fil.band_start_freq
    band_end_freq = fil.band_end_freq
    expected1 = np.array([  0.,  64., 128., 192., 256., 320.])
    expected2 = np.array([ 64., 128., 192., 256., 320., 384.])
    assert np.array_equal(expected1, band_start_freq)
    assert np.array_equal(expected2, band_end_freq)
    
def test_setup_bands_winsize500ms():
    fil = pyst.BandSubtraction(win_size_ms = 500)
    fil.setup_bands()
    band_start_freq = fil.band_start_freq
    band_end_freq = fil.band_end_freq
    expected1 = np.array([    0.,  2000.,  4000.,  6000.,  8000., 10000.])
    expected2 = np.array([ 2000.,  4000.,  6000.,  8000., 10000., 12000.])
    assert np.array_equal(expected1, band_start_freq)
    assert np.array_equal(expected2, band_end_freq)
    
def test_update_posteri_bands_noisy():
    noise_max = 0.3
    fil = pyst.BandSubtraction(num_bands = 3)
    fil.setup_bands()
    time = np.arange(0, 10, 0.01)
    signal = np.sin(time)[:fil.frame_length]
    np.random.seed(seed=0)
    noise = np.random.normal(np.mean(signal),
                             np.mean(signal)+noise_max,
                             fil.frame_length)
    powspec = np.abs(np.fft.fft(signal))**2
    powspec_noisy = np.abs(np.fft.fft(signal + noise))**2
    fil.update_posteri_bands(powspec, powspec_noisy)
    snr_bands = fil.snr_bands
    expected = np.array([ -2.02865226, -41.70672353, -45.45654087])
    assert np.allclose(expected, snr_bands)
    

def test_update_posteri_bands_verynoisy():
    noise_max = 0.7
    fil = pyst.BandSubtraction(num_bands = 3)
    fil.setup_bands()
    time = np.arange(0, 10, 0.01)
    signal = np.sin(time)[:fil.frame_length]
    np.random.seed(seed=0)
    noise = np.random.normal(np.mean(signal),
                             np.mean(signal)+noise_max,
                             fil.frame_length)
    powspec = np.abs(np.fft.fft(signal))**2
    powspec_noisy = np.abs(np.fft.fft(signal + noise))**2
    fil.update_posteri_bands(powspec, powspec_noisy)
    snr_bands = fil.snr_bands
    expected = np.array([ -2.82864994, -46.76075799, -50.50670912])
    assert np.allclose(expected, snr_bands)
    
def test_update_posteri_bands_nonoise():
    fil = pyst.BandSubtraction(num_bands = 3)
    fil.setup_bands()
    time = np.arange(0, 10, 0.01)
    signal = np.sin(time)[:fil.frame_length]
    powspec = np.abs(np.fft.fft(signal))**2
    powspec_noisy = powspec
    fil.update_posteri_bands(powspec, powspec_noisy)
    snr_bands = fil.snr_bands
    expected = np.array([0., 0., 0.])
    assert np.allclose(expected, snr_bands)
    
def test_calc_oversub_factor_noisy():
    noise_max = 0.3
    fil = pyst.BandSubtraction(num_bands = 4)
    fil.setup_bands()
    time = np.arange(0, 10, 0.01)
    signal = np.sin(time)[:fil.frame_length]
    np.random.seed(seed=0)
    noise = np.random.normal(np.mean(signal),
                             np.mean(signal)+noise_max,
                             fil.frame_length)
    powspec = np.abs(np.fft.fft(signal))**2
    powspec_noisy = np.abs(np.fft.fft(signal + noise))**2
    fil.update_posteri_bands(powspec, powspec_noisy)
    a = fil.calc_oversub_factor()
    expected = np.array([4.28678354, 4.75,       4.75,       4.75      ])
    assert np.allclose(expected, a)
    
def test_calc_oversub_factor_nonoise():
    noise_max = 0.3
    fil = pyst.BandSubtraction(num_bands = 4)
    fil.setup_bands()
    time = np.arange(0, 10, 0.01)
    signal = np.sin(time)[:fil.frame_length]
    powspec = np.abs(np.fft.fft(signal))**2
    fil.update_posteri_bands(powspec, powspec)
    a = fil.calc_oversub_factor()
    expected = np.array([4., 4., 4., 4.])
    assert np.allclose(expected, a)
    
def test_calc_relevant_band1():
    fil = pyst.BandSubtraction(num_bands = 6)
    fil.setup_bands()
    band_index = 0
    freq = fil.band_start_freq[band_index] 
    time = np.arange(0, 10, 0.01)
    full_circle = 2 * np.pi
    signal = np.sin((freq*full_circle)*time)[:fil.frame_length]
    powspec = np.abs(np.fft.fft(signal))**2
    rel_band, pow_levels = fil.calc_relevant_band(powspec)
    print('IF ERROR, PERHAPS DUE TO HARMONICS??? OR BAND SPACING???')
    print('Testing frequency: ', freq)
    print('Expected most relevant band: ', band_index)
    print('Which covers frequencies between {} and {}.'.format(
        fil.band_start_freq[band_index], fil.band_end_freq[band_index]))
    print('Calculated energy levels of bands: ', pow_levels)
    print('Most energetic frequency band: ', rel_band)
    print('Which covers frequencies between {} and {}.'.format(
        fil.band_start_freq[rel_band], fil.band_end_freq[rel_band]))
    expected = band_index
    assert expected == rel_band
    
    
def test_calc_relevant_band2():
    fil = pyst.BandSubtraction(num_bands = 6)
    fil.setup_bands()
    band_index = 1
    freq = fil.band_start_freq[band_index] 
    time = np.arange(0, 10, 0.01)
    full_circle = 2 * np.pi
    signal = np.sin((freq*full_circle)*time)[:fil.frame_length]
    powspec = np.abs(np.fft.fft(signal))**2
    rel_band, pow_levels = fil.calc_relevant_band(powspec)
    print('IF ERROR, PERHAPS DUE TO HARMONICS??? OR BAND SPACING???')
    print('Testing frequency: ', freq)
    print('Expected most relevant band: ', band_index)
    print('Which covers frequencies between {} and {}.'.format(
        fil.band_start_freq[band_index], fil.band_end_freq[band_index]))
    print('Calculated energy levels of bands: ', pow_levels)
    print('Most energetic frequency band: ', rel_band)
    print('Which covers frequencies between {} and {}.'.format(
        fil.band_start_freq[rel_band], fil.band_end_freq[rel_band]))
    expected = band_index
    assert expected == rel_band
    
    
def test_calc_relevant_band4():
    fil = pyst.BandSubtraction(num_bands = 6)
    fil.setup_bands()
    band_index = 2
    freq = fil.band_start_freq[band_index] 
    time = np.arange(0, 10, 0.01)
    full_circle = 2 * np.pi
    signal = np.sin((freq*full_circle)*time)[:fil.frame_length]
    powspec = np.abs(np.fft.fft(signal))**2
    rel_band, pow_levels = fil.calc_relevant_band(powspec)
    print('IF ERROR, PERHAPS DUE TO HARMONICS??? OR BAND SPACING???')
    print('Testing frequency: ', freq)
    print('Expected most relevant band: ', band_index)
    print('Which covers frequencies between {} and {}.'.format(
        fil.band_start_freq[band_index], fil.band_end_freq[band_index]))
    print('Calculated energy levels of bands: ', pow_levels)
    print('Most energetic frequency band: ', rel_band)
    print('Which covers frequencies between {} and {}.'.format(
        fil.band_start_freq[rel_band], fil.band_end_freq[rel_band]))
    expected = band_index
    assert expected == rel_band
    
def test_calc_relevant_band4():
    fil = pyst.BandSubtraction(num_bands = 6)
    fil.setup_bands()
    band_index = 3
    freq = fil.band_start_freq[band_index]
    time = np.arange(0, 10, 0.01)
    full_circle = 2 * np.pi
    signal = np.sin((freq*full_circle)*time)[:fil.frame_length]
    powspec = np.abs(np.fft.fft(signal))**2
    rel_band, pow_levels = fil.calc_relevant_band(powspec)
    print('IF ERROR, PERHAPS DUE TO HARMONICS??? OR BAND SPACING???')
    print('Testing frequency: ', freq)
    print('Expected most relevant band: ', band_index)
    print('Which covers frequencies between {} and {}.'.format(
        fil.band_start_freq[band_index], fil.band_end_freq[band_index]))
    print('Calculated energy levels of bands: ', pow_levels)
    print('Most energetic frequency band: ', rel_band)
    print('Which covers frequencies between {} and {}.'.format(
        fil.band_start_freq[rel_band], fil.band_end_freq[rel_band]))
    expected = band_index
    assert expected == rel_band
    
def test_calc_relevant_band():
    fil = pyst.BandSubtraction(num_bands = 4)
    fil.setup_bands()
    time = np.arange(0, 10, 0.01)
    signal = np.cos(time)[:fil.frame_length]
    powspec = np.abs(np.fft.fft(signal))**2
    rel_band, pow_levels = fil.calc_relevant_band(powspec)
    expected = 0
    assert expected == rel_band
    
def test_bandsub_reset_samplerate_22050():
    sr = 22050 
    fil = pyst.BandSubtraction(num_bands=4, sr=sr)
    updated_sr = fil.sr
    expected = 48000
    assert expected == updated_sr
    
# TODO: just seems a bit complicated.. remove?
#def test_sub_noise():
    #fil = pyst.BandSubtraction(num_bands = 4)
    #fil.setup_bands()
    #time = np.arange(0, 10, 0.01)
    #signal = np.sin(time)[:fil.frame_length]
    #powspec = np.abs(np.fft.fft(signal))**2
    ## add noise
    #np.random.seed(seed=0)
    #noise = 0.1 * np.random.randn(len(signal))
    #noisy_signal = signal + noise
    #powspec_noisy = np.abs(np.fft.fft(noisy_signal))**2
    ## calculate other necessary variables
    #fil.update_posteri_bands(powspec, powspec_noisy)
    #a = fil.calc_oversub_factor()
    #sub_signal = fil.sub_noise(powspec, powspec_noisy, 
                               #oversub_factor = a,
                               #speech = True)

def test_filtersettings_getsamples_default_wiener():
    wf = pyst.WienerFilter()
    samps_wf = wf.get_samples(test_audiofile,
                              dur_sec = 1)
    assert wf.sr == 48000
    assert len(samps_wf) == wf.sr
    
def test_filtersettings_getsamples_default_bandsubtraction():
    bs = pyst.BandSubtraction()
    samps_bs = bs.get_samples(test_audiofile,
                              dur_sec = 1)
    assert bs.sr == 48000
    assert len(samps_bs) == bs.sr
    
    
def test_filtersettings_getsamples_sr22050_wiener():
    sr = 22050
    wf = pyst.WienerFilter(sr=sr)
    samps_wf = wf.get_samples(test_audiofile,
                              dur_sec = 1)
    assert wf.sr == sr
    assert len(samps_wf) == wf.sr
    
def test_filtersettings_getsamples_sr22050_bandsubtraction():
    sr = 22050
    sr_permanent = 48000
    bs = pyst.BandSubtraction(sr=sr)
    samps_bs = bs.get_samples(test_audiofile,
                              dur_sec = 1)
    print('IF ERROR: Check whether or not BandSubtraction works with '+\
        'sample rates other than 48000. If not, the sr must stay at 48000.')
    assert bs.sr == sr_permanent
    assert len(samps_bs) == bs.sr

def test_filtersettings_getsamples_sr8000_wiener():
    sr = 8000
    wf = pyst.WienerFilter(sr=sr)
    samps_wf = wf.get_samples(test_audiofile,
                              dur_sec = 1)
    assert wf.sr == sr
    assert len(samps_wf) == wf.sr

def test_filtersettings_getsamples_sr8000_bandsubtraction():
    sr = 8000
    sr_permanent = 48000
    bs = pyst.BandSubtraction(sr=sr)
    samps_bs = bs.get_samples(test_audiofile,
                              dur_sec = 1)
    print('IF ERROR: Check whether or not BandSubtraction works with '+\
        'sample rates other than 48000. If not, the sr must stay at 48000.')
    assert bs.sr == sr_permanent
    assert len(samps_bs) == bs.sr
    
def test_filtersignal_wiener_simple_doesitrun_uselibrosa_False():
    signal, sr = pyst.filtersignal(test_noisyfile, filter_type = 'wiener',
                                   use_librosa=False)
    sig_expected, sr_expected = librosa.load(test_filtered_wiener, sr=sr)
    assert np.allclose(signal, sig_expected)
    assert sr == sr_expected

def test_filtersignal_wiener_posfilter_simple_doesitrun_uselibrosa_False():
    signal, sr = pyst.filtersignal(test_noisyfile, filter_type = 'wiener_pf',
                                   use_librosa=False)
    sig_expected, sr_expected = librosa.load(test_filtered_wiener_postfilter, sr=sr)
    assert np.allclose(signal, sig_expected)
    assert sr == sr_expected
    
def test_filtersignal_bandsubtraction_simple_doesitrun_uselibrosa_False():
    signal, sr = pyst.filtersignal(test_noisyfile, filter_type = 'bandsubtraction',
                                   use_librosa=False)
    sig_expected, sr_expected = librosa.load(test_filtered_bandsub,sr=sr)
    assert np.allclose(signal, sig_expected)
    assert sr == sr_expected
    
