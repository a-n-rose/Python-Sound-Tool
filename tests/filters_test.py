
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pysoundtool as pyst
import numpy as np
import pytest


def test_setup_bands_default():
    fil = pyst.Filter()
    fil.setup_bands()
    band_start_freq = fil.band_start_freq
    band_end_freq = fil.band_end_freq
    value1 = np.array([  0., 160., 320., 480., 640., 800.])
    value2 = np.array([160., 320., 480., 640., 800., 960.])
    assert np.array_equal(value1, band_start_freq)
    assert np.array_equal(value2, band_end_freq)
    
def test_setup_bands_8():
    fil = pyst.Filter(num_bands = 8)
    fil.setup_bands()
    band_start_freq = fil.band_start_freq
    band_end_freq = fil.band_end_freq
    value1 = np.array([  0., 120., 240., 360., 480., 600., 720., 840.])
    value2 = np.array([120., 240., 360., 480., 600., 720., 840., 960.])
    assert np.array_equal(value1, band_start_freq)
    assert np.array_equal(value2, band_end_freq)

def test_setup_bands_frameduration16ms():
    fil = pyst.Filter(frame_duration = 16)
    fil.setup_bands()
    band_start_freq = fil.band_start_freq
    band_end_freq = fil.band_end_freq
    value1 = np.array([  0., 128., 256., 384., 512., 640.])
    value2 = np.array([128., 256., 384., 512., 640., 768.])
    assert np.array_equal(value1, band_start_freq)
    assert np.array_equal(value2, band_end_freq)
    
def test_setup_bands_frameduration500ms():
    fil = pyst.Filter(frame_duration = 500)
    fil.setup_bands()
    band_start_freq = fil.band_start_freq
    band_end_freq = fil.band_end_freq
    value1 = np.array([    0.,  4000.,  8000., 12000., 16000., 20000.])
    value2 = np.array([ 4000.,  8000., 12000., 16000., 20000., 24000.])
    assert np.array_equal(value1, band_start_freq)
    assert np.array_equal(value2, band_end_freq)
    
def test_update_posteri_bands_noisy():
    noise_max = 0.3
    fil = pyst.Filter(num_bands = 3)
    fil.setup_bands()
    time = np.arange(0, 10, 0.01)
    signal = np.sin(time)[:fil.frame_length]
    noise = np.random.normal(np.mean(signal),
                             np.mean(signal)+noise_max,
                             fil.frame_length)
    powspec = np.abs(np.fft.fft(signal))**2
    powspec_noisy = np.abs(np.fft.fft(signal + noise))**2
    fil.update_posteri_bands(powspec, powspec_noisy)
    snr_bands = fil.snr_bands
    value1 = np.array([ -2.38856707, -45.41496251,  -1.48348645])
    assert np.allclose(value1, snr_bands)
    

def test_update_posteri_bands_verynoisy():
    noise_max = 0.7
    fil = pyst.Filter(num_bands = 3)
    fil.setup_bands()
    time = np.arange(0, 10, 0.01)
    signal = np.sin(time)[:fil.frame_length]
    noise = np.random.normal(np.mean(signal),
                             np.mean(signal)+noise_max,
                             fil.frame_length)
    powspec = np.abs(np.fft.fft(signal))**2
    powspec_noisy = np.abs(np.fft.fft(signal + noise))**2
    fil.update_posteri_bands(powspec, powspec_noisy)
    snr_bands = fil.snr_bands
    value1 = np.array([ -3.94990272, -50.07895978,  -3.31517877])
    assert np.allclose(value1, snr_bands)
    
def test_update_posteri_bands_nonoise():
    fil = pyst.Filter(num_bands = 3)
    fil.setup_bands()
    time = np.arange(0, 10, 0.01)
    signal = np.sin(time)[:fil.frame_length]
    powspec = np.abs(np.fft.fft(signal))**2
    powspec_noisy = powspec
    fil.update_posteri_bands(powspec, powspec_noisy)
    snr_bands = fil.snr_bands
    value1 = np.array([0., 0., 0.])
    assert np.allclose(value1, snr_bands)
    
def test_calc_oversub_factor_noisy():
    noise_max = 0.3
    fil = pyst.Filter(num_bands = 4)
    fil.setup_bands()
    time = np.arange(0, 10, 0.01)
    signal = np.sin(time)[:fil.frame_length]
    noise = np.random.normal(np.mean(signal),
                             np.mean(signal)+noise_max,
                             fil.frame_length)
    powspec = np.abs(np.fft.fft(signal))**2
    powspec_noisy = np.abs(np.fft.fft(signal + noise))**2
    fil.update_posteri_bands(powspec, powspec_noisy)
    a = fil.calc_oversub_factor()
    value1 = np.array([4.33042222, 4.75,       4.75,       4.16179247])
    assert np.allclose(value1, a)
    
def test_calc_oversub_factor_nonoise():
    noise_max = 0.3
    fil = pyst.Filter(num_bands = 4)
    fil.setup_bands()
    time = np.arange(0, 10, 0.01)
    signal = np.sin(time)[:fil.frame_length]
    powspec = np.abs(np.fft.fft(signal))**2
    fil.update_posteri_bands(powspec, powspec)
    a = fil.calc_oversub_factor()
    value1 = np.array([4., 4., 4., 4.])
    assert np.allclose(value1, a)
    
def test_calc_relevant_band1():
    fil = pyst.Filter(num_bands = 6)
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
    value1 = band_index
    assert value1 == rel_band
    
    
def test_calc_relevant_band2():
    fil = pyst.Filter(num_bands = 6)
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
    value1 = band_index
    assert value1 == rel_band
    
    
def test_calc_relevant_band4():
    fil = pyst.Filter(num_bands = 6)
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
    value1 = band_index
    assert value1 == rel_band
    
def test_calc_relevant_band4():
    fil = pyst.Filter(num_bands = 6)
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
    value1 = band_index
    assert value1 == rel_band
    
def test_calc_relevant_band():
    fil = pyst.Filter(num_bands = 4)
    fil.setup_bands()
    time = np.arange(0, 10, 0.01)
    signal = np.cos(time)[:fil.frame_length]
    powspec = np.abs(np.fft.fft(signal))**2
    rel_band, pow_levels = fil.calc_relevant_band(powspec)
    value1 = 0
    assert value1 == rel_band
    
