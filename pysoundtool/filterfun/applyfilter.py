#!/bin/bash
# Copyright 2019 Peggy Sylopp und Aislyn Rose GbR
# All rights reserved
# This file is part of the  NoIze-framework
# The NoIze-framework is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by the  
# Free Software Foundation, either version 3 of the License, or (at your option) 
# any later version.
#
#@author Aislyn Rose
#@version 0.1
#@date 31.08.2019
#
# The  NoIze-framework  is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details. 
#
# You should have received a copy of the GNU AFFERO General Public License 
# along with the NoIze-framework. If not, see http://www.gnu.org/licenses/.

###############################################################################
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
packagedir = os.path.dirname(parentdir)
sys.path.insert(0, packagedir)


import sys
import numpy as np
from scipy.fftpack import fft, rfft, ifft, irfft
from scipy.io import wavfile
from scipy.signal import hamming, hanning
import librosa
import cmath


import pysoundtool as pyst


def apply_band_specsub(output_wave_name, 
             target_wav, 
             noise_wav=None, 
             gain_type = 'power estimation',
             window_type='hamming', 
             sampling_rate = 48000,
             apply_postfilter = False,
             threshold=0.4,
             scale=10,
             vol_scale=400):
    
    fil = pyst.Filter(filter_type='spectral subtraction',
                sampling_rate=sampling_rate,
                gain_type= gain_type, #'power estimation',#'wiener'
                window_type = window_type,
                real_signal = True,
                apply_postfilter = apply_postfilter,
                num_bands = 6,
                band_spacing = 'linear') 
    
    print(repr(fil))

    target = fil.load_signal(target_wav)
    #only use 120 ms of noise
    noise_clip = fil.samprate // 1000 * 120
    
    #if multiple channels, reduce to first channel
    if len(target.shape) > 1 and target.shape[1] > 1:
        print(target.shape)
        target = target[:,0]
        print(target.shape)

    noise = target[0:noise_clip]

    fil.set_num_subframes(len(noise),noise=True)
    fil.set_num_subframes(len(target),noise=False)
    print("num target subframes: ",fil.target_subframes)
    print("num noise subframes: ", fil.noise_subframes)
    
    fil.setup_bands()
    # TODO: why extra dimension?
    print(fil.fft_bins) # 1024
    noise_power_matrix = fil.create_empty_matrix((fil.fft_bins,1,),complex_vals=False)
    #calculate and collect power of noise
    section = 0 
    for frame in range(fil.noise_subframes):
        
        noise_sect = noise[section:section+fil.frame_length]
        
        noise_w_win = fil.apply_window(noise_sect)
        noise_fft = fil.calc_fft(noise_w_win,fft_length=None)

        noise_power = fil.calc_power(noise_fft)
        
        for i, row in enumerate(noise_power):
            noise_power_matrix[i] += row 
            
        section += fil.overlap
    print(noise_power_matrix.shape)
    
    
    noise_power_matrix = fil.calc_average_power(noise_power_matrix,fil.noise_subframes) 
    phase_matrix = fil.create_empty_matrix((fil.fft_bins,fil.target_subframes), complex_vals=True)

    #total_rows = fil.fft_bins//2+1
    total_rows = fil.frame_length
    enhanced_signal = fil.create_empty_matrix((total_rows,fil.target_subframes), complex_vals=False)
    section = 0
    for frame in range(fil.target_subframes):

        target_section = target[section:section + fil.frame_length]
        print('target section shape ', target_section.shape)
        target_w_win = fil.apply_window(target_section)
        print('target w window ', target_w_win.shape)
        target_fft = fil.calc_fft(target_w_win,fft_length=None)
        print('target_fft ', target_fft.shape)
        target_power = fil.calc_power(target_fft)
        print('target_power ', target_power.shape)
        target_phase = fil.calc_phase(target_fft)
        print('phase matrix shape ', phase_matrix.shape)
        phase_matrix[:,frame] += target_phase
        
        print("target phase shape: ",target_phase.shape)
        fil.update_posteri_bands(target_power,noise_power_matrix)
        beta = fil.calc_oversub_factor()
        reduced_noise_target = fil.sub_noise(target_power, noise_power_matrix, beta)
        print('shape reduced_noise_target: ', reduced_noise_target.shape)
        #now mirror, as fft would be / reconstruct spectrum
        print(fil.num_bands)
        print(fil.bins_per_band)
        print(reduced_noise_target.shape)
        #reduced_noise_target = reduced_noise_target.transpose()
        #print(reduced_noise_target.shape)
        
        # TODO phase info is only in first row of second dimension
        # what's going on there?
        for i, row in enumerate(reduced_noise_target):
            enhanced_signal[i][frame] += fil.apply_original_phase(
                row,
                phase_matrix[i][0])[0]
    
    enhanced_signal = fil.reconstruct_spectrum(enhanced_signal)
    print(enhanced_signal)
    
    #enhanced_signal = fil.apply_original_phase(enhanced_signal,phase_matrix)
    #enhanced_signal = enhanced_signal.real
    enhanced_signal = fil.calc_ifft(enhanced_signal)
    #print(enhanced_signal.shape)
    
    
    #overlap add:
    enhanced_signal = fil.overlap_add(enhanced_signal)
    print('final signal shape: ',enhanced_signal.shape) #ideal? (143040, 1)
    enhanced_signal = fil.increase_volume(enhanced_signal)
    
    fil.save_wave(output_wave_name,enhanced_signal)
    
    return True


def filtersignal(output_filename, wavfile, noise_file=None,
                    scale=1, apply_postfilter=False, duration_ms=1000,
                    max_vol = 0.4, filter_type = 'wiener'):
    """Apply Wiener filter to signal using noise. Saves at `output_filename`.

    Parameters 
    ----------
    output_filename : str
        path and name the filtered signal is to be saved
    wavfile : str 
        the filename to the signal for filtering; if None, a signal will be 
        generated (default None)
    noise_file : str optional
        path to either noise wavfile or .npy file containing average power 
        spectrum values or noise samples. If None, the beginning of the
        `wavfile` will be used for noise data. (default None)
    scale : int or float
        The scale at which the filter should be applied. (default 1) 
        Note: `scale` cannot be set to 0.
    apply_postfilter : bool
        Whether or not the post filter should be applied. The post filter 
        reduces musical noise (i.e. distortion) in the signal as a byproduct
        of filtering.
    duration_ms : int or float
        The amount of time in milliseconds to use from noise to apply the 
        Welch's method to. In other words, how much of the noise to use 
        when approximating the average noise power spectrum.
    max_vol : int or float 
        The maximum volume level of the filtered signal.
    filter_type : str
        Type of filter to apply. Options 'wiener' or 'band_specsub'.
    
    Returns
    -------
    None
    """
    if filter_type == 'wiener':
        fil = pyst.WienerFilter(max_vol = max_vol)
    elif filter_type == 'band_specsub':
        fil = pyst.Filter(max_vol = max_vol)

    # load signal (to be filtered)
    samples_orig = fil.get_samples(wavfile)
    # set how many subframes are needed to process entire target signal
    fil.set_num_subframes(len(samples_orig), is_noise=False)

    # prepare noise
    # set up how noise will be considered: either as wavfile, averaged
    # power values, or the first section of the target wavfile (i.e. None)
    samples_noise = None
    if noise_file:
        if '.wav' == str(noise_file)[-4:]:
            samples_noise = fil.get_samples(noise_file)
        elif '.npy' == str(noise_file)[-4:]:
            if 'powspec' in noise_file.stem:
                noise_power = fil.load_power_vals(noise_file)
                samples_noise = None
            elif 'beg' in noise_file.stem:
                samples_noise = pyst.paths.load_feature_data(noise_file)
    else:
        starting_noise_len = pyst.dsp.calc_frame_length(fil.sr, 
                                                         duration_ms)
        samples_noise = samples_orig[:starting_noise_len]
    # if noise samples have been collected...
    if samples_noise is not None:
        # set how many subframes are needed to process entire noise signal
        fil.set_num_subframes(len(samples_noise), is_noise=True)

    # prepare noise power matrix (if it's not loaded already)
    if fil.noise_subframes:
        total_rows = fil.num_fft_bins
        noise_power = pyst.matrixfun.create_empty_matrix((total_rows,))
        section = 0
        for frame in range(fil.noise_subframes):
            noise_section = samples_noise[section:section+fil.frame_length]
            noise_w_win = pyst.dsp.apply_window(noise_section, fil.get_window())
            noise_fft = pyst.dsp.calc_fft(noise_w_win)
            noise_power_frame = pyst.dsp.calc_power(noise_fft)
            noise_power += noise_power_frame
            section += fil.overlap_length
        # welch's method: take average of power that has been collected
        # in windows
        noise_power = pyst.dsp.calc_average_power(noise_power, 
                                                   fil.noise_subframes)
        assert section == fil.noise_subframes * fil.overlap_length

    # prepare target power matrix
    total_rows = fil.frame_length * fil.target_subframes
    filtered_sig = pyst.matrixfun.create_empty_matrix(
        (total_rows,), complex_vals=True)
    section = 0
    row = 0
    target_power_baseline = 0
    noise_power *= scale
    try:
        for frame in range(fil.target_subframes):
            target_section = samples_orig[section:section+fil.frame_length]
            target_w_window = pyst.dsp.apply_window(target_section,
                                                    fil.get_window())
            target_fft = pyst.dsp.calc_fft(target_w_window)
            target_power_frame = pyst.dsp.calc_power(target_fft)
            # now start filtering!!
            # initialize SNR matrix
            if frame == 0:
                posteri = pyst.matrixfun.create_empty_matrix(
                    (len(target_power_frame),))
                fil.posteri_snr = pyst.dsp.calc_posteri_snr(
                    target_power_frame, noise_power)
                fil.posteri_prime = pyst.dsp.calc_posteri_prime(
                    fil.posteri_snr)
                fil.priori_snr = pyst.dsp.calc_prior_snr(snr=fil.posteri_snr,
                                                snr_prime=fil.posteri_prime,
                                                smooth_factor=fil.beta,
                                                first_iter=True,
                                                gain=None)
            elif frame > 0:
                fil.posteri_snr = pyst.dsp.calc_posteri_snr(
                    target_power_frame,
                    noise_power)
                fil.posteri_prime = pyst.dsp.calc_posteri_prime(
                    fil.posteri_snr)
                fil.priori_snr = pyst.dsp.calc_prior_snr(
                    snr=fil.posteri_snr_prev,
                    snr_prime=fil.posteri_prime,
                    smooth_factor=fil.beta,
                    first_iter=False,
                    gain=fil.gain_prev)
            fil.gain = pyst.dsp.calc_gain(prior_snr=fil.priori_snr)
            enhanced_fft = pyst.dsp.apply_gain_fft(target_fft, fil.gain)
            if apply_postfilter:
                target_noisereduced_power = pyst.dsp.calc_power(enhanced_fft)
                fil.gain = pyst.dsp.postfilter(target_power_frame,
                                        target_noisereduced_power,
                                        gain=fil.gain,
                                        threshold=0.9,
                                        scale=20)
                enhanced_fft = pyst.dsp.apply_gain_fft(target_fft, fil.gain)
            enhanced_ifft = pyst.dsp.calc_ifft(enhanced_fft)
            filtered_sig[row:row+fil.frame_length] += enhanced_ifft
            # prepare for next iteration
            fil.posteri_snr_prev = fil.posteri_snr
            fil.gain_prev = fil.gain
            row += fil.overlap_length
            section += fil.overlap_length
    except ValueError as e:
        print(e)
        print(frame)
    assert row == fil.target_subframes * fil.overlap_length
    assert section == fil.target_subframes * fil.overlap_length
    # make enhanced_ifft values real
    enhanced_signal = filtered_sig.real
    enhanced_signal = fil.check_volume(enhanced_signal)
    if len(enhanced_signal) > len(samples_orig):
        enhanced_signal = enhanced_signal[:len(samples_orig)]
    fil.save_filtered_signal(str(output_filename), 
                            enhanced_signal,
                            overwrite=True)
    return None
