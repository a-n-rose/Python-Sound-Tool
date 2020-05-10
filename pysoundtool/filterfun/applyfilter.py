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
import matplotlib.pyplot as plt

import pysoundtool as pyst

# temporary

def visualize_feats(feature_matrix, feature_type, 
                    save_pic=False, name4pic=None, scale='power_to_db'):
    '''Visualize feature extraction; frames on x axis, features on y axis. 
    
    Parameters
    ----------
    feature_matrix : numpy.ndarray
        Matrix of feeatures.
    feature_type : str
        Either 'mfcc' or 'fbank' features. MFCC: mel frequency cepstral
        coefficients; FBANK: mel-log filterbank energies (default 'fbank')
    scale : str, optional
        If features need to be adjusted, e.g. from power to decibels. 
        Default is 'power_to_db', as the visuals look best when working 
        with features extracted via librosa package.
    '''
    if 'fbank' in feature_type:
        axis_feature_label = 'Num Mel Filters'
    elif 'mfcc' in feature_type:
        axis_feature_label = 'Num Mel Freq Cepstral Coefficients'
    elif 'stft' in feature_type:
        axis_feature_label = 'Short-Time Fourier Transform'
    elif 'signal' in feature_type:
        axis_feature_label = 'Amplitude'
    if scale is None:
        pass
    elif scale == 'power_to_db':
        feature_matrix = librosa.power_to_db(feature_matrix)
    elif scale == 'db_to_power':
        feature_matrix = librosa.db_to_power(feature_matrix)
    elif scale == 'amplitude_to_db':
        feature_matrix = librosa.amplitude_to_db(feature_matrix)
    elif scale == 'db_to_amplitude':
        feature_matrix = librosa.db_to_amplitude(feature_matrix)
    plt.clf()
    if feature_type == 'signal':
        plt.plot(feature_matrix)
    else:
        plt.pcolormesh(feature_matrix.T)
    plt.xlabel('Frames')
    plt.ylabel(axis_feature_label)
    plt.title('{} Features'.format(feature_type.upper()))
    if save_pic:
        outputname = name4pic or 'visualize{}feats'.format(feature_type.upper())
        plt.savefig('{}.png'.format(outputname))
    else:
        plt.show()
        
'''
%  References:
%   [1] Kamath, S. and Loizou, P. (2002). A multi-band spectral subtraction 
%       method for enhancing speech corrupted by colored noise. Proc. IEEE Int.
%       Conf. Acoust.,Speech, Signal Processing
%   
% Authors: Sunil Kamath and Philipos C. Loizou
%
% Copyright (c) 2006 by Philipos C. Loizou
% $Revision: 0.0 $  $Date: 10/09/2006 $
'''

def apply_band_specsub(output_wave_name, 
             target_wav, 
             noise_file=None):
    
    fil = pyst.BandSubtraction() 

    target = fil.get_samples(target_wav)
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
    else:
        starting_noise_len = pyst.dsp.calc_frame_length(fil.samprate, 
                                                         duration_ms)
        samples_noise = samples_orig[:starting_noise_len]

    if samples_noise is not None:
        # set how many subframes are needed to process entire noise signal
        fil.set_num_subframes(len(samples_noise), is_noise=True)

    fil.set_num_subframes(len(target),is_noise=False)
    print("num target subframes: ",fil.target_subframes)
    print("num noise subframes: ", fil.noise_subframes)
    
    # define frequency limits for each band
    fil.setup_bands()
    noise = samples_noise
    #visualize_feats(noise, 'signal')

    total_rows = fil.frame_length
    noise_power = pyst.matrixfun.create_empty_matrix((total_rows,))
    section = 0
    window = pyst.dsp.create_window(fil.window_type, fil.frame_length)
    for frame in range(fil.noise_subframes):
        noise_section = samples_noise[section:section+fil.frame_length]
        noise_w_win = pyst.dsp.apply_window(noise_section, window)
        
        noise_fft = pyst.dsp.calc_fft(noise_w_win)
        noise_power_frame = pyst.dsp.calc_power(noise_fft)
        noise_power += noise_power_frame
        section += fil.overlap_length
    # welch's method: take average of power that has been collected
    # in windows
    noise_power = pyst.dsp.calc_average_power(noise_power, 
                                                fil.noise_subframes)
    assert section == fil.noise_subframes * fil.overlap_length
    visualize_feats(noise_power.reshape(noise_power.shape[0],1), 'stft')

    phase_matrix = pyst.matrixfun.create_empty_matrix((fil.num_fft_bins,fil.target_subframes), complex_vals=True)

    total_rows = fil.frame_length
    enhanced_signal = pyst.matrixfun.create_empty_matrix((total_rows,fil.target_subframes), complex_vals=False)
    section = 0
    for frame in range(fil.target_subframes):

        target_section = target[section:section + fil.frame_length]
        print('target section shape ', target_section.shape)
        target_w_win = pyst.dsp.apply_window(target_section, window)
        print('target w window ', target_w_win.shape)
        target_fft = pyst.dsp.calc_fft(target_w_win)
        print('target_fft ', target_fft.shape)
        target_power = pyst.dsp.calc_power(target_fft)
        print('target_power ', target_power.shape)
        target_phase = pyst.dsp.calc_phase(target_fft, radians=True)
        print('phase matrix shape ', phase_matrix.shape)
        phase_matrix[:,frame] += target_phase
        
        print("target phase shape: ",target_phase.shape)
        fil.update_posteri_bands(target_power,noise_power)
        beta = fil.calc_oversub_factor()
        reduced_noise_target = fil.sub_noise(target_power, noise_power, beta)
        print('shape reduced_noise_target: ', reduced_noise_target.shape)
        #now mirror, as fft would be / reconstruct spectrum
        print(fil.num_bands)
        print(fil.bins_per_band)
        print(reduced_noise_target.shape)
        #reduced_noise_target = reduced_noise_target.transpose()
        #print(reduced_noise_target.shape)
        print(enhanced_signal[0].shape)
        for i, row in enumerate(reduced_noise_target):
            enhanced_signal[i] += row 
        
    visualize_feats(enhanced_signal, 'stft')

    enhanced_signal = pyst.matrixfun.reconstruct_whole_spectrum(
        enhanced_signal,
        n_fft = fil.num_fft_bins)

    visualize_feats(enhanced_signal, 'stft')
    
    enhanced_signal = fil.apply_original_phase(enhanced_signal,phase_matrix)
    
    visualize_feats(enhanced_signal, 'stft')
    
    enhanced_signal = pyst.dsp.calc_ifft(enhanced_signal)
    visualize_feats(enhanced_signal, 'stft')
    enhanced_signal = enhanced_signal.real
    visualize_feats(enhanced_signal, 'stft')
    enhanced_signal = pyst.matrixfun.overlap_add(
        enhanced_signal,
        frame_length = fil.frame_length,
        overlap = fil.overlap_length,
        complex_vals = False)
    visualize_feats(enhanced_signal, 'signal')
    print('final signal shape: ',enhanced_signal.shape) #ideal? (143040, 1)
    enhanced_signal = fil.check_volume(enhanced_signal)
    if len(enhanced_signal) > len(target):
        enhanced_signal = enhanced_signal[:len(target)]
    visualize_feats(enhanced_signal, 'signal')
    fil.save_filtered_signal(str(output_wave_name), 
                            enhanced_signal,
                            overwrite=True)
    
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

    visualize_feats(samples_noise, 'signal')
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
