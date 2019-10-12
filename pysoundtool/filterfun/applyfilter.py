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

import pysoundtool as pyst

def filtersignal(output_filename, wavfile, noise_file=None,
                    scale=1, apply_postfilter=False, duration_ms=1000,
                    max_vol = 0.4):
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
    
    Returns
    -------
    None
    """
    wf = pyst.WienerFilter(max_vol = max_vol)

    # load signal (to be filtered)
    samples_orig = wf.get_samples(wavfile)
    # set how many subframes are needed to process entire target signal
    wf.set_num_subframes(len(samples_orig), is_noise=False)

    # prepare noise
    # set up how noise will be considered: either as wavfile, averaged
    # power values, or the first section of the target wavfile (i.e. None)
    samples_noise = None
    if noise_file:
        if '.wav' == str(noise_file)[-4:]:
            samples_noise = wf.get_samples(noise_file)
        elif '.npy' == str(noise_file)[-4:]:
            if 'powspec' in noise_file.stem:
                noise_power = wf.load_power_vals(noise_file)
                samples_noise = None
            elif 'beg' in noise_file.stem:
                samples_noise = pyst.paths.load_feature_data(noise_file)
    else:
        starting_noise_len = pyst.dsp.calc_frame_length(wf.sr, 
                                                         duration_ms)
        samples_noise = samples_orig[:starting_noise_len]
    # if noise samples have been collected...
    if samples_noise is not None:
        # set how many subframes are needed to process entire noise signal
        wf.set_num_subframes(len(samples_noise), is_noise=True)

    # prepare noise power matrix (if it's not loaded already)
    if wf.noise_subframes:
        total_rows = wf.num_fft_bins
        noise_power = pyst.matrixfun.create_empty_matrix((total_rows,))
        section = 0
        for frame in range(wf.noise_subframes):
            noise_section = samples_noise[section:section+wf.frame_length]
            noise_w_win = pyst.dsp.apply_window(noise_section, wf.get_window())
            noise_fft = pyst.dsp.calc_fft(noise_w_win)
            noise_power_frame = pyst.dsp.calc_power(noise_fft)
            noise_power += noise_power_frame
            section += wf.overlap_length
        # welch's method: take average of power that has been colleced
        # in windows
        noise_power = pyst.dsp.calc_average_power(noise_power, 
                                                   wf.noise_subframes)
        assert section == wf.noise_subframes * wf.overlap_length

    # prepare target power matrix
    total_rows = wf.frame_length * wf.target_subframes
    filtered_sig = pyst.matrixfun.create_empty_matrix(
        (total_rows,), complex_vals=True)
    section = 0
    row = 0
    target_power_baseline = 0
    noise_power *= scale
    try:
        for frame in range(wf.target_subframes):
            target_section = samples_orig[section:section+wf.frame_length]
            target_w_window = pyst.dsp.apply_window(target_section, wf.get_window())
            target_fft = pyst.dsp.calc_fft(target_w_window)
            target_power_frame = pyst.dsp.calc_power(target_fft)
            # now start filtering!!
            # initialize SNR matrix
            if frame == 0:
                posteri = pyst.matrixfun.create_empty_matrix(
                    (len(target_power_frame),))
                wf.posteri_snr = pyst.dsp.calc_posteri_snr(
                    target_power_frame, noise_power)
                wf.posteri_prime = pyst.dsp.calc_posteri_prime(
                    wf.posteri_snr)
                wf.priori_snr = pyst.dsp.calc_prior_snr(snr=wf.posteri_snr,
                                                snr_prime=wf.posteri_prime,
                                                smooth_factor=wf.beta,
                                                first_iter=True,
                                                gain=None)
            elif frame > 0:
                wf.posteri_snr = pyst.dsp.calc_posteri_snr(
                    target_power_frame,
                    noise_power)
                wf.posteri_prime = pyst.dsp.calc_posteri_prime(
                    wf.posteri_snr)
                wf.priori_snr = pyst.dsp.calc_prior_snr(
                    snr=wf.posteri_snr_prev,
                    snr_prime=wf.posteri_prime,
                    smooth_factor=wf.beta,
                    first_iter=False,
                    gain=wf.gain_prev)
            wf.gain = pyst.dsp.calc_gain(prior_snr=wf.priori_snr)
            enhanced_fft = pyst.dsp.apply_gain_fft(target_fft, wf.gain)
            if apply_postfilter:
                target_noisereduced_power = pyst.dsp.calc_power(enhanced_fft)
                wf.gain = pyst.dsp.postfilter(target_power_frame,
                                        target_noisereduced_power,
                                        gain=wf.gain,
                                        threshold=0.9,
                                        scale=20)
                enhanced_fft = pyst.dsp.apply_gain_fft(target_fft, wf.gain)
            enhanced_ifft = pyst.dsp.calc_ifft(enhanced_fft)
            filtered_sig[row:row+wf.frame_length] += enhanced_ifft
            # prepare for next iteration
            wf.posteri_snr_prev = wf.posteri_snr
            wf.gain_prev = wf.gain
            row += wf.overlap_length
            section += wf.overlap_length
    except ValueError as e:
        print(e)
        print(frame)
    assert row == wf.target_subframes * wf.overlap_length
    assert section == wf.target_subframes * wf.overlap_length
    # make enhanced_ifft values real
    enhanced_signal = filtered_sig.real
    enhanced_signal = wf.check_volume(enhanced_signal)
    if len(enhanced_signal) > len(samples_orig):
        enhanced_signal = enhanced_signal[:len(samples_orig)]
    wf.save_filtered_signal(str(output_filename), 
                            enhanced_signal,
                            overwrite=True)
    return None
