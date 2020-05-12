'''Useful code to copy and paste for documentation. Will remove at some point

        Examples
        --------
        >>> 
        >>> 
        >>> 
        >>> 
        >>> 
        >>> 
        >>> 
        >>> 
        >>> 
        >>> 
        
    Examples
    --------
    >>> 
    >>> 
    >>> 
    >>> 
    >>> 
    >>> 
    >>> 
    >>> 
    >>> 
    >>> 
'''





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

'''
The filters module covers functions related to the filtering out of noise of
a target signal, whether that be the collection of power spectrum values to
calculate the average power spectrum of each audio class or to measure 
signal-to-noise ratio of a signal and ultimately the the actual filtering 
process.
'''
###############################################################################
import numpy as np

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
packagedir = os.path.dirname(parentdir)
sys.path.insert(0, packagedir)

# TODO consolidate
import sys
import numpy as np
from scipy.fftpack import fft, rfft, ifft, irfft
from scipy.io import wavfile
from scipy.signal import hamming, hanning
import librosa
import cmath

import pysoundtool as pyst

# what Wiener Filter and Average pow spec can inherit
class FilterSettings:
    """Basic settings for filter related classes to inherit from.

    Attributes
    ----------
    frame_dur : int, float
        Time in milliseconds of each audio frame window. (default 20)
    sr : int 
        Desired sampling rate of audio; audio will be resampled to match if
        audio has other sampling rate. (default 48000)
    frame_length : int 
        Number of audio samples in each frame: frame_dur multiplied with
        sampling_rate, divided by 1000. (default 960)
    percent_overlap : float
        Percentage of overlap between frames.
    overlap_length : int 
        Number of overlapping audio samples between subsequent frames: 
        frame_length multiplied by percent_overlap, floored. (default 480)
    window_type : str
        Type of window applied to audio frames: hann vs hamming (default
        'hamming')
    num_fft_bins : int 
        The number of frequency bins used when calculating the fft. 
        Currently the `frame_length` is used to set `num_fft_bins`.
    """
    def __init__(self,
                 frame_duration_ms=None,
                 percent_overlap=None,
                 sampling_rate=None,
                 window_type=None):
        # set defaults if no values given
        self.frame_dur = frame_duration_ms if frame_duration_ms else 20
        self.percent_overlap = percent_overlap if percent_overlap else 0.5
        self.sr = sampling_rate if sampling_rate else 48000
        self.window_type = window_type if window_type else 'hamming'
        # set other attributes based on above values
        self.frame_length = pyst.dsp.calc_frame_length(
            self.frame_dur,
            self.sr)
        self.overlap_length = pyst.dsp.calc_num_overlap_samples(
            self.frame_length,
            self.percent_overlap)
        self.num_fft_bins = self.frame_length
        
    def get_window(self):
        '''Returns window acc. to attributes `window_type` and `frame_length`
        '''
        window = pyst.dsp.create_window(self.window_type, self.frame_length)
        return window


class Filter(FilterSettings):
    """Interactive class to explore Wiener filter settings on audio signals.

    These class methods implement research based algorithms with low 
    computational cost, aimed for noise reduction via mobile phone.

    Attributes
    ----------
    beta : float
        Value applied in Wiener filter that smooths the application of 'gain';
        default set according to previous research. (default 0.98)
    first_iter : bool, optional
        Keeps track if `first_iter` is relevant in filtering. If True, 
        filtering has just started, and calculations made for filtering cannot
        use information from previous frames; if False, calculations for 
        filtering use information from previous frames; if None, no difference
        is applied when processing the 1st vs subsequent frames. (default None)
    target_subframes : int, None
        The number of total subsections within the total number of samples 
        belonging to the target signal (i.e. wavfile being filtered). Until
        `target_subframes` is calculated, it is set to None. (default None) 
    noise_subframes : int, None
        The number of total subsections within the total number of samples 
        belonging to the noise signal. If noise power spectrum is used, this
        doesn't need to be calculated. Until `noise_subframes` is calculated, 
        it is set to None. (default None)
    gain : ndarray, None
        Once calculated, the attenuation values to be applied to the fft for 
        noise reduction. Until calculated, None. (default None)
    max_vol : float, int 
        The maximum volume allowed for the filtered signal. (default 0.4)
    """

    def __init__(self,
                 frame_duration_ms=None,
                 percent_overlap=None,
                 sampling_rate=None,
                 window_type=None,
                 max_vol = None):
        FilterSettings.__init__(self, 
                                frame_duration_ms=frame_duration_ms,
                                percent_overlap=percent_overlap,
                                sampling_rate=sampling_rate,
                                window_type=window_type)
        self.max_vol = max_vol if max_vol else 0.4
        self.target_subframes = None
        self.noise_subframes = None

    def get_samples(self, wavfile, dur_sec=None):
        """Load signal and save original volume

        Parameters
        ----------
        wavfile : str
            Path and name of wavfile to be loaded
        dur_sec : int, float optional
            Max length of time in seconds (default None)

        Returns 
        ----------
        samples : ndarray
            Array containing signal amplitude values in time domain
        """
        samples, sr = pyst.dsp.load_signal(
            wavfile, self.sr, dur_sec=dur_sec)
        self.set_volume(samples, max_vol = self.max_vol)
        return samples

    def set_volume(self, samples, max_vol = 0.4, min_vol = 0.15):
        """Records and limits the maximum amplitude of original samples.

        This enables the output wave to be within a range of
        volume that does not go below or too far above the 
        orignal maximum amplitude of the signal. 

        Parameters
        ----------
        samples : ndarray
            The original samples of a signal (1 dimensional), of any length
        max_vol : float
            The maximum volume level. If a signal has values higher than this 
            number, the signal is curtailed to remain at and below this number.
        min_vol : float
            The minimum volume level. If a signal has only values lower than
            this number, the signal is amplified to be at this number and below.
        
        Returns
        -------
        None
        """
        if isinstance(samples, np.ndarray):
            max_amplitude = samples.max()
        else:
            max_amplitude = max(samples)
        self.vol_orig = max_amplitude
        if max_amplitude > max_vol:
            self.max_vol = max_vol
        elif max_amplitude < min_vol:
            self.max_vol = min_vol
        else:
            self.max_vol = max_amplitude
        return None

    def set_num_subframes(self, len_samples, is_noise=False):
        """Sets the number of target or noise subframes available for processing

        Parameters
        ----------
        len_samples : int 
            The total number of samples in a given signal
        is_noise : bool
            If False, subframe number saved under self.target_subframes, otherwise 
            self.noise_subframes (default False)

        Returns
        -------
        None
        """
        if is_noise:
            self.noise_subframes = pyst.dsp.calc_num_subframes(
                tot_samples=len_samples,
                frame_length=self.frame_length,
                overlap_samples=self.overlap_length
            )
        else:
            self.target_subframes = pyst.dsp.calc_num_subframes(
                tot_samples=len_samples,
                frame_length=self.frame_length,
                overlap_samples=self.overlap_length
            )
        return None

    def load_power_vals(self, path_npy):
        """Loads and checks shape compatibility of averaged power values

        Parameters
        ----------
        path_npy : str, pathlib.PosixPath
            Path to .npy file containing power information. 

        Returns
        -------
        power_values : ndarray
            The power values as long as they have the shape (self.num_fft_bins, 1)
        """
        power_values = pyst.paths.load_feature_data(path_npy)
        if power_values.shape[0] != self.num_fft_bins:
            raise ValueError("Power value shape does not match settings.\
                \nProvided power value shape: {}\
                \nExpected shape: ({},)".format(
                power_values.shape, self.num_fft_bins))
        # get rid of extra, unnecessary dimension
        if power_values.shape == (self.num_fft_bins, 1):
            power_values = power_values.reshape(self.num_fft_bins,)
        return power_values

    def check_volume(self, samples):
        """ensures volume of filtered signal is within the bounds of the original
        """
        max_orig = round(max(samples), 2)
        samples = pyst.dsp.control_volume(samples, self.max_vol)
        max_adjusted = round(max(samples), 2)
        if max_orig != max_adjusted:
            print("volume adjusted from {} to {}".format(max_orig, max_adjusted))
        return samples

    def save_filtered_signal(self, output_file, samples, overwrite=False):
        saved, filename = pyst.paths.save_wave(
            output_file, samples, self.sr, overwrite=overwrite)
        if saved:
            print('Wavfile saved under: {}'.format(filename))
            return True
        else:
            print('Error occurred. {} not saved.'.format(filename))
            return False
        
class WienerFilter(Filter):
    def __init__(self,
                 frame_duration_ms=None,
                 percent_overlap=None,
                 sampling_rate=None,
                 window_type=None,
                 max_vol = 0.4,
                 smooth_factor=0.98,
                 first_iter=None):
        Filter.__init__(self, 
                        frame_duration_ms=frame_duration_ms,
                        sampling_rate=sampling_rate,
                        window_type=window_type,
                        max_vol=max_vol)
        self.beta = smooth_factor
        self.first_iter = first_iter
        self.gain = None
        
    def apply_wienerfilter(self, frame_index, target_fft, target_power_frame, noise_power):
        if frame_index == 0:
            posteri = pyst.matrixfun.create_empty_matrix(
                (len(target_power_frame),))
            self.posteri_snr = pyst.dsp.calc_posteri_snr(
                target_power_frame, noise_power)
            self.posteri_prime = pyst.dsp.calc_posteri_prime(
                self.posteri_snr)
            self.priori_snr = pyst.dsp.calc_prior_snr(snr=self.posteri_snr,
                                            snr_prime=self.posteri_prime,
                                            smooth_factor=self.beta,
                                            first_iter=True,
                                            gain=None)
        elif frame_index > 0:
            self.posteri_snr = pyst.dsp.calc_posteri_snr(
                target_power_frame,
                noise_power)
            self.posteri_prime = pyst.dsp.calc_posteri_prime(
                self.posteri_snr)
            self.priori_snr = pyst.dsp.calc_prior_snr(
                snr=self.posteri_snr_prev,
                snr_prime=self.posteri_prime,
                smooth_factor=self.beta,
                first_iter=False,
                gain=self.gain_prev)
        self.gain = pyst.dsp.calc_gain(prior_snr=self.priori_snr)
        enhanced_fft = pyst.dsp.apply_gain_fft(target_fft, self.gain)
        return enhanced_fft
    
    def apply_postfilter(self, enhanced_fft, target_fft, 
                         target_power_frame):
        target_noisereduced_power = pyst.dsp.calc_power(enhanced_fft)
        self.gain = pyst.dsp.postfilter(target_power_frame,
                                target_noisereduced_power,
                                gain=self.gain,
                                threshold=0.9,
                                scale=20)
        enhanced_fft = pyst.dsp.apply_gain_fft(target_fft, self.gain)
        return enhanced_fft


class BandSubtraction(Filter):
    def __init__(self,
                 frame_duration_ms=None,
                 percent_overlap=None,
                 sampling_rate=None,
                 window_type=None,
                 max_vol = 0.4,
                 num_bands = 6,
                 band_spacing = 'linear'):
        Filter.__init__(self, 
                        frame_duration_ms=frame_duration_ms,
                        sampling_rate=sampling_rate,
                        window_type=window_type,
                        max_vol=max_vol)
        self.num_bands = num_bands
        self.band_spacing = band_spacing
        
    def apply_bandspecsub(self, target_power, target_phase, noise_power):
        self.setup_bands()
        self.update_posteri_bands(target_power,noise_power)
        beta = self.calc_oversub_factor()
        reduced_noise_target = self.sub_noise(target_power, noise_power, beta)
        
        # perhaps don't need. TODO test this with real signal (ie half of fft)
        if len(reduced_noise_target) < len(target_power):
            reduced_noise_target = reduced_noise_target.reshape((reduced_noise_target.shape[0],))
            temp = np.zeros(target_power.shape)
            for i, item in enumerate(reduced_noise_target):
                temp[i] += item
            reduced_noise_target = temp

            reduced_noise_target = pyst.matrixfun.reconstruct_whole_spectrum(
                reduced_noise_target, n_fft=self.num_fft_bins)
            
        # if the phase is complex, it is represented in a spectrum
        if isinstance(target_phase[0], np.complex):
            phase_radians = False
        # otherewise, phase is in radians
        else:
            phase_radians = True
        
        # apply original phase to reduced noise power 
        enhanced_fft = pyst.dsp.apply_original_phase(
            reduced_noise_target,
            target_phase, 
            radians=phase_radians)
        return enhanced_fft
    
    def setup_bands(self):
        '''Provides starting and ending frequncy bins/indices for each band.
        
        Parameters
        ----------
        self : class
            Contains variables `num_bands` (if None, set to 6) and `frame_length`
            
        Returns
        -------
        None
            Sets the class variables `band_start_freq` and `band_end_freq`.
            
        Examples
        --------
        >>> import pysoundtool as pyst
        >>> import numpy as np
        >>> # Default is set to 6 bands:
        >>> fil = pyst.Filter()
        >>> fil.setup_bands()
        >>> fil.band_start_freq
        [  0. 160. 320. 480. 640. 800.]
        >>> fil.band_end_freq
        [160. 320. 480. 640. 800. 960.]
        >>> # change default settings
        >>> fil = pyst.Filter(num_bands=5)
        >>> fil.setup_bands()
        >>> fil.band_start_freq
        [  0. 192. 384. 576. 768.]
        >>> fil.band_end_freq
        [192. 384. 576. 768. 960.]
        '''
        if self.num_bands is None:
            self.num_bands = 6
        if 'linear' in self.band_spacing.lower():
            
            try:
            #calc number of bins per band
                assert self.frame_length / self.num_bands % 2 == 0
            except AssertionError:
                print("The number of bands must be equally divisible by the frame length.")
                sys.exit()
            self.bins_per_band = self.frame_length//(2 * self.num_bands)
                
            band_start_freq = np.zeros((self.num_bands,))
            band_end_freq = np.zeros((self.num_bands,))
            try:
                for i in range(self.num_bands):
                    
                    band_start_freq[i] = int(i*self.bins_per_band)
                    band_end_freq[i] = int(band_start_freq[i] + self.bins_per_band)
            except TypeError:
                print(band_start_freq[i] + self.bins_per_band-1)
                sys.exit()
        # TODO: implement other band spacing types
        # other options of band spacing 
        elif 'log' in self.band_spacing.lower():
            pass
        elif 'mel' in self.band_spacing.lower():
            pass
        
        self.band_start_freq = band_start_freq
        self.band_end_freq = band_end_freq
        
        return None
    
    def update_posteri_bands(self,target_powspec, noise_powspec):
        '''Updates SNR of each set of bands.
        
        MATLAB code from speech enhancement book uses power, 
        puts it into magnitude (via square root), then puts
        it back into power..? And uses some sort of 'norm' function...
        which I think is actually just the sum. Original equation 
        can be found in the paper below. 
        page 117 from book?
        
        paper:
        Kamath, S. D. & Loizou, P. C. (____), A multi-band spectral subtraction method for enhancing speech corrupted by colored noise.
        
        I am using power for the time being. 
        
        Examples
        --------
        >>> import pysoundtool as pyst
        >>> import numpy as np
        >>> # setting to 4 bands for space:
        >>> fil = pyst.Filter(num_bands=4)
        >>> fil.setup_bands()
        >>> # generate sine signal with and without noise
        >>> time = np.arange(0, 10, 0.01)
        >>> signal = np.sin(time)[:fil.frame_length]
        >>> noise = np.random.normal(np.mean(signal),np.mean(signal)+0.3,960)
        >>> powerspec_clean = np.abs(np.fft.fft(signal))**2
        >>> powerspec_noisy = np.abs(np.fft.fft(signal + noise))**2
        >>> fil.update_posteri_bands(powerspec_clean, powerspec_noisy)
        >>> fil.snr_bands
        [ -2.16634315 -44.82425811 -44.81157068  -1.15580912]
        >>> # compare with no noise in signal:
        >>> fil.update_posteri_bands(powerspec_clean, powerspec_clean)
        [0. 0. 0. 0.]
        '''
        snr_bands = np.zeros((self.num_bands,))
        for band in range(self.num_bands):
            start_bin = int(self.band_start_freq[band])
            stop_bin = int(self.band_end_freq[band])
            numerator = sum(target_powspec[start_bin:stop_bin])
            denominator = sum(noise_powspec[start_bin:stop_bin])
            snr_bands[band] += 10*np.log10(numerator/denominator)
        self.snr_bands = snr_bands
        return None
    
    def calc_oversub_factor(self):
        '''Calculate over subtraction factor used in the cited paper.
        
        Uses decibel SNR values calculated in update_posteri_bands()
        
        paper:
        Kamath, S. D. & Loizou, P. C. (____), A multi-band spectral subtraction method ofr enhancing speech corrupted by colored noise.
        
        Examples
        --------
        >>> import pysoundtool as pyst
        >>> import numpy as np
        >>> # setting to 4 bands for space:
        >>> fil = pyst.Filter(num_bands=4)
        >>> fil.setup_bands()
        >>> # generate sine signal with and without noise
        >>> time = np.arange(0, 10, 0.01)
        >>> signal = np.sin(time)[:fil.frame_length]
        >>> noise = np.random.normal(np.mean(signal),np.mean(signal)+0.3,960)
        >>> powerspec_clean = np.abs(np.fft.fft(signal))**2
        >>> powerspec_noisy = np.abs(np.fft.fft(signal + noise))**2
        >>> fil.update_posteri_bands(powerspec_clean, powerspec_noisy)
        >>> fil.snr_bands
        [ -2.16634315 -44.82425811 -44.81157068  -1.15580912]
        >>> a = fil.calc_oversub_factor()
        >>> a
        [4.33042222 4.75       4.75       4.16179247]
        >>> # compare with no noise in signal:
        >>> fil.update_posteri_bands(powerspec_clean, powerspec_clean)
        [0. 0. 0. 0.]
        >>> a = fil.calc_oversub_factor()
        >>> a
        [4. 4. 4. 4.]
        '''
        a = np.zeros(self.snr_bands.shape[0])
        for band in range(self.num_bands):
            band_snr = self.snr_bands[band]
            if band_snr >= -5.0 and band_snr <= 20:
                a[band] = 4 - band_snr*3/20
            elif band_snr < -5.0:
                a[band] = 4.75
            else:
                a[band] = 1
        return a
    
    def calc_relevant_band(self,target_powspec):
        '''Calculates band with highest energy levels.
        
        Parameters
        ----------
        self : class instance
            Contains class variables `band_start_freq` and `band_end_freq`.
        target_powerspec : np.ndarray
            Power spectrum of the target signal.
        
        Returns
        -------
        rel_band_index : int
            Index for which band contains the most energy.
        band_energy_matrix : np.ndarray [size=(num_bands, ), dtype=np.float]
            Power levels of each band.
            
        Examples
        --------
        >>> import pysoundtool as pyst
        >>> import numpy as np
        >>> # setting to 4 bands for this example (default is 6):
        >>> fil = pyst.Filter(num_bands=4)
        >>> fil.setup_bands()
        >>> # generate sine signal with and with frequency 25
        >>> time = np.arange(0, 10, 0.01)
        >>> full_circle = 2 * np.pi
        >>> freq = 25
        >>> signal = np.sin((freq*full_circle)*time)[:fil.frame_length]
        >>> powerspec_clean = np.abs(np.fft.fft(signal))**2
        >>> rel_band_inex, band_power_energies = fil.calc_relevant_band(powerspec_clean)
        >>> rel_band_index
        1
        >>> # and with frequency 50
        >>> freq = 50
        >>> signal = np.sin((freq*full_circle)*time)[:fil.frame_length]
        >>> powerspec_clean = np.abs(np.fft.fft(signal))**2
        >>> rel_band_inex, band_power_energies = fil.calc_relevant_band(powerspec_clean)
        >>> rel_band_index
        2
        '''
        band_energy_matrix = np.zeros(self.num_bands)
        for band in range(self.num_bands):
            start_bin = int(self.band_start_freq[band])
            end_bin = int(self.band_end_freq[band])
            target_band = target_powspec[start_bin:end_bin]
            band_energy_matrix[band] += sum(target_band)/len(target_band)
        rel_band_index = np.argmax(band_energy_matrix)
        
        return rel_band_index, band_energy_matrix
    
    def apply_floor(self, sub_band, original_band, floor=0.002, book=True):
        for i, val in enumerate(sub_band):
            if val < 0:
                sub_band[i] = floor * original_band[i]
            if book:
                #this adds a bit of noise from original signal
                #to avoid musical noise distortion
                sub_band[i] += 0.5*original_band[i]
        
        return sub_band
    
    def sub_noise(self,target_powspec, noise_powspec, oversub_factor, speech=True):
        #apply higher or lower noise subtraction (i.e. delta)
        #lower frequency / bin == lower delta --> reduce speech distortion
        if not speech:
            relevant_band, __ = self.calc_relevant_band(target_powspec)
        else:
            relevant_band = 0
        sub_signal = np.zeros((self.num_bands*self.bins_per_band,1))
        #sub_signal = np.zeros((self.num_bands*self.bins_per_band,1))
        section = 0
        for band in range(self.num_bands):
            start_bin = int(self.band_start_freq[band])
            end_bin = int(self.band_end_freq[band])
            target_band = target_powspec[start_bin:end_bin]
            target_band = np.expand_dims(target_band, axis=1)
            noise_band = noise_powspec[start_bin:end_bin]
            noise_band = np.expand_dims(noise_band, axis=1)
            beta = oversub_factor[band] 
            if band == relevant_band:
                delta = 1 #don't interfer too much with important target band
            else: 
                delta = 2.5 #less important bands --> more noise subtraction
            adjusted = target_band - (beta  * noise_band * delta)
            start = section
            end = start + self.bins_per_band
            sub_signal[start:end,:] = adjusted
            sub_signal[start:end,:]  = self.apply_floor(
                sub_signal[start:end,:] , 
                target_band, book=True)
            section += self.bins_per_band
            
        return sub_signal
            

class WelchMethod(FilterSettings):
    """Applies Welch's method according to filter class attributes. 

    Attributes
    ----------
    len_noise_sec : int, float
        The amount of time in seconds to use from signal to apply the
        Welch's method. (default 1)
    target_subframes : int, None
        The number of total subsections within the total number of samples 
        belonging to the target signal (i.e. wavfile being filtered). Until
        `target_subframes` is calculated, it is set to None. Note: if the
        target signal contains time sensitive information, e.g. speech, 
        it is not advised to apply Welch's method as the time senstive 
        data would be lost. (default None) 
    noise_subframes : int, None
        The number of total subsections within the total number of samples 
        belonging to the noise signal. Until `noise_subframes` is calculated, 
        it is set to None. (default None)
    """
    def __init__(self,
                 len_noise_sec=1):
        FilterSettings.__init__(self)
        self.len_noise_sec = len_noise_sec
        self.target_subframes = None
        self.noise_subframes = None

    def set_num_subframes(self, len_samples, noise=True):
        '''Calculate and set num subframes required to process all samples.
        
        Parameters
        ----------
        len_samples : int 
            Number of total samples in signal.
        noise : bool 
            If True, the class attribute `noise_subframes` will be set; if
            False, the class attribute `target_subframes` will be set.
            
        Returns
        -------
        None
        '''
        if noise:
            self.noise_subframes = pyst.dsp.calc_num_subframes(tot_samples=len_samples,
                                                          frame_length=self.frame_length,
                                                          overlap_samples=self.overlap_length)
        else:
            self.target_subframes = pyst.dsp.calc_num_subframes(tot_samples=len_samples,
                                                           frame_length=self.frame_length,
                                                           overlap_samples=self.overlap_length)
        return None

    def get_power(self, samples, matrix2store_power):
        '''Calculates and adds the power of the noise signal. 
        
        Parameters
        ----------
        samples : ndarray
            The samples from the noise signal.
        matrix2store_power : ndarray
            Where the power values will be added to. Note:
            this is to be later averaged to complete Welch's
            method.
            
        Returns 
        -------
        matrix2store_power : ndarray 
            `matrix2store_power` with the current frame of 
            noise power added.
        '''
        section_start = 0
        for frame in range(self.noise_subframes):
            noise_sect = samples[section_start:section_start +
                                 self.frame_length]
            noise_w_win = pyst.dsp.apply_window(noise_sect, self.get_window())
            noise_fft = pyst.dsp.calc_fft(noise_w_win)
            noise_power = pyst.dsp.calc_power(noise_fft)
            for i, row in enumerate(noise_power):
                matrix2store_power[i] += row
            section_start += self.overlap_length
        return matrix2store_power

    def coll_pow_average(self, wave_list, scale=None, augment_data=False):
        '''Performs Welch's method on (noise) signals in `wave_list`.
        
        Parameters
        ----------
        wave_list : list
            List of wavfiles belonging to noise class. The Welch's method
            will be applied to entire noise class. 
        scale : float, int, optional
            A value to increase or decrease the noise values. This will 
            result in a stronger or weaker filtering.
        augment_data : bool
            If True, the sound data will be augmented. Currently, this 
            uses three versions of each wavefile, at three different 
            energy levels. This is to increase representation of noise 
            that is quiet, mid, and loud.
            
        Returns
        -------
        noise_powspec : ndarray
            The average power spectrum of the entire `wave_list`
        '''
        pwspec_shape = self.get_window().shape+(1,)
        noise_powspec = pyst.matrixfun.create_empty_matrix(
            pwspec_shape, complex_vals=False)
        for j, wav in enumerate(wave_list):
            n, sr = pyst.dsp.load_signal(wav, dur_sec=self.len_noise_sec)
            if augment_data:
                samples = pyst.augmentdata.spread_volumes(n)
            else:
                samples = (n,)
            for sampledata in samples:
                if scale:
                    sampledata *= scale
                noise_powspec = self.get_power(sampledata, noise_powspec)

            progress = (j+1) / len(wave_list) * 100
            sys.stdout.write(
                "\r%d%% through sound class wavfile list" % progress)
            sys.stdout.flush()
        print('\nFinished\n')
        tot_powspec_collected = self.noise_subframes * \
            len(wave_list) * len(samples)
        noise_powspec = pyst.dsp.calc_average_power(
            noise_powspec, tot_powspec_collected)
        return noise_powspec

def get_average_power(class_waves_dict, encodelabel_dict,
                      powspec_dir, duration_sec=1, augment_data=False):
    '''Collects ave. power spectrum from audio classes; saves in .npy files
    
    Parameters
    ----------
    class_waves_dict : dict 
        Dictionary containing audio class labels and the wavfile paths
        of the files belonging to each audio class. 
    encodelabel_dict : dict
        Dictionary with keys matching the audio class labels and the 
        values matching the integer each audio class is encoded as.
    powspec_dir : str, pathlib.PosixPath
        Path to where average power spectrum files will be stored.
    duration_sec : int, float
        The length in seconds to be processed when calculating the 
        average power spectrum of each wavfile. (default 1)
    augmentdata : bool
        If True, the samples will be augmented in their energy levels
        so that the sounds are represented at quiet, mid, and loud 
        levels. If False, no augmentation will take place. 
        
    Returns
    -------
    None
    '''
    avspec = WelchMethod(len_noise_sec=duration_sec)
    avspec.set_num_subframes(int(avspec.len_noise_sec * avspec.sr),
                             noise=True)
    total_classes = len(class_waves_dict)
    count = 0
    for key, value in class_waves_dict.items():
        print('\nProcessing class {} out of {}'.format(count+1, total_classes))
        # value = str(waves list)
        wave_list = pyst.paths.string2list(value)
        noise_powspec = avspec.coll_pow_average(wave_list,
                                                augment_data=augment_data)
        path2save = powspec_dir.joinpath(
            # key is str label of class--> encoded integer
            'powspec_noise_{}.npy'.format(encodelabel_dict[key]))
        pyst.paths.save_feature_data(path2save, noise_powspec)
        count += 1
    return None

def calc_audioclass_powerspecs(path_class, dur_ms = 1000,
                               augment_data=False):
    '''Uses class's path settings to set up Welch's method for audio classes.
    
    The settings applied for average power spectrum collection are also saved
    in a .csv file.
    
    Parameters
    ----------
    path_class : class
        Class with attributes for necessary paths to load relevant wavfiles and 
        save average power spectrum values.
    dur_ms : int, float
        Time in milliseconds for the Welch's method / average power spectrum
        calculation to be applied for each wavfile. (default 1000)
    augment_data : bool
        Whether or not the sound data should be augmented. If True, the sound
        data will be processed three times: with low energy, mid energy, and 
        high energy. (default False)
        
    Returns
    -------
    None
    '''
    class_waves_dict = pyst.paths.load_dict(path_class.labels_waves_path)
    labels_encoded_dict = pyst.paths.load_dict(path_class.labels_encoded_path)
    encodelabel_dict = {}
    for key, value in labels_encoded_dict.items():
        encodelabel_dict[value] = key
    total_classes = len(class_waves_dict)
    fs = FilterSettings()
    powspec_settings = fs.__dict__
    # add number of classes to settings dictionary to check all classes get processed
    powspec_settings['num_audio_classes'] = total_classes
    powspec_settings['processing_window_sec'] = dur_ms/1000.
    powspec_settings_filename = path_class.powspec_path.joinpath(
        path_class._powspec_settings)
    pyst.paths.save_dict(powspec_settings, powspec_settings_filename)

    get_average_power(class_waves_dict, encodelabel_dict,
                      path_class.powspec_path,
                      duration_sec=dur_ms/1000.0,
                      augment_data=augment_data)
    return None

def coll_beg_audioclass_samps(path_class, 
                              feature_class, 
                              num_each_audioclass=1, 
                              dur_ms=1000):
    '''Saves `dur_ms` of `num_each_audioclass` wavfiles of each audio class.
    
    This is an option for using noise data that comes from an audio class but is 
    not an average of the entire class. It is raw sample data from one or more 
    random noise wavfiles from each class. 
    
    Parameters
    ----------
    path_class : class
        Class with attributes for necessary paths to load relevant wavfiles and 
        save sample values.
    feature_class : class
        Class with attributes for sampling rate used in feature extraction and/or 
        filtering. This is useful to maintain consistency in sampling rate
        throughout the modules.
    num_each_audioclass : int 
        The number of random wavfiles from each audio class chosen for raw 
        sample collection. (default 1)
    dur_ms : int, float
        Time in milliseconds of raw sample data to be saved. (default 1000)
    
    Returns
    -------
    None
    '''
    class_waves_dict = pyst.paths.load_dict(path_class.labels_waves_path)
    labels_encoded_dict = pyst.paths.load_dict(path_class.labels_encoded_path)
    encodelabel_dict = {}
    for key, value in labels_encoded_dict.items():
        encodelabel_dict[value] = key
    for key, value in class_waves_dict.items():
        wavlist = pyst.paths.string2list(value)
        label_int = encodelabel_dict[key]
        rand_indices = np.random.randint(
            0,len(class_waves_dict),num_each_audioclass)
        noisefiles = []
        for index in rand_indices:
            noisefiles.append(wavlist[index])
        get_save_begsamps(noisefiles, label_int,
                          path_class.powspec_path,
                          samplerate=feature_class.sr,
                          dur_ms=dur_ms)
    return None

def get_save_begsamps(wavlist,audioclass_int,
                      powspec_dir,samplerate=48000,dur_ms=1000):
    '''Saves the beginning raw samples from the wavfiles in `wavlist`.
    
    Parameters
    ----------
    wavlist : list
        List containing paths of relevant wavfiles 
    audioclass_int : int
        The integer the audio class is encoded as.
    powspec_dir : str, pathlib.PosixPath
        Path to where data relevant for audio class power spectrum data
        are to be saved.
    samplerate : int 
        The sampling rate of wavfiles. This is needed to calculate num 
        samples necessary to get `dur_ms` of sound. (default 48000)
    dur_ms : int, float
        Time in milleseconds of the wavefiles to collect and save.
    '''
    numsamps = pyst.dsp.calc_frame_length(dur_ms,samplerate)
    for i, wav in enumerate(wavlist):
        y, sr = pyst.dsp.load_signal(wav,sampling_rate=samplerate)
        y_beg = y[:numsamps]
        filename = powspec_dir.joinpath(
            'beg{}ms{}sr_{}_audioclass{}.npy'.format(dur_ms, 
                                                     samplerate, 
                                                     i,
                                                     audioclass_int))
        pyst.paths.save_feature_data(filename, y_beg)
    return None
