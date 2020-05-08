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
                 frame_duration_ms=20,
                 percent_overlap=0.5,
                 sampling_rate=48000,
                 window_type='hamming'):
        self.frame_dur = frame_duration_ms
        self.sr = sampling_rate
        self.frame_length = pyst.dsp.calc_frame_length(
            frame_duration_ms,
            sampling_rate)
        self.percent_overlap = percent_overlap
        self.overlap_length = pyst.dsp.calc_num_overlap_samples(
            self.frame_length,
            percent_overlap)
        self.window_type = window_type
        self.num_fft_bins = self.frame_length
        
    def get_window(self):
        '''Returns window acc. to attributes `window_type` and `frame_length`
        '''
        window = pyst.dsp.create_window(self.window_type, self.frame_length)
        return window


class WienerFilter(FilterSettings):
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
                 smooth_factor=0.98,
                 first_iter=None,
                 max_vol = 0.4):
        FilterSettings.__init__(self)
        self.beta = smooth_factor
        self.first_iter = first_iter
        self.target_subframes = None
        self.noise_subframes = None
        self.gain = None
        self.max_vol = max_vol

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
        

class Filter:
    def __init__(self, frame_duration = 20, 
                 percent_overlap = 50, 
                 filter_type = 'wiener', #or spectral subtraction
                 sampling_rate = 48000, 
                 smooth_factor = 0.98, 
                 gain_type = 'power estimation', 
                 window_type = 'hamming', 
                 real_signal = True,
                 apply_postfilter = False,
                 num_bands = None,
                 band_spacing = 'linear'):
        self.filter_type = filter_type
        self.frame_dur = frame_duration
        self.samprate = sampling_rate
        self.frame_length = frame_duration * sampling_rate //1000
        if percent_overlap > 1: 
            percent_overlap /= 100
        self.overlap = int(self.frame_length * percent_overlap) 
        self.common_length = self.frame_length-self.overlap #mband.m script
        self.beta = smooth_factor
        self.gain_type = gain_type
        self.window = self.create_window(window_type)
        self.real_signal = real_signal
        self.apply_postfilter = apply_postfilter
        self.num_bands = num_bands
        self.band_spacing = band_spacing
        # just starting value:
        self.fft_bins = 2
        
        


    def __repr__(self):
        return (f'{self.__class__.__name__}('
            f'\nFilter type: {self.filter_type!r} ms'
            f'\nFrame duration: {self.frame_dur!r} ms'
            f'\nSample rate: {self.samprate!r} hz'
            f'\nFrame length: {self.frame_length!r} samples'
            f'\nFrame overlap: {self.overlap!r} samples'
            f'\nSmoothing factor: {self.beta!r}'
            f'\nGain type: {self.gain_type!r}'
            f'\nWindow type: {self.window_type!r})'
            )
        
        
    def create_window(self, window_type):
        self.window_type = window_type
        if window_type.lower() == 'hamming':
            window = hamming(self.frame_length)
        elif window_type.lower() == 'hanning':
            window = hanning(self.frame_length)
        self.norm_win = np.dot(window, window) / self.frame_length
        
        return window


    def normalize_signal(self,signal, max_val, min_val):
        signal = (signal - min_val) / (max_val - min_val)
        
        return signal
    
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


    def load_signal(self,wav):
        try:
            signal, sr = librosa.load(wav,sr=self.samprate)
            #sr, signal = wavfile.read(wav)
            assert sr == self.samprate
            self.data_type = signal.dtype
        except AssertionError as e:
            print(e)
            print("Scipy found sampling rate: {} while you set the sampling rate as {}".format(sr, self.samprate))
            print("Please adjust the sampling rate.")
            sys.exit()
        return signal


    def create_empty_matrix(self,shape,complex_vals=False):
        if complex_vals:
            matrix = np.zeros(shape,dtype=np.complex_)
        else:
            matrix = np.zeros(shape)
        return matrix


    def calc_fft(self,signal,fft_length=None):
        if self.real_signal:
            fft_vals = rfft(signal,fft_length)
        else:
            fft_vals = fft(signal,fft_length)
        return fft_vals
    
    
    def calc_ifft(self,signal):
        if self.real_signal:
            ifft_vals = irfft(signal)
        else:
            ifft_vals = ifft(signal)
        
        return ifft_vals
    
    
    def calc_average_power(self, matrix, num_iters):
        for i in range(len(matrix)):
            matrix[i] /= num_iters
        return matrix
    

    def apply_window(self, samples):
        try:
            samples_win = samples.copy()
            assert samples.shape == self.window.shape
            samples_win *= self.window
            assert samples_win.shape == samples.shape
            
            return samples_win
        
        except AssertionError as e:
            print(e)
            print('Shapes do not align')
            print('samples = {}; window = {}'.format(samples.shape,self.window.shape))
            if samples_win:
                print('samples_win = {}'.format(samples_win.shape))
            sys.exit()
        
        return None


    def set_num_subframes(self, len_samples, noise=True):
        if noise:
            self.noise_subframes = int(len_samples/ self.overlap) -1
        else:
            self.target_subframes = int(len_samples/ self.overlap) -1
        return None


    def calc_power(self, fft, normalization=True):
        if normalization:
            power_spec = np.abs(fft)**2 / (self.frame_length * self.norm_win)
        else:
            power_spec = np.abs(fft)**2
        
        return power_spec


    def update_posteri(self, target_power_spec, noise_power_spec):
        #each power spectrum is divided by their corresponding frequency bin
        
        posteri_snr = np.zeros(target_power_spec.shape)
        for i in range(len(target_power_spec)):
            posteri_snr[i] += target_power_spec[i] / noise_power_spec[i]
        assert posteri_snr.shape == (self.frame_length,) 
        self.posteri_snr = posteri_snr
        
        return None


    def calc_posteri_prime(self,positive=True):
        """
        estimate snr
        flooring at 0 --> half-wave rectification (values cannot be negative)
        this may cause issues in lower SNRs
        """
        posteri_prime = self.posteri_snr - 1
        if positive:
            posteri_prime[posteri_prime < 0] = 0
        
        return posteri_prime


    def update_priori_snr(self, book=False, first_iter=True):
        """
        From the paper:
        Scalart, P. and Filho, J. (1996). Speech enhancement based on a 
        priori signal to noise estimation. Proc. IEEE Int. Conf. Acoust., 
        Speech, Signal Processing, 629-632.
        
        Applied in MATLAB code from the book 
        Speech Enhancement: Theory and Practice (2013), by Philipos C. Loizou
        wiener_as.m
        """
        #only keep positive values (half-wave rectification)
        #may reduce in performance in near 0 SNRs
        self.posteri_prime_floored = self.calc_posteri_prime()
        
        if not book:
            #calculate according to apriori SNR equation (6) in paper
            #Scalart, P. and Filho, J. (1996)
            self.priori_snr = (1 - self.beta) * posteri_prime_floored + self.beta * self.posteri_snr
        else:
            #calculate following procedure from MATLAB code in book Loizou (2013)
            if first_iter:
                #don't yet have previous gain or snr values to apply
                self.priori_snr = self.beta + (1-self.beta) * self.posteri_prime_floored
            else:
                #now have previous gain and snr values
                self.priori_snr = self.beta * (self.gain_prev**2) * self.posteri_prev + (1 - self.beta) * self.posteri_prime_floored
        
        return None


    def update_gain(self):
        if self.gain_type.lower() == 'power estimation':
            #gain calculated in MATLAB code from the book 
            #Speech Enhancement: Theory and Practice (2013)
            #can be found in paper as equation (7), paired with equation (6)
            self.gain = np.sqrt(self.priori_snr/(1+self.priori_snr))
            
        #gain calculated from the paper for wiener 
        #Scalart & Filho (1996):
        elif self.gain_type.lower() == 'wiener':
            #can be found in paper as equation (8), paired with equation (6)
            self.gain = self.priori_snr / (1 + self.priori_snr)
        
        return None


    def apply_gain_fft(self, fft):
        enhanced_fft = fft * self.gain
        assert enhanced_fft.shape == fft.shape
        
        return enhanced_fft


    def save_wave(self,wavefile_name, signal_values):
        wavfile.write(wavefile_name, self.samprate, signal_values)
        
        return True
    
    
    def calc_noise_frame_len(self,SNR_decision):
        '''
        window for calculating moving average 
        for lower SNRs, larger window
        '''
        if SNR_decision < 1:
            soft_decision = 1 - (SNR_decision/self.threshold)
            soft_decision_scaled = round((soft_decision) * self.scale)
            noise_frame_len = 2 * soft_decision_scaled + 1
        else:
            noise_frame_len = SNR_decision
        
        return noise_frame_len
    
    
    def calc_linear_impulse(self,noise_frame_len, num_freq_bins):
        linear_filter_impulse = np.zeros((num_freq_bins,))
        for i in range(num_freq_bins):
            if i < noise_frame_len:
                linear_filter_impulse[i] = 1 / noise_frame_len
            else:
                linear_filter_impulse[i] = 0
            
        return linear_filter_impulse
    
    
    def calc_power_ratio(self, original_powerspec, noisereduced_powerspec):
        '''
        Where some issues happen.. just because power ratio is same, 
        louder noises don't get filtered out during speech
        Even though they have a different frequency makeup.
        '''
        power_ratio = sum(noisereduced_powerspec)/sum(original_powerspec)#/len(noisereduced_powerspec)
        
        
        return power_ratio
        
    
    def postfilter(self, original_powerspec, noisereduced_powerspec, threshold = 0.4, scale = 10):
        '''
        Goal: reduce musical noise
        
        From the paper:
        T. Esch and P. Vary, "Efficient musical noise suppression for speech enhancement 
        system," Proceedings of IEEE International Conference on Acoustics, Speech and 
        Signal Processing, Taipei, 2009.
        '''

        self.threshold = threshold
        self.scale = scale
        
        
        power_ratio_current_frame = self.calc_power_ratio(original_powerspec,noisereduced_powerspec)
        #is there speech? If so, SNR decision = 1
        if power_ratio_current_frame < threshold:
            SNR_decision = power_ratio_current_frame
        else:
            SNR_decision = 1
            
        noise_frame_len = self.calc_noise_frame_len(SNR_decision)
        #apply window
        postfilter_coeffs = self.calc_linear_impulse(noise_frame_len, self.frame_length)
        
        #Esch and Vary (2009) use convolution to adjust gain
        self.gain = np.convolve(self.gain,postfilter_coeffs,mode='valid')
        
        
        return None
        
    def calc_phase(self, fft_vals, normalization=False):
        '''
        Parameters
        ----------
        fft_vals : np.ndarray
            matrix with fft values [size = (num_fft, )]
            example (960, )
            
        Returns
        -------
        phase : np.ndarray
            Phase values for fft_vals [size = (num_fft,)]
            example (960, )
        '''
        # in radians
        #if normalization:
            #phase = np.angle(fft_vals) / (self.frame_length * self.norm_win)
        #else:
            #phase = np.angle(fft_vals)
        # not in radians
        # calculates mag /power and phase (power=1 --> mag 2 --> power)
        __, phase = librosa.magphase(fft_vals,power=2)
        return phase
    
    # TODO improve on this...
    def update_fft_length(self):
        if self.frame_length % 2 == 0:
            self.frame_length = self.frame_length
        else:
            raise TypeError('Frame length must be an even number. '+ \
                'Currently frame length {} is not acceptable.'.format(
                    self.frame_length))
        #if self.fft_bins < self.frame_length:
            #self.fft_bins *= 2
            #self.fft_bins = self.update_fft_length()
        return self.fft_bins
    

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
        self.update_fft_length()
        if 'linear' in self.band_spacing.lower():
            
            try:
            #calc number of bins per band
                assert self.frame_length / self.num_bands % 2 == 0
            except AssertionError:
                print("The number of bands must be equally divisible by the frame length.")
                sys.exit()
            self.bins_per_band = self.frame_length//self.num_bands
                
            band_start_freq = np.zeros((self.num_bands,))
            band_end_freq = np.zeros((self.num_bands,))
            try:
                for i in range(self.num_bands):
                    
                    band_start_freq[i] = int(i*self.bins_per_band)
                    band_end_freq[i] = int(band_start_freq[i] + self.bins_per_band)
            except TypeError:
                print(band_start_freq[i] + self.bins_per_band-1)
                sys.exit()
        elif 'log' in self.band_spacing.lower():
            pass
        elif 'mel' in self.band_spacing.lower():
            pass
        
        self.band_start_freq = band_start_freq
        self.band_end_freq = band_end_freq
        
        return None
    
    def update_posteri_bands(self,target_powspec, noise_powspec):
        '''
        MATLAB code from speech enhancement book uses power, 
        puts it into magnitude (via square root), then puts
        it back into power..? And uses some sort of 'norm' function...
        which I think is actually just the sum. Original equation 
        can be found in the paper below. 
        page 117 from book?
        
        paper:
        Kamath, S. D. & Loizou, P. C. (____), A multi-band spectral subtraction method for enhancing speech corrupted by colored noise.
        
        I am using power for the time being. 
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
        '''
        calculate over subtraction factor
        uses decibel SNR values calculated in update_posteri_bands()
        
        paper:
        Kamath, S. D. & Loizou, P. C. (____), A multi-band spectral subtraction method ofr enhancing speech corrupted by colored noise.
        '''
        a = np.zeros(self.snr_bands.shape[0])
        for band in range(self.num_bands):
            band_snr = self.snr_bands[band]
            if band_snr >= -5.0 and band_snr <= 20:
                a[band] += 4 - band_snr*3/20
            elif band_snr < -5.0:
                a[band] += 4.75
            else:
                a[band] += 1
        
        return a
    
    
    def calc_relevant_band(self,target_powspec):
        band_power = np.zeros(self.num_bands)
        for band in range(self.num_bands):
            start_bin = int(self.band_start_freq[band])
            end_bin = int(self.band_end_freq[band])
            target_band = target_powspec[start_bin:end_bin]
            band_power[band] += sum(target_band)
        rel_band = np.argmax(band_power)
        
        return rel_band
    
    
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
            relevant_band = self.calc_relevant_band(target_powspec)
        else:
            relevant_band = 0
        sub_signal = np.zeros((self.num_bands*self.bins_per_band,1))
        section = 0
        for band in range(self.num_bands):
            start_bin = int(self.band_start_freq[band])
            #print("start bin: ", start_bin)
            end_bin = int(self.band_end_freq[band])
            #print("end bin: ", end_bin)
            target_band = target_powspec[start_bin:end_bin]
            #print(target_band.shape)
            target_band = target_band.reshape(target_band.shape+(1,))
            noise_band = noise_powspec[start_bin:end_bin]
            #print(noise_band.shape)
            beta = oversub_factor[band] 
            #print(beta.shape)
            if band == relevant_band:
                delta = 1 #don't interfer too much with important target band
            else: 
                delta = 2.5 #less important bands --> more noise subtraction
            adjusted = target_band - beta * noise_band * delta
            #print("adjusted shape: ", adjusted.shape)
            sub_signal[section:section+self.bins_per_band] += adjusted
            self.apply_floor(sub_signal[section:section+self.bins_per_band], target_band, book=True)
            section += self.bins_per_band
            
        return sub_signal
            
            
    def reconstruct_spectrum(self, band_reduced_noise_matrix):
        total_rows = self.fft_bins
        output_matrix = np.zeros((total_rows,band_reduced_noise_matrix.shape[1]))
        print('band_reduced_noise_matrix : ',band_reduced_noise_matrix.shape)
        flipped_matrix = np.flip(band_reduced_noise_matrix)
        print('flipped_matrix', flipped_matrix.shape)
        output_matrix[0:self.fft_bins//2,:] += band_reduced_noise_matrix[0:self.fft_bins//2,:]#remove extra zeros at the end
        output_matrix[self.fft_bins//2:self.fft_bins,:] += flipped_matrix[self.fft_bins//2:self.fft_bins,:]#remove extra zeros at the beginning
        
        return output_matrix
    
    
    def apply_original_phase(self, spectrum, phase):
        
        #spectrum_complex = self.create_empty_matrix(spectrum.shape,complex_vals=True)
        ##spectrum = spectrum**(1/2)
        #phase_prepped = (1/2) * np.cos(phase) + cmath.sqrt(-1) * np.sin(phase)
        spectrum_complex = spectrum * phase
        
        return spectrum_complex
    
    def overlap_add(self, enhanced_matrix):
        start= self.frame_length - self.overlap
        mid= start + self.overlap
        stop= start + self.frame_length
        
        new_signal = self.create_empty_matrix(
            (self.overlap*(enhanced_matrix.shape[1]+1),),
            complex_vals=False)
        
        for i in range(enhanced_matrix.shape[1]):
            if i == 0:
                new_signal[:self.frame_length] += enhanced_matrix[:self.frame_length,i]
            else:
                new_signal[start:mid] += enhanced_matrix[:self.overlap,i] 
                new_signal[mid:stop] += enhanced_matrix[self.overlap:self.frame_length,i]
                start = mid
                mid = start+self.overlap
                stop = start+self.frame_length
        
        return new_signal
    
    
    def increase_volume(self,sample_values,minimum_max_val=0.13):
        sample_values *= 1.25
        if max(sample_values) < minimum_max_val:
            sample_values = self.increase_volume(sample_values,minimum_max_val)
        else:
            print('volume adjusted to {} '.format(max(sample_values)))
        
        return sample_values
    


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
