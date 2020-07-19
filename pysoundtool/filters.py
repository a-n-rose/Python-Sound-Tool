
'''Filters module covers functions related to the filtering out of noise of
a target signal.
'''
###############################################################################
import numpy as np

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)
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
        sr, divided by 1000. (default 960)
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
    zeropad : bool, optional
        If False, only full frames of audio data are processed. 
        If True, the last partial frame will be zeropadded. (default False)
    """
    def __init__(self,
                 win_size_ms=None,
                 percent_overlap=None,
                 sr=None,
                 window_type=None,
                 zeropad = None):
        # set defaults if no values given
        self.frame_dur = win_size_ms if win_size_ms else 20
        self.percent_overlap = percent_overlap if percent_overlap else 0.5
        self.sr = sr if sr else 48000
        self.window_type = window_type if window_type else 'hamming'
        # set other attributes based on above values
        self.frame_length = pyst.dsp.calc_frame_length(
            self.frame_dur,
            self.sr)
        self.overlap_length = pyst.dsp.calc_num_overlap_samples(
            self.frame_length,
            self.percent_overlap)
        self.num_fft_bins = self.frame_length
        self.zeropad = zeropad or False
        
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
        belonging to the target signal (i.e. audiofile being filtered). Until
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
                 win_size_ms=None,
                 percent_overlap=None,
                 sr=None,
                 window_type=None,
                 max_vol = None,
                 zeropad = None):
        FilterSettings.__init__(self, 
                                win_size_ms=win_size_ms,
                                percent_overlap=percent_overlap,
                                sr=sr,
                                window_type=window_type,
                                zeropad = zeropad)
        self.max_vol = max_vol if max_vol else 0.4
        self.target_subframes = None
        self.noise_subframes = None

    # TODO remove
    def get_samples(self, audiofile, dur_sec=None):
        """Load signal and save original volume

        Parameters
        ----------
        audiofile : str
            Path and name of audiofile to be loaded
        dur_sec : int, float optional
            Max length of time in seconds (default None)

        Returns 
        ----------
        samples : ndarray
            Array containing signal amplitude values in time domain
        """
        samples, sr = pyst.loadsound(
            audiofile, self.sr, dur_sec=dur_sec)
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

    def set_num_subframes(self, len_samples, is_noise=False, zeropad=False):
        """Sets the number of target or noise subframes available for processing

        Parameters
        ----------
        len_samples : int 
            The total number of samples in a given signal
        is_noise : bool
            If False, subframe number saved under self.target_subframes, otherwise 
            self.noise_subframes (default False)
        zeropad : bool
            If False, number of frames limited to full frames. If True, last frame is zeropadded.

        Returns
        -------
        None
        """
        if is_noise:
            self.noise_subframes = pyst.dsp.calc_num_subframes(
                tot_samples=len_samples,
                frame_length=self.frame_length,
                overlap_samples=self.overlap_length,
                zeropad=zeropad
            )
        else:
            self.target_subframes = pyst.dsp.calc_num_subframes(
                tot_samples=len_samples,
                frame_length=self.frame_length,
                overlap_samples=self.overlap_length,
                zeropad=zeropad
            )
        return None

    def check_volume(self, samples):
        """ensures volume of filtered signal is within the bounds of the original
        """
        max_orig = round(max(samples), 2)
        samples = pyst.dsp.scalesound(samples, max_val=self.max_vol)
        max_adjusted = round(max(samples), 2)
        if max_orig != max_adjusted:
            print("volume adjusted from {} to {}".format(max_orig, max_adjusted))
        return samples
        
class WienerFilter(Filter):
    def __init__(self,
                 win_size_ms=None,
                 percent_overlap=None,
                 sr=None,
                 window_type=None,
                 max_vol = 0.4,
                 smooth_factor=0.98,
                 first_iter=None,
                 zeropad = None):
        Filter.__init__(self, 
                        win_size_ms=win_size_ms,
                        sr=sr,
                        window_type=window_type,
                        max_vol=max_vol,
                        zeropad = zeropad)
        self.beta = smooth_factor
        self.first_iter = first_iter
        self.gain = None
        
    def apply_wienerfilter(self, frame_index, target_fft, target_power_frame, noise_power):
        if frame_index == 0:
            # TODO: remove commented line
            #posteri = pyst.dsp.create_empty_matrix(
                #(len(target_power_frame),))
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
        # set attributes for next iteration
        self.gain_prev = self.gain
        self.posteri_snr_prev = self.posteri_snr
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
                 win_size_ms=None,
                 percent_overlap=None,
                 sr=None,
                 window_type=None,
                 max_vol = 0.4,
                 num_bands = 6,
                 band_spacing = 'linear',
                 zeropad = None,
                 smooth_factor=0.98,
                 first_iter=None):
        Filter.__init__(self, 
                        win_size_ms=win_size_ms,
                        sr=sr,
                        window_type=window_type,
                        max_vol=max_vol,
                        zeropad = zeropad)
        self.num_bands = num_bands
        self.band_spacing = band_spacing
        # Band spectral subtraction has been successful with 48000 sr.
        if self.sr != 48000:
            print('Band spectral subtraction requires a 48000 Hz sampling rate.'+\
                'Sampling rate is automatically adjusted accordingly.')
            self.sr = 48000
        # for applying the postfilter
        self.posteri_snr = None
        self.posteri_prime = None
        self.priori_snr = None
        self.beta = smooth_factor
        self.first_iter = first_iter
        self.gain = None
        
        
    def apply_bandspecsub(self, target_power, target_phase, noise_power):
        self.setup_bands()
        self.update_posteri_bands(target_power,noise_power)
        beta = self.calc_oversub_factor()
        reduced_noise_target = self.sub_noise(target_power, noise_power, beta)
        # perhaps don't need. TODO test this with real signal (ie half of fft)
        if len(reduced_noise_target) < len(target_power):
            reduced_noise_target = pyst.dsp.reconstruct_whole_spectrum(
                reduced_noise_target, n_fft = self.num_fft_bins)
        
        # apply original phase to reduced noise power 
        enhanced_fft = pyst.dsp.apply_original_phase(
            reduced_noise_target,
            target_phase)
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
        >>> fil = pyst.BandSubtraction()
        >>> fil.setup_bands()
        >>> fil.band_start_freq
        array([  0.,  80., 160., 240., 320., 400.])
        >>> fil.band_end_freq
        array([ 80., 160., 240., 320., 400., 480.])
        >>> # change default settings
        >>> fil = pyst.BandSubtraction(num_bands=5)
        >>> fil.setup_bands()
        >>> fil.band_start_freq
        array([  0.,  96., 192., 288., 384.])
        >>> fil.band_end_freq
        array([ 96., 192., 288., 384., 480.])
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
        >>> fil = pyst.BandSubtraction(num_bands=4)
        >>> fil.setup_bands()
        >>> # generate sine signal with and without noise
        >>> time = np.arange(0, 10, 0.01)
        >>> signal = np.sin(time)[:fil.frame_length]
        >>> np.random.seed(0)
        >>> noise = np.random.normal(np.mean(signal),np.mean(signal)+0.3,960)
        >>> powerspec_clean = np.abs(np.fft.fft(signal))**2
        >>> powerspec_noisy = np.abs(np.fft.fft(signal + noise))**2
        >>> fil.update_posteri_bands(powerspec_clean, powerspec_noisy)
        >>> fil.snr_bands
        array([ -1.91189028, -39.22078063, -44.16682922, -45.65265895])
        >>> # compare with no noise in signal:
        >>> fil.update_posteri_bands(powerspec_clean, powerspec_clean)
        >>> fil.snr_bands
        array([0., 0., 0., 0.])
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
        >>> fil = pyst.BandSubtraction(num_bands=4)
        >>> fil.setup_bands()
        >>> # generate sine signal with and without noise
        >>> time = np.arange(0, 10, 0.01)
        >>> signal = np.sin(time)[:fil.frame_length]
        >>> np.random.seed(0)
        >>> noise = np.random.normal(np.mean(signal),np.mean(signal)+0.3,960)
        >>> powerspec_clean = np.abs(np.fft.fft(signal))**2
        >>> powerspec_noisy = np.abs(np.fft.fft(signal + noise))**2
        >>> fil.update_posteri_bands(powerspec_clean, powerspec_noisy)
        >>> fil.snr_bands
        array([ -1.91189028, -39.22078063, -44.16682922, -45.65265895])
        >>> a = fil.calc_oversub_factor()
        >>> a
        array([4.28678354, 4.75      , 4.75      , 4.75      ])
        >>> # compare with no noise in signal:
        >>> fil.update_posteri_bands(powerspec_clean, powerspec_clean)
        >>> fil.snr_bands
        array([0., 0., 0., 0.])
        >>> a = fil.calc_oversub_factor()
        >>> a
        array([4., 4., 4., 4.])
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
        >>> fil = pyst.BandSubtraction(num_bands=4)
        >>> fil.setup_bands()
        >>> # generate sine signal with and with frequency 25
        >>> time = np.arange(0, 10, 0.01)
        >>> full_circle = 2 * np.pi
        >>> freq = 25
        >>> signal = np.sin((freq*full_circle)*time)[:fil.frame_length]
        >>> powerspec_clean = np.abs(np.fft.fft(signal))**2
        >>> rel_band_index, band_power_energies = fil.calc_relevant_band(powerspec_clean)
        >>> rel_band_index
        2
        >>> # and with frequency 50
        >>> freq = 50
        >>> signal = np.sin((freq*full_circle)*time)[:fil.frame_length]
        >>> powerspec_clean = np.abs(np.fft.fft(signal))**2
        >>> rel_band_index, band_power_energies = fil.calc_relevant_band(powerspec_clean)
        >>> rel_band_index
        3
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
        sub_power = np.zeros(target_powspec.shape)
        #sub_power = np.zeros((self.num_bands*self.bins_per_band,))
        #sub_power = np.zeros((self.num_bands*self.bins_per_band,))
        section = 0
        for band in range(self.num_bands):
            start_bin = int(self.band_start_freq[band])
            end_bin = int(self.band_end_freq[band])
            target_band = target_powspec[start_bin:end_bin]
            #target_band = np.expand_dims(target_band, axis=1)
            noise_band = noise_powspec[start_bin:end_bin]
            #noise_band = np.expand_dims(noise_band, axis=1)
            beta = oversub_factor[band] 
            if band == relevant_band:
                delta = 1 #don't interfer too much with important target band
            else: 
                delta = 2.5 #less important bands --> more noise subtraction
            adjusted = target_band - (beta  * noise_band * delta)
            start = section
            end = start + self.bins_per_band
            sub_power[start:end,] = adjusted
            sub_power[start:end,]  = self.apply_floor(
                sub_power[start:end,] , 
                target_band, book=True)
            section += self.bins_per_band
        # assert input and output shapes are same
        assert sub_power.shape == target_powspec.shape
        return sub_power
    
    def apply_postfilter(self, enhanced_fft, target_fft, 
                         target_power_frame, noise_power):
        if self.first_iter is not False:
            self.posteri_snr = pyst.dsp.calc_posteri_snr(target_power_frame,
                                                    noise_power)
            self.posteri_prime = pyst.dsp.calc_posteri_prime(self.posteri_snr)
            self.prior_snr = pyst.dsp.calc_prior_snr(snr = self.posteri_snr,
                                                snr_prime = self.posteri_prime,
                                                smooth_factor = self.beta,
                                                first_iter = True,
                                                gain = None)
            self.first_iter = False
        else:
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
        
        self.gain = pyst.dsp.calc_gain(self.prior_snr)
        target_noisereduced_power = pyst.dsp.calc_power(enhanced_fft)
        # update gain with postfilter:
        self.gain = pyst.dsp.postfilter(target_power_frame,
                                target_noisereduced_power,
                                gain=self.gain,
                                threshold=0.9,
                                scale=20)
        enhanced_fft = pyst.dsp.apply_gain_fft(target_fft, self.gain)
        # set attributes for next iteration
        self.gain_prev = self.gain
        self.posteri_snr_prev = self.posteri_snr
        return enhanced_fft


if __name__ == '__main__':
    import doctest
    doctest.testmod()
