
'''Filters module covers functions related to the filtering out of noise of
a target signal.
'''
###############################################################################
import os, sys
import inspect
import numpy as np

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
        samples = pyst.dsp.control_volume(samples, self.max_vol)
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
            posteri = pyst.dsp.create_empty_matrix(
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
        self.gain = pyst.filters.postfilter(target_power_frame,
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
                 zeropad = None):
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
    
def filtersignal(audiofile, 
                 sr = None,
                 noise_file=None,
                 filter_type='wiener', # 'band_specsub'
                 filter_scale=1,
                 apply_postfilter=False,
                 duration_noise_ms=120,
                 win_size_ms = None,
                 percent_overlap = None,
                 real_signal=False,
                 zeropad = False,
                 phase_radians=True,
                 num_bands=None,
                 visualize=False,
                 visualize_every_n_windows=50,
                 max_vol = 0.4,
                 min_vol = 0.15,
                 save2wav=False,
                 output_filename=None,
                 overwrite=False,
                 use_librosa=True):
    """Apply Wiener or band spectral subtraction filter to signal using noise. 

    Parameters 
    ----------
    audiofile : str, np.ndarray [size=(num_samples,) or (num_samples, num_channels)] 
        Filename or the audio data of the signal to be filtered.
    sr : int
        The sample rate of the audio. If `audiofile` is type np.ndarray, sr is 
        required. (default None)
    noise_file : str, tuple, optional
        Path to either noise audiofile or .npy file containing average power 
        spectrum values. If tuple, must include samples and sr.
        If None, the beginning of the `audiofile` will be used for noise data. 
        (default None)
    filter_type : str
        Type of filter to apply. Options 'wiener' or 'band_specsub'.
    filter_scale : int or float
        The scale at which the filter should be applied. This value will be multiplied 
        to the noise levels thereby increasing or decreasing the filter strength. 
        (default 1) 
    apply_postfilter : bool
        Whether or not the post filter should be applied. The post filter 
        reduces musical noise (i.e. distortion) in the signal as a byproduct
        of filtering.
    duration_noise_ms : int or float
        The amount of time in milliseconds to use from noise to apply the 
        Welch's method to. In other words, how much of the noise to use 
        when approximating the average noise power spectrum.
    win_size_ms : int or float 
        The length of window or frame for processing the audio in milliseconds.
        (default None)
    percent_overlap : int or float 
        The amount each window or frame should overlap. Typical is 0.5 or half the window.
        (default None)
    real_signal : bool 
        If True, only half of the (mirrored) fast Fourier transform will be used 
        during filtering. For audio, there is no difference. This is visible in the 
        plots, however, if you are interested. (default False)
    zeropad : bool, optional
        If False, only full frames of audio data are processed. 
        If True, last partial frame is zeropadded. (default False)
    phase_radians : bool 
        Relevant for band spectral subtraction: whether phase should be calculated in 
        radians or complex values/ power spectrum. (default True)
    num_bands : int
        Relevant for band spectral subtraction: the number of bands to section frequencies
        into. By grouping sections of frequencies during spectral subtraction filtering, 
        musical noise or distortion should be reduced. (defaults to 6)
    visualize : bool 
        If True, plots of the windows and filtered signal will be made. (default False)
    visualize_every_n_windows : int 
        If `visualize` is set to True, this controls how often plots are made: every 50
        windows, for example. (default 50)
    max_vol : int or float 
        The maximum volume level of the filtered signal. This is useful if you know you
        do not want the signal to be louder than a certain value. Ears are important
        (default 0.4) TODO improve on matching volume to original signal? At least use 
        objective measures.
    min_vol : int or float 
        The minimum volume level of the filtered signal. (default 0.15) 
        TODO improve on matching volume to original signal.
    save2wav : bool 
        If True, will save the filtered signal as a .wav file
    output_filename : str, pathlib.PosixPath, optional
        path and name the filtered signal is to be saved. (default None) If no filename
        provided, will save under date.
    overwrite : bool 
        If True and an audiofile by the same name exists, that file will be overwritten.
    use_librosa : bool 
        If True, audiofiles will be loaded using librosa. Otherwise, scipy.io.wavfile.
        (default True)
    
    Returns
    -------
    enhanced_signal : np.ndarray [size = (num_samples, )]
        The enhanced signal in raw sample form. Stereo audio has not yet been tested.
    sr : int 
        The sample rate of the enhanced/ filtered signal.
        
    References
    ----------
    Kamath, S. and Loizou, P. (2002). A multi-band spectral subtraction 
    method for enhancing speech corrupted by colored noise. Proc. IEEE Int.
    Conf. Acoust.,Speech, Signal Processing
    
    Kamath, S. and Loizou, P. (2006). mband.m MATLAB code from the book:
    
    C Loizou, P. (2013). Speech Enhancement: Theory and Practice. 
    """
    if 'wiener' in filter_type:
        if sr == 22050:
            import warnings
            warnings.warn('\n\nWARNING: sample rate 22050 may have some '+\
                'missing frames within the filtered signal. \nIf possible, '+\
                    'perhaps use 8000, 16000, 41000, or 48000 sample rate instead.\n')
        fil = pyst.WienerFilter(win_size_ms=win_size_ms, 
                                percent_overlap=percent_overlap,
                                sr=sr,
                                zeropad = zeropad)
    elif 'band' in filter_type:
        # at this point, band spectral subtraction only works with 
        if sr != 48000:
            import warnings
            warnings.warn('\n\nWARNING: Band spectral subtraciton requires a sample rate'+\
                ' of 48 kHz. Sample rate adjusted from {} to 48000.\n'.format(sr))
            sr = 48000
        fil = pyst.BandSubtraction(win_size_ms=win_size_ms,
                                   percent_overlap=percent_overlap,
                                   num_bands=num_bands,
                                   sr=48000,
                                   zeropad = zeropad)
    if visualize:
        frame_subtitle = 'frame size {}ms, window shift {}ms'.format(fil.frame_dur, int(fil.percent_overlap*fil.frame_dur))

    # load signal (to be filtered)
    if not isinstance(audiofile, np.ndarray):
        samples_orig, sr = pyst.loadsound(audiofile, fil.sr, dur_sec=None, use_librosa=use_librosa)
    else:
        samples_orig, sr = audiofile, sr
    if sr != fil.sr:
        samples_orig, sr = pyst.dsp.resample_audio(samples_orig, 
                                                   sr_original = sr, 
                                                   sr_desired = fil.sr)
    assert fil.sr == sr
    # TODO improve on volume control, improve SNR 
    # set volume max and min, or based on original sample data
    fil.set_volume(samples_orig, max_vol = max_vol, min_vol = min_vol)
    # set how many subframes are needed to process entire target signal
    fil.set_num_subframes(len(samples_orig), is_noise=False, zeropad=fil.zeropad)
    # prepare noise
    # set up how noise will be considered: either as audiofile, averaged
    # power values, or the first section of the target audiofile (i.e. None)
    samples_noise = None
    if noise_file:
        if isinstance(noise_file, tuple):
            # tuple must contain samples and sampling rate
            samples_noise, sr_noise = noise_file
            if sr_noise != fil.sr:
                samples_noise, sr_noise = pyst.dsp.resample_audio(samples_noise,
                                                                  sr_noise,
                                                                  fil.sr)
                assert sr_noise == fil.sr
        # ensure string objects converted to pathlib.PosixPath objects:
        elif not isinstance(noise_file, pathlib.PosixPath) and isinstance(noise_file, str):
            noise_file = pathlib.Path(noise_file)
        # find out path information
        if isinstance(noise_file, pathlib.PosixPath):
            extension = noise_file.suffix
            if '.npy' in extension:
                # if noise power spectrum already calculated or not
                if 'powspec' in noise_file.stem or 'powerspectrum' in noise_file.stem:
                    noise_power = fil.load_power_vals(noise_file)
                    samples_noise = None
                elif 'stft' in noise_file.stem:
                    noise_power = np.load(noise_file)
                    noise_power = np.abs(noise_power)**2
            else:
                # assume audio pathway
                if duration_noise_ms is not None:
                    dur_sec = duration_noise_ms/1000
                else:
                    dur_sec = None
                samples_noise, sr_noise = pyst.loadsound(noise_file, 
                                                         fil.sr, 
                                                         dur_sec=dur_sec,
                                                         use_librosa=use_librosa)
                assert sr_noise == fil.sr
        if samples_noise is None and noise_power is None:
            raise TypeError('Expected one of the following: '+\
                '\ntype tuple containing (samples, samplerate) of noise data'+\
                    '\naudiofile pathway to noise file'+\
                        '\n.npy file with powerspectrum values for noise'+\
                            '\n\nDid not expect {} as input.'.format(noise_file))

    else:
        starting_noise_len = pyst.dsp.calc_frame_length(fil.sr, 
                                                         duration_noise_ms)
        samples_noise = samples_orig[:starting_noise_len]
    # if noise samples have been collected...
    # TODO improve snr / volume measurements
    if samples_noise is not None:
        # set how many subframes are needed to process entire noise signal
        fil.set_num_subframes(len(samples_noise), is_noise=True, zeropad=fil.zeropad)
    if visualize:
        pyst.feats.plot(samples_orig, 'signal', title='Signal to filter'.upper(),sr = fil.sr)
        pyst.feats.plot(samples_noise, 'signal', title= 'Noise samples to filter out'.upper(), sr=fil.sr)
    # prepare noise power matrix (if it's not loaded already)
    if fil.noise_subframes:
        if real_signal:
            #only the first half of fft (+1)
            total_rows = fil.num_fft_bins//2+1
        else:
            total_rows = fil.num_fft_bins
        noise_power = pyst.dsp.create_empty_matrix((total_rows,))
        section = 0
        for frame in range(fil.noise_subframes):
            noise_section = samples_noise[section:section+fil.frame_length]
            noise_w_win = pyst.dsp.apply_window(noise_section, fil.get_window(),
                                                zeropad=fil.zeropad)
            noise_fft = pyst.dsp.calc_fft(noise_w_win, real_signal=real_signal)
            noise_power_frame = pyst.dsp.calc_power(noise_fft)
            noise_power += noise_power_frame
            section += fil.overlap_length
        # welch's method: take average of power that has been collected
        # in windows
        noise_power = pyst.dsp.calc_average_power(noise_power, 
                                                   fil.noise_subframes)
        assert section == fil.noise_subframes * fil.overlap_length
    if visualize:
        pyst.feats.plot(noise_power, 'stft',title='Average noise power spectrum'.upper()+'\n{}'.format(frame_subtitle), scale='power_to_db',
                             sr = fil.sr)
    
    # prepare target power matrix
    increment_length = int(fil.frame_length * fil.percent_overlap)
    total_rows =  increment_length + increment_length * fil.target_subframes
    filtered_sig = pyst.dsp.create_empty_matrix(
        (total_rows,), complex_vals=True)
    section = 0
    row = 0
    target_power_baseline = 0
    # increase/decrease noise values to increase strength of filter
    if not filter_scale:
        noise_power *= filter_scale
    else:
        filter_scale = 1
    try:
        for frame in range(fil.target_subframes):
            target_section = samples_orig[section:section+fil.frame_length]
            target_w_window = pyst.dsp.apply_window(target_section,
                                                    fil.get_window(), 
                                                    zeropad=fil.zeropad)
            if visualize and frame % visualize_every_n_windows == 0:
                pyst.feats.plot(target_section,'signal', title='Signal'.upper()+' \nframe {}: {}'.format( frame+1,frame_subtitle),sr = fil.sr)
                pyst.feats.plot(target_w_window,'signal', title='Signal with {} window'.format(fil.window_type).upper()+'\nframe {}: {}'.format( frame+1,frame_subtitle),sr = fil.sr)
            target_fft = pyst.dsp.calc_fft(target_w_window, real_signal=real_signal)
            target_power = pyst.dsp.calc_power(target_fft)
            # now start filtering!!
            # initialize SNR matrix
            if visualize and frame % visualize_every_n_windows == 0:
                pyst.feats.plot(target_power,'stft', title='Signal power spectrum'.upper()+'\nframe {}: {}'.format( frame+1,frame_subtitle), scale='power_to_db', sr = fil.sr)
            if 'wiener' in filter_type:
                enhanced_fft = fil.apply_wienerfilter(frame, 
                                                       target_fft, 
                                                       target_power, 
                                                       noise_power)
                if apply_postfilter:
                    enhanced_fft = fil.apply_postfilter(enhanced_fft,
                                                        target_fft,
                                                        target_power)
            elif 'band' in filter_type:
                target_phase = pyst.dsp.calc_phase(target_fft, 
                                                   radians=phase_radians)
                enhanced_fft = fil.apply_bandspecsub(target_power, 
                                                      target_phase, 
                                                      noise_power)
            
            enhanced_ifft = pyst.dsp.calc_ifft(enhanced_fft, 
                                               real_signal=real_signal)
            try:
                filtered_sig[row:row+fil.frame_length] += enhanced_ifft
            except ValueError:
                # with sample rate 22050, had some problems... zeropad missing frames
                if len(filtered_sig[row:row+fil.frame_length]) < fil.frame_length:
                    diff = fil.frame_length - len(filtered_sig[row:row+fil.frame_length])
                    filtered_sig[row:row+fil.frame_length] += \
                        enhanced_ifft[:fil.frame_length-diff]
                elif len(enhanced_ifft) < fil.frame_length:
                    diff = fil.frame_length - len(enhanced_ifft)
                    filtered_sig[row:row+fil.frame_length-diff] += enhanced_ifft
            if visualize and frame % visualize_every_n_windows == 0:
                pyst.feats.plot(filtered_sig,'signal', title='Filtered signal'.upper()+'\nup to frame {}: {}'.format(frame+1,frame_subtitle), sr = fil.sr)
            # prepare for next iteration
            if 'wiener' in filter_type:
                fil.posteri_snr_prev = fil.posteri_snr
                fil.gain_prev = fil.gain
            row += fil.overlap_length
            section += fil.overlap_length
    except ValueError as e:
        raise e
    assert row == fil.target_subframes * fil.overlap_length
    assert section == fil.target_subframes * fil.overlap_length
    # make enhanced_ifft values real
    enhanced_signal = filtered_sig.real
    if visualize:
        rows = len(filtered_sig)//increment_length
        cols = increment_length
        pyst.feats.plot(np.abs(filtered_sig.reshape((
                rows,cols,)))**2,
            'stft', title='Final filtered signal power spectrum'.upper()+'\n{}: {}'.format(filter_type,frame_subtitle), scale='power_to_db')
        pyst.feats.plot(enhanced_signal,'signal', title='Final filtered signal'.upper()+'\n{}'.format(filter_type), sr = fil.sr)
    enhanced_signal = fil.check_volume(enhanced_signal)
    if len(enhanced_signal) > len(samples_orig):
        enhanced_signal = enhanced_signal[:len(samples_orig)]
    # for backwards compatibility
    if output_filename is not None or save2wav:
        if output_filename is None:
            output_filename = pyst.utils.get_date()+'.wav'
        saved_filename = pyst.savesound(str(output_filename), 
                                        enhanced_signal,
                                        sr=fil.sr,
                                        overwrite=overwrite)
    return enhanced_signal, fil.sr

def postfilter(original_powerspec, noisereduced_powerspec, gain,
               threshold=0.4, scale=10):
    '''Apply filter that reduces musical noise resulting from other filter.
    
    If it is estimated that speech (or target signal) is present, reduced
    filtering is applied.

    References 
    ----------
    
    T. Esch and P. Vary, "Efficient musical noise suppression for speech enhancement 
    system," Proceedings of IEEE International Conference on Acoustics, Speech and 
    Signal Processing, Taipei, 2009.
    '''
    power_ratio_current_frame = pyst.dsp.calc_power_ratio(
        original_powerspec,
        noisereduced_powerspec)
    # is there speech? If so, SNR decision = 1
    if power_ratio_current_frame < threshold:
        SNR_decision = power_ratio_current_frame
    else:
        SNR_decision = 1
    noise_frame_len = pyst.dsp.calc_noise_frame_len(SNR_decision, threshold, scale)
    # apply window
    postfilter_coeffs = pyst.dsp.calc_linear_impulse(
        noise_frame_len,
        num_freq_bins = original_powerspec.shape[0])
    gain_postfilter = np.convolve(gain, postfilter_coeffs, mode='valid')
    return gain_postfilter


if __name__ == '__main__':
    import doctest
    doctest.testmod()
