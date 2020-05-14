'''
MATLAB script mband.m for band spectral subtraction applies square root when 
calculating the average noise power... 
line near 100 in MATLAB code: n_spect = sqrt(noise_pow/Noisefr);

could apply averaging to smooth spectrum, for now not 
search 'AVRGING' in MATLAB script mband.m

add VAD, for now none

Can also adjust delta weights... 
MATLAB code has low weights for first and last frequency bands
'''
###############################################################################
import sys
import numpy as np
from scipy.fftpack import fft, rfft, ifft, irfft
from scipy.io import wavfile
from scipy.signal import hamming, hanning
import librosa
import cmath
import copy


class Filter:
    def __init__(self, frame_duration = 20, 
                 percent_overlap = 50, 
                 filter_type = 'wiener', #or spectral subtraction
                 sampling_rate = 48000, 
                 window_type = 'hamming', 
                 real_signal = True,
                 num_bands = None,
                 band_spacing = 'linear',
                 sqrt=False):
        self.filter_type = filter_type
        self.frame_dur = frame_duration
        self.samprate = sampling_rate
        self.frame_length = frame_duration * sampling_rate //1000
        if percent_overlap > 1: 
            percent_overlap /= 100
        self.overlap = int(self.frame_length * percent_overlap) 
        self.common_length = self.frame_length-self.overlap #mband.m script
        self.sqrt = sqrt
        self.window = self.create_window(window_type)
        self.real_signal = real_signal
        self.num_bands = num_bands
        self.band_spacing = band_spacing
        self.fft_bins = 2
        
        
        


    def __repr__(self):
        return (f'{self.__class__.__name__}('
            f'\nFilter type: {self.filter_type!r} ms'
            f'\nFrame duration: {self.frame_dur!r} ms'
            f'\nSample rate: {self.samprate!r} hz'
            f'\nFrame length: {self.frame_length!r} samples'
            f'\nFrame overlap: {self.overlap!r} samples'
            f'\nWindow type: {self.window_type!r})'
            )
        
        
    def create_window(self, window_type):
        self.window_type = window_type
        if window_type.lower() == 'hamming':
            window = hamming(self.frame_length)
        elif window_type.lower() == 'hanning':
            window = hanning(self.frame_length)
        if self.sqrt:
            window = np.sqrt(window)
        self.norm_win = np.dot(window, window) / self.frame_length
        
        return window


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
            samples_win = copy.deepcopy(samples)
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


    def calc_power(self, fft_vals, normalization=False):
        if normalization:
            power_spec = np.abs(fft_vals)**2 / (self.frame_length * self.norm_win)
        else:
            power_spec = np.abs(fft_vals)**2
        
        return power_spec


    def save_wave(self,wavefile_name, signal_values):
        wavfile.write(wavefile_name, self.samprate, signal_values)
        
        return True
        
        
    def calc_phase(self, fft_vals, normalization=False):
        if normalization:
            phase = np.angle(fft_vals) / (self.frame_length * self.norm_win)
        else:
            phase = np.angle(fft_vals)
        
        return phase
        
        
    def update_fft_length(self):
        if self.fft_bins < self.frame_length:
            self.fft_bins *= 2
            self.fft_bins = self.update_fft_length()
        
        return self.fft_bins
    

    def setup_bands(self):
        self.update_fft_length()
        if 'linear' in self.band_spacing.lower():
            
            #calc number of bins per band
            self.bins_per_band = self.fft_bins//(2*self.num_bands)
            low_bins = np.zeros((self.num_bands,))
            high_bins = np.zeros((self.num_bands,))
            try:
                for i in range(self.num_bands):
                    
                    low_bins[i] = int(i*self.bins_per_band)
                    high_bins[i] = int(low_bins[i] + self.bins_per_band)
                    #bands_matrix[i] = self.bins_per_band
            except TypeError:
                print(low_bins[i] + self.bins_per_band-1)
                sys.exit()
        elif 'log' in self.band_spacing.lower():
            pass
        elif 'mel' in self.band_spacing.lower():
            pass
        
        self.low_bins = low_bins
        self.high_bins = high_bins
        
        return None
        
    
    def calc_oversub_factor(self,band_snr):
        '''
        calculate over subtraction factor
        uses decibel SNR values calculated in update_posteri_bands()
        
        paper:
        Kamath, S. D. & Loizou, P. C. (____), A multi-band spectral subtraction method ofr enhancing speech corrupted by colored noise.
        '''
        if band_snr >= -5.0 and band_snr <= 20:
            a = 4 - band_snr*3/20
        elif band_snr < -5.0:
            a = 4.75
        else:
            a = 1

        return a
    
    
    def calc_relevant_band(self,target_powspec):
        band_power = np.zeros(self.num_bands)
        for band in range(self.num_bands):
            start_bin = int(self.low_bins[band])
            end_bin = int(self.high_bins[band])
            target_band = target_powspec[start_bin:end_bin]
            band_power[band] += sum(target_band)
        rel_band = np.argmax(band_power)
        
        return rel_band
    
    
    def apply_floor(self, sub_band, original_band, floor=0.002, soften=False):
        for i, val in enumerate(sub_band):
            if val < 0:
                sub_band[i] = floor * original_band[i]
            if soften:
                sub_band[i] += 0.5*original_band[i]
        
        return sub_band
    
    
    def sub_noise(self,
                  target_powspec, 
                  noise_powspec,
                  speech=True,
                  first_iter=False):
        if not speech:
            relevant_band = self.calc_relevant_band(target_powspec)
        else:
            relevant_band = 0
        sub_signal = np.zeros((self.num_bands*self.bins_per_band,1))
        section = 0
        for band in range(self.num_bands):
            start_bin = int(self.low_bins[band])
            end_bin = int(self.high_bins[band])
            target_band = copy.deepcopy(target_powspec[start_bin:end_bin])
            target_mag = np.sqrt(target_band)
            target_norm = np.linalg.norm(target_mag,2)
            target_squared = target_norm**2
            target_band = target_band.reshape(target_band.shape+(1,))
            noise_band = copy.deepcopy(noise_powspec[start_bin:end_bin])
            noise_mag = np.sqrt(noise_band)
            noise_norm = np.linalg.norm(noise_mag,2)
            noise_squared = noise_norm**2
            snr = 10*np.log10(target_squared/noise_squared)

            beta = self.calc_oversub_factor(snr) 
            if band == relevant_band:
                delta = 1 #don't interfer too much with important target band
            else: 
                delta = 2.5 #less important bands --> more noise subtraction

            adjusted = target_band - beta * noise_band * delta
            sub_signal[section:section+self.bins_per_band] += adjusted
            sub_signal[section:section+self.bins_per_band] = self.apply_floor(sub_signal[section:section+self.bins_per_band], target_band, soften=False)
            section += self.bins_per_band
            
        return sub_signal
            
            
    ############################ Having trouble here #########################
            
    def reconstruct_spectrum(self, band_reduced_noise_matrix):
        total_rows = self.fft_bins
        output_matrix = np.zeros((total_rows,band_reduced_noise_matrix.shape[1]))
        flipped_matrix = np.flip(band_reduced_noise_matrix)
        output_matrix[0:self.fft_bins//2,:] += band_reduced_noise_matrix[:-1,:]#remove extra zeros at the end
        output_matrix[self.fft_bins//2:self.fft_bins,:] += flipped_matrix[1:,:]#remove extra zeros at the beginning
        
        return output_matrix
    
    
    
    def apply_original_phase(self, spectrum, phase):
        #TODO: make test
        spectrum_complex = self.create_empty_matrix(spectrum.shape,complex_vals=True)
        
        ##experimental and don't look good:
        ##phase_prepped = np.cos(phase) + (1j * np.sin(phase))
        ##phase_prepped = np.exp((np.cos(phase) + (1j * np.sin(phase)))/2)
        ##phase_prepped = np.exp((np.cos(phase) + (1j * np.sin(phase))))
        
        ##is there a difference? First one looks best... from MATLAB code
        ##1
        phase_prepped = (np.cos(phase) + cmath.sqrt(-1) * np.sin(phase))
        powspec_prepped = np.power(spectrum,1/2) #don't know why 
        spectrum_complex = powspec_prepped * phase_prepped
        
        ##2
        #phase_prepped = (np.cos(phase) + (1j * np.sin(phase)))/2
        #spectrum_complex = spectrum * phase_prepped
        
        return spectrum_complex
    
    
    def overlap_add(self, enhanced_matrix, complex_vals=False):
        start= self.frame_length - self.overlap
        mid= start + self.overlap
        stop= start + self.frame_length

        new_signal = self.create_empty_matrix(
            (self.overlap*(enhanced_matrix.shape[1]+1),),
            complex_vals=complex_vals)

        for i in range(enhanced_matrix.shape[1]):
            if i == 0:
                if complex_vals:
                    new_signal[:self.frame_length] += enhanced_matrix[:self.frame_length,i]
                else:
                    new_signal[:self.frame_length] += enhanced_matrix[:self.frame_length,i].real
            else:
                if complex_vals:
                    new_signal[start:mid] += enhanced_matrix[:self.overlap,i] 
                    new_signal[mid:stop] += enhanced_matrix[self.overlap:self.frame_length,i]
                else:
                    new_signal[start:mid] += enhanced_matrix[:self.overlap,i].real 
                    new_signal[mid:stop] += enhanced_matrix[self.overlap:self.frame_length,i].real
                start = mid
                mid = start+self.overlap
                stop = start+self.frame_length
                
        return new_signal
    
    
def get_starting_noise(samples,dur_ms,sr):
    clip = sr//1000*dur_ms
    noise = samples[:clip]
    return noise
    
def make_1channel(samples):
    if len(samples.shape) > 1 and samples.shape[1] > 1:
        samples = samples[:,0]
    return samples
    

def apply_mband(output_wave_name, 
             target_wav, 
             window_type='hamming', 
             sampling_rate = 48000):
    
    wf = Filter(filter_type='spectral subtraction',
                sampling_rate=sampling_rate,
                window_type = window_type,
                real_signal = False,
                num_bands = 6,
                band_spacing = 'linear',
                sqrt = False) 
    
    print(repr(wf))

    target = wf.load_signal(target_wav)
    #if multiple channels, reduce to first channel
    target = make_1channel(target)
    #only use 120 ms of noise
    noise = get_starting_noise(target,dur_ms=120,sr=wf.samprate)

    wf.set_num_subframes(len(noise),noise=True)
    wf.set_num_subframes(len(target),noise=False)
    
    #noise_power_matrix = wf.create_empty_matrix(wf.window.shape+(1,),complex_vals=False)
    wf.setup_bands()
    noise_power_matrix = wf.create_empty_matrix((wf.fft_bins,1,),complex_vals=False)
    #calculate and collect power of noise
    section = 0 
    for frame in range(wf.noise_subframes):

        noise_sect = noise[section:section+wf.frame_length]
        noise_w_win = wf.apply_window(noise_sect)
        noise_fft = wf.calc_fft(noise_w_win,fft_length=wf.fft_bins)
        noise_mag = np.abs(noise_fft)
        noise_power = wf.calc_power(noise_fft, normalization=False)
        for i, row in enumerate(noise_power):
            noise_power_matrix[i] += row 
        section += wf.overlap #MATLAB code doesn't overlap, uses 6 frames; I use 11 overlapping and results in basically same magnitude
        
        n_ph = wf.calc_phase(noise_fft,normalization=False)
    noise_power_matrix = wf.calc_average_power(noise_power_matrix,wf.noise_subframes) 
    noise_mag_matrix = np.sqrt(noise_power_matrix)
    phase_matrix = wf.create_empty_matrix((wf.fft_bins,wf.target_subframes))

    total_rows = wf.fft_bins//2+1
    enhanced_signal = wf.create_empty_matrix((total_rows,wf.target_subframes), complex_vals=False)
    
    section = 0
    for frame in range(wf.target_subframes):
        sub_speech = np.zeros((wf.bins_per_band,))
        target_section = copy.deepcopy(target[section:section + wf.frame_length])
        target_w_win = wf.apply_window(target_section)
        target_fft = wf.calc_fft(target_w_win,fft_length=wf.fft_bins)
        target_power = wf.calc_power(target_fft, normalization=False)
        target_mag = np.abs(target_power)
        
    
        target_phase = wf.calc_phase(target_fft,normalization=False)
        phase_matrix[:,frame] += target_phase
        if frame == 0:
            first_iter=True
        else:
            first_iter=False
        reduced_noise_target = wf.sub_noise(target_power, noise_power_matrix,first_iter=first_iter)

        reduced_noise_target = reduced_noise_target.transpose()

        for i, row in enumerate(reduced_noise_target[0]):
            enhanced_signal[i][frame] += row 
        section += wf.overlap

    enhanced_signal = wf.reconstruct_spectrum(enhanced_signal)
    enhanced_signal = wf.apply_original_phase(enhanced_signal,phase_matrix)
    enhanced_signal = wf.calc_ifft(enhanced_signal)
    enhanced_signal = enhanced_signal
    
    #overlap add:
    enhanced_signal = wf.overlap_add(enhanced_signal,complex_vals=True)
    wf.save_wave(output_wave_name,enhanced_signal.real)
    return True



if __name__=='__main__':
    outputwave = 'band_spectral_subtraction_short_test.wav'
    apply_mband(outputwave,"./short_speech_noise.wav")
    
