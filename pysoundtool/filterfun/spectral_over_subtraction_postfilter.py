'''
This is an approximation of spectral over-substraction.
The amoount noise is subtracted can be increased or decreased with the 'strength' variable.
There is a bit of musical noise in reduced noise result. 
It is, however, pretty simple to implement.

2 wave samples are required: samples of target recording as well as samples depicting the noise in that signal are required.

'''

import numpy as np
import librosa
import math
import random



def rednoise_pf(samples_recording,samples_noise, sampling_rate, strength, vad=None, postfilter=False, threshold=0.4, scale=10):
    '''
    calculates the power in noise signal and subtracts that from the recording
    vad = voice activity detection (cut samples to where voice activity begins)
    
    returns recording samples with reduced noise
    '''
    
    #1) time domain to frequency domain:
    #get the short-time fourier transform (STFT) of noise and recording
    stft_n = samps2stft(samples_noise,sampling_rate)
    stft_r = samps2stft(samples_recording, sampling_rate)
    
    #2) calculate the power
    power_n = stft2power(stft_n)
    power_r = stft2power(stft_r)
    
    #3) calculate the power mean, and power variance of noise
    power_mean_n = get_mean_bandwidths(power_n)
    power_var_n = get_var_bandwidths(power_n)
    
    #4) subtract noise from recording:
    #using list comprehension to work through all samples of recording
    stft_r_rednoise = np.array([subtract_noise(power_mean_n,power_var_n,power_r[i],stft_r[i],strength) for i in range(stft_r.shape[0])])

    #5) detect speech and where when it starts:
    energy_r = get_energy_rms(stft_r)
    energy_r_mean = get_energy_mean(energy_r)
    
    
    if vad:
        stft_r_rednoise = voice_activity_detection(stft_r_rednoise, energy_r, energy_r_mean)       
        
    if postfilter:
        power_r_rn = stft2power(stft_r_rednoise)
        coefficient_matrix = calc_filter_coeffs(power_r, power_r_rn, power_n, threshold, scale)
        print(coefficient_matrix)
        stft_r_rednoise = stft_r * coefficient_matrix
        #stft_r_rednoise *= coefficient_matrix
    ##save this to see if it worked:
    stft_r_rednoise= stft2samps(stft_r_rednoise, len(samples_recording))
    
    return stft_r_rednoise


def calc_filter_coeffs(power_orig, power_rednoise, power_noise, threshold, scale):
    # go through time-frame by time-frame 
    # for now, time is rows, and frequencies are columns
    #postfilter_coeffs_matrix = np.zeros((power_orig.shape))
    gain_matrix = np.zeros((power_orig.shape))
    
    for i, row in enumerate(power_orig):
        pow_ratio = calc_power_ratio(row,power_rednoise[i])
        if pow_ratio < threshold:
            SNR_decision = pow_ratio
        else:
            SNR_decision = 1
        
        noise_frame_len = calc_noise_frame_len(SNR_decision, threshold, scale)
        postfilter_coeffs = calc_linear_impulse(noise_frame_len, power_rednoise.shape[1])
        #postfilter_coeffs_matrix[i] += postfilter_coeffs
        
        priori_snr = calc_priori(power_rednoise[i],power_noise[i])
        gains = calc_gain(priori_snr)
        #gain_matrix[i] += gains
        gain_matrix[i] += np.convolve(gains,postfilter_coeffs, mode='valid')
        
    return gain_matrix
    

def samps2stft(y, sr):
    if len(y)%2 != 0:
        y = y[:-1]
    #print("shape of samples: {}".format(y.shape))
    stft = librosa.stft(y)
    #print("shape of stft: {}".format(stft.shape))
    #D : np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
    stft = np.transpose(stft)
    #print("transposed shape: {}".format(stft.shape))
    #rows = time/frames, columns = frequency bin
    return stft


def stft2samps(stft,len_origsamp):
    #print("shape of stft: {}".format(stft.shape))
    istft = np.transpose(stft.copy())
    ##print("transposed shape: {}".format(istft.shape))
    samples = librosa.istft(istft,length=len_origsamp)
    return samples

def stft2power(stft_matrix):
    if stft_matrix is not None:
        if len(stft_matrix) > 0:
            stft = stft_matrix.copy()
            power = np.abs(stft)**2
            return power
        else:    
            raise TypeError("STFT Matrix is empty. Function 'stft2power' needs a non-empty matrix.")
    else:
        raise TypeError("STFT Matrix does not exist. Function 'stft2power' needs an existing matrix.")
    return None

    
def get_energy_rms(stft_matrix):
    #stft.shape[1] == bandwidths/frequencies
    #stft.shape[0] pertains to the time domain
    rms_list = [np.sqrt(sum(np.abs(stft_matrix[row])**2)/stft_matrix.shape[1]) for row in range(len(stft_matrix))]
    return rms_list

def get_energy_mean(energy_list):
    mean = sum(energy_list)/len(energy_list)
    return mean

def get_mean_bandwidths(matrix_bandwidths):
    bw = matrix_bandwidths.copy()
    bw_mean = [np.mean(bw[:,bandwidth]) for bandwidth in range(bw.shape[1])]
    return bw_mean

def get_var_bandwidths(matrix_bandwidths):
    if len(matrix_bandwidths) > 0:
        bw = matrix_bandwidths.copy()
        bw_var = [np.var(bw[:,bandwidth]) for bandwidth in range(bw.shape[1])]
        return bw_var
    return None


def subtract_noise(noise_powerspec_mean,noise_powerspec_variance, speech_powerspec_row,speech_stft_row,strength):
    npm = noise_powerspec_mean
    npv = noise_powerspec_variance
    spr = speech_powerspec_row
    stft_r = speech_stft_row.copy()
    for i in range(len(spr)):
        if spr[i] <= npm[i] + strength * npv[i]:
            stft_r[i] = 1e-3
    return stft_r

def voice_activity_detection(stft, energy_matrix, energy_mean, start=True):
    voice_start,voice = sound_index(energy_matrix,energy_mean,start=True,)
    if voice:
        #print("Speech detected at index: {}".format(voice_start))
        stft = stft[voice_start:]
        
    else:
        print("No speech detected.")
    return stft

  
def suspended_energy(speech_energy,speech_energy_mean,row,start):
    try:
        if start == True:
            if row <= len(speech_energy)-4:
                if speech_energy[row+1] and speech_energy[row+2] and speech_energy[row+3] > speech_energy_mean:
                    return True
        else:
            if row >= 3:
                if speech_energy[row-1] and speech_energy[row-2] and speech_energy[row-3] > speech_energy_mean:
                    return True
    except IndexError as ie:
        return False

def sound_index(speech_energy,speech_energy_mean,start = True):
    if start == True:
        side = 1
        beg = 0
        end = len(speech_energy)
    else:
        side = -1
        beg = len(speech_energy)-1
        end = -1
    for row in range(beg,end,side):
        if speech_energy[row] > speech_energy_mean:
            if suspended_energy(speech_energy, speech_energy_mean, row,start=start):
                if start==True:
                    #to catch plosive sounds
                    while row >= 0:
                        row -= 1
                        row -= 1
                        if row < 0:
                            row = 0
                        break
                    return row, True
                else:
                    #to catch quiet consonant endings
                    while row <= len(speech_energy):
                        row += 1
                        row += 1
                        if row > len(speech_energy):
                            row = len(speech_energy)
                        break
                    return row, True
    else:
        pass
    return beg, False


def calc_noise_frame_len(SNR_decision, threshold, scale):
    '''
    window for calculating moving average 
    for lower SNRs, larger window
    '''
    if SNR_decision < 1:
        soft_decision = 1 - (SNR_decision/threshold)
        soft_decision_scaled = round((soft_decision) * scale)
        noise_frame_len = abs(2 * soft_decision_scaled + 1)
    else:
        noise_frame_len = SNR_decision
    
    return noise_frame_len


def calc_priori(redniose_power,noise_power):
    priori_snr = redniose_power/noise_power
    
    return priori_snr


def calc_gain(priori_snr):
    gain = np.zeros(priori_snr.shape)
    for i, snr in enumerate(priori_snr):
        #gain[i] += np.sqrt(snr / (1+snr))
        gain[i] +=  snr / (1 + snr)
    return gain


def calc_linear_impulse(noise_frame_len, num_freq_bins):
    #postfilter_coeffs = np.zeros(noise_frame_len.shape)
    #for i, row in enumerate(noise_frame_len):
        #if target_fft[i] < row:
            #postfilter_coeffs[i] = 1/(noise_frame_len[i])
        #else:
            #postfilter_coeffs[i] = 0
    #applying moving filter with certain frame lengths same as
    #linear filter impulse reponse:
    linear_filter_impulse = np.zeros((num_freq_bins,))
    #print(num_freq_bins)
    for i in range(num_freq_bins):
        #print(i)
        #print(noise_frame_len)
        if i < noise_frame_len:
            linear_filter_impulse[i] = 1 / noise_frame_len
        else:
            linear_filter_impulse[i] = 0
        
    return linear_filter_impulse


def calc_power_ratio(original_powerspec, noisereduced_powerspec):
    '''
    Where some issues happen.. just because power ratio is same, 
    louder noises don't get filtered out during speech
    Even though they have a different frequency makeup.
    '''
    power_ratio = sum(noisereduced_powerspec)/sum(original_powerspec)#/len(noisereduced_powerspec)
    #print(power_ratio)
    
    return power_ratio

    

def postfilter(original_powerspec, noisereduced_powerspec, threshold = 0.4, scale = 10):
    '''
    Goal: reduce musical noise
    
    From the paper:
    T. Esch and P. Vary, "Efficient musical noise suppression for speech enhancement 
    system," Proceedings of IEEE International Conference on Acoustics, Speech and 
    Signal Processing, Taipei, 2009.
    '''
    
    
    power_ratio_current_frame = calc_power_ratio(original_powerspec,noisereduced_powerspec)
    #is there speech? If so, SNR decision = 1
    if power_ratio_current_frame < threshold:
        SNR_decision = power_ratio_current_frame
    else:
        SNR_decision = 1
        
    noise_frame_len = calc_noise_frame_len(SNR_decision, threshold, scale)
    #apply window
    postfilter_coeffs = calc_linear_impulse(noise_frame_len, frame_length)
    
    
    return postfilter_coeffs
        
