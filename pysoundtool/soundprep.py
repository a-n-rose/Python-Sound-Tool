'''PySoundTool: visualization and analysis of sound in Python
Copyright (C) 2019  Aislyn Rose

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.
'''

#Librosa and Soundfile are great Python libraries for analysing
#and manipulating sound. However, they are not importable in 
#Jupyter notebooks. These libraries may not work in for Jupyter
#notebooks; however if one has a soundfile they would like to use 
#in a Jupyter notebook, they can use these functions, which use
#Librosa and soundfile, to prepare the soundfile(s) for use a 
#Jupyter notebook with the scipy.io.wavfile module.

from scipy.io.wavfile import read
from scipy.signal import resample
import soundfile as sf
import librosa
import numpy as np


def normsound(samples,min_val=-1,max_val=1):
    '''Scales the input array to range between `min_val` and `max_val`
    '''
    samples = np.interp(samples,(samples.min(), samples.max()),(min_val, max_val))
    return samples


def loadsound(filename, samplerate=None, norm=True, mono=True):
    '''Loads sound file with scipy.io.wavfile.read
    
    If the sound file is not compatible with scipy's read module
    this functions converts the file to .wav format and/or
    changes the bit depth to be compatible. 
    
    Parameters
    ----------
    filename : str
        The filename of the sound to be loaded
    '''
    try:
        sr, data = read(filename)
        if samplerate:
            if sr != samplerate:
                data, sr = resample_audio(data, sr_original = sr, sr_desired = samplerate)
    except ValueError:
        print("Step 1: ensure filetype is compatible with scipy library".format(filename))
        filename = convert2wav(filename)
        try:
            data, sr = loadsound(filename)
            print("Success!")
        except ValueError:
            print("Step 2: ensure bitdepth is compatible with scipy library")
            filename = newbitdepth(filename)
            data, sr = loadsound(filename)
            print("Success!")
    if mono and len(data.shape) > 1:
        if data.shape[1] > 1:
            data = stereo2mono(data)
    if norm:
        data = normsound(data, -1, 1)
    return data, sr

def prep4scipywavfile(filename):
    '''Takes soundfile and saves it in a format compatible with scipy.io.wavfile
    
    Parameters
    ----------
    filename : str
        Filename of the soundfile to load with scipy.io.wavfile
    
    Returns
    -------
    filename : str
        Filename of the soundfile compatible with scipy.io.wavfile
    '''
    try:
        sr, data = read(filename)
        return filename
    except ValueError as e:
        import pathlib
        if pathlib.Path(filename).suffix.lower() != '.wav': 
            print("Converting file to .wav")
            filename = convert2wav(filename)
            print("Saved file as {}".format(filename))
        elif 'bitdepth' not in str(filename):
            print("Ensuring bitdepth is compatible with scipy library")
            filename = newbitdepth(filename)
            print("Saved file as {}".format(filename))
        else:
            #some other error
            raise e
        filename = prep4scipywavfile(filename)
    return filename

def convert2wav(filename, samplerate=None):
    '''Converts soundfile to .wav type 
    '''
    import pathlib
    f = pathlib.Path(filename)
    extension_orig = f.suffix
    if not extension_orig:
        f = str(f)
    else:
        f = str(f)[:len(str(f))-len(extension_orig)]
    f_wavfile = f+'.wav'
    if samplerate:
        data, sr = librosa.load(filename, sr=samplerate)
    else:
        data, sr = sf.read(filename)
    sf.write(f_wavfile, data, sr)
    return f_wavfile

def replace_ext(filename, extension):
    '''Adds or replaces an extension in the filename
    
    Parameters
    ----------
    filename : str or pathlib.PosixPath
        Filename with the missing or incorrect extension
    extension : str
        The correct extension for the given filename.
    
    Returns
    -------
    file_newext : str
        The filename with the new extension
    '''
    if isinstance(filename, str):
        import pathlib
        filename = pathlib.Path(filename)
    filestring = str(filename)[:len(str(filename))-len(filename.suffix)]
    if extension[0] != '.':
        extension = '.'+extension
    file_newext = filestring + extension
    return file_newext

def match_ext(filename1, filename2):
    '''Matches the file extensions. 
    
    If both have extensions, default set to that of `filename1`.
    '''
    import pathlib
    f1 = pathlib.Path(filename1)
    f2 = pathlib.Path(filename2)
    if not f1.suffix:
        if not f2.suffix:
            raise TypeError('No file extension provided. Check the filenames.')
        else: 
            extension = f2.suffix
    else: 
        extension = f1.suffix 
    if f1.suffix != extension:
        f1 = replace_ext(f1, extension)
    else:
        f1 = str(f1)
    if f2.suffix != extension:
        f2 = replace_ext(f2, extension)
    else:
        f2 = str(f2)
    return f1, f2

def newbitdepth(wave, bitdepth=16, newname=None, overwrite=False):
    '''Convert bitdepth to 16 or 32, to ensure compatibility with scipy.io.wavfile
    
    Scipy.io.wavfile is easily used online, for example in Jupyter notebooks.
    
    Reference
    ---------
    https://stackoverflow.com/questions/44812553/how-to-convert-a-24-bit-wav-file-to-16-or-32-bit-files-in-python3
    '''
    if bitdepth == 16:
        newbit = 'PCM_16'
    elif bitdepth == 32:
        newbit = 'PCM_32'
    else:
        raise ValueError('Provided bitdepth is not an option. Available bit depths: 16, 32')
    data, sr = sf.read(wave)
    if overwrite:
        sf.write(wave, data, sr, subtype=newbit)
        savedname = wave
    else:
        try:
            sf.write(newname, data, sr, subtype=newbit)
        except TypeError as e:
            if not newname:
                newname = adjustname(wave, adjustment='_bitdepth{}'.format(bitdepth))
                print("No new filename provided. Saved file as '{}'".format(newname))
                sf.write(newname, data, sr, subtype=newbit)
            elif newname:
                #make sure new extension matches original extension
                wave, newname = match_ext(wave, newname)
                sf.write(newname, data, sr, subtype=newbit)
            else:
                raise e
        savedname = newname
    return savedname

def adjustname(filename, adjustment=None):
    '''Adjusts filename.
    
    Parameters
    ----------
    filename : str
        The filename to be adjusted
    adjustment : str, optional
        The adjustment to add to the filename. If None, 
        the string '_adj' will be added.
    
    Returns
    -------
    fname : str 
        The adjusted filename with the original extension
        
    Examples
    --------
    >>> adjustname('happy.md')
    'happy_adj.md'
    >>> adjustname('happy.md', '_not_sad')
    'happy_not_sad.md'
    '''
    import pathlib
    f = pathlib.Path(filename)
    fname = f.stem
    if adjustment:
        fname += adjustment
    else:
        fname += '_adj'
    fname += f.suffix
    return fname

def resample_audio(samples, sr_original, sr_desired):
    '''Allows audio samples to be resampled to desired sample rate.
    '''
    time_sec = len(samples)/sr_original 
    num_samples = int(time_sec * sr_desired)
    resampled = resample(samples, num_samples)
    return resampled, sr_desired

def stereo2mono(data):
    '''If sound data has multiple channels, reduces to first channel
    
    Parameters
    ----------
    data : numpy.ndarray
        The series of sound samples, with 1+ columns/channels
    
    Returns
    -------
    data_mono : numpy.ndarray
        The series of sound samples, with first column
        
    Examples
    --------
    >>> import numpy as np
    >>> data = np.linspace(0,20)
    >>> data_2channel = data.reshape(25,2)
    >>> data_2channel[5]
    array([[0.        , 0.40816327],
       [0.81632653, 1.2244898 ],
       [1.63265306, 2.04081633],
       [2.44897959, 2.85714286],
       [3.26530612, 3.67346939]])
    >>> data_mono = stereo2mono(data)
    >>> data_mono[:5]
    array([0.        , 0.81632653, 1.63265306, 2.44897959, 3.26530612])
    '''
    data_mono = data.copy()
    if len(data.shape) > 1 and data.shape[1] > 1:
        data_mono = np.take(data,0,axis=-1) 
    return data_mono

def add_sound_to_signal(signal, sound, scale=1, delay_target_sec = 1, total_len_sec=None):
    '''Adds a sound (i.e. background noise) to a target signal 
    
    Parameters
    ----------
    signal : str 
        Sound file of the target sound
    sound : str 
        Sound file of the background noise or sound
    scale : int or float, optional
        The loudness of the sound to be added (default 1)
    delay_target_sec : int or float, optional
        Length of time in seconds the sound will be played before the target
        (default 1)
    total_len_sec : int or float, optional
        Total length of combined sound in seconds. If none, the sound will end
        after target sound ends (default None)
    
    Returns
    -------
    combined : numpy.ndarray
        The samples of the sounds added together
    sr : int 
        The sample rate of the samples
    '''
    target, sr = loadsound(signal)
    sound2add, sr2 = loadsound(sound)
    if sr != sr2:
        sound2add, sr = resample_audio(sound2add, sr2, sr)
    sound2add *= scale
    if total_len_sec:
        total_samps = int(sr*total_len_sec)
    else:
        total_samps = int(sr*(len(target)+delay_target_sec))
    if len(sound2add) < total_samps:
        sound2add = extend_sound(sound2add, total_samps)
    beginning_delay = sound2add[:int(sr*delay_target_sec)]
    noise2mix = sound2add[int(sr*delay_target_sec):len(target)+int(sr*delay_target_sec)]
    ending = sound2add[len(target)+int(sr*delay_target_sec):total_samps]
    combined = noise2mix + target
    if delay_target_sec:
        combined = np.concatenate((beginning_delay,combined))
    if total_len_sec:
        combined = np.concatenate((combined, ending))
    return combined, sr

def extend_sound(data, target_len):
    '''Extends a sound by repeating it until its `target_len`
    
    This is perhaps useful when working with repetitive or
    stationary sounds.
    '''
    new_data = np.zeros((target_len,))
    row_id = 0
    while row_id < len(new_data):
        if row_id + len(data) > len(new_data):
            diff = row_id + len(data) - len(new_data)
            new_data[row_id:] += data[:-diff]
        else:
            new_data[row_id:row_id+len(data)] += data
            row_id += len(data)
    return new_data


def zeropad_sound(data, target_len, sr, delay_sec=1):
    '''If the sound data needs to be a certain length, zero pad it.
    
    Parameters
    ----------
    data : numpy.ndarray
        The sound data that needs zero padding. Shape (len(data),). 
        Expects mono channel.
    target_len : int 
        The number of samples the `data` should have
    sr : int
        The samplerate of the `data`
    delay_sec : int, float, optional
        If the data should be zero padded also at the beginning.
        (default 1)
    
    Returns
    -------
    signal_zeropadded : numpy.ndarray
        The data zero padded to the shape (target_len,)
    '''
    if len(data.shape) > 1 and data.shape[1] > 1: 
        data = stereo2mono(data)
    delay_samps = sr * delay_sec
    if len(data) < target_len:
        diff = target_len - len(data)
        signal_zeropadded = np.zeros((data.shape[0] + int(diff)))
        for i, row in enumerate(data):
            signal_zeropadded[i+delay_samps] += row
    return signal_zeropadded

def combine_sounds(file1, file2, match2shortest=True, time_delay_sec=1,total_dur_sec=5):
    '''Combines sounds
    
    Parameters
    ----------
    file1 : str 
        One of two files to be added together
    file2 : str 
        Second of two files to be added together
    match2shortest : bool
        If the lengths of the addition should be limited by the shorter sound. 
        (defaul True)
    time_delay_sec : int, float, optional
        The amount of time in seconds before the sounds are added together. 
        The longer sound will play for this period of time before the shorter
        sound is added to it. (default 1)
    total_dur_sec : int, float, optional
        The total duration in seconds of the combined sounds. (default 5)
        
    Returns
    -------
    added_sound : numpy.ndarray
        The sound samples of the two soundfiles added together
    sr1 : int 
        The sample rate of the original signals and added sound
    '''
    data1, sr1 = loadsound(file1)
    data2, sr2 = loadsound(file2)
    if sr1 != sr2:
        data2, sr2 = resample_audio(data2, sr2, sr1)
    if time_delay_sec:
        num_extra_samples = int(sr1*time_delay_sec)
    else:
        num_extra_samples = 0
    if len(data1) > len(data2):
        data_long = data1
        data_short = data2
    else:
        data_long = data2
        data_short = data1
    dl_copy = data_long.copy()
    ds_copy = data_short.copy()
    
    if match2shortest:
        data_short = zeropad_sound(data_short, len(ds_copy) + num_extra_samples, sr1, delay_sec= time_delay_sec)
        data_long = data_long[:len(ds_copy)+num_extra_samples]
    else:
        data_short = zeropad_sound(data_short,len(dl_copy), sr1, delay_sec= time_delay_sec)
    added_sound = data_long + data_short
    
    if total_dur_sec:
        added_sound = added_sound[:sr1*total_dur_sec]
    return added_sound, sr1

def soundfile_limitduration(newfilename, soundfile, samplerate=None, 
                            dur_sec=None, overwrite=False):
    if samplerate:
        data, sr = librosa.load(soundfile,sr=samplerate, duration=dur_sec)
    else:
        data, sr = librosa.load(soundfile, duration=dur_sec)
    sf.write(newfilename, data, sr)
    
