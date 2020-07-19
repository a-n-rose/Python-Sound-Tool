'''The files module contains functions related to handling audio data files, for example loading audio files, saving audio files, and examing and reformatting audio files.
'''
import numpy as np
import random
import collections
import math 
import pathlib
from scipy.io.wavfile import write, read
from scipy.signal import resample
import librosa

import os, sys, tarfile
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)

import pysoundtool_online as pyst



def loadsound(filename, sr=None, mono=True, dur_sec = None, use_scipy=True):
    '''Loads sound file with scipy.io.wavfile.read or librosa.load (default librosa)
    
    Parameters
    ----------
    filename : str
        The filename of the sound to be loaded
    sr : int, optional
        The desired sample rate of the audio samples. If None, 
        the sample rate of the audio file will be used.
    mono : bool
        If True, the samples will be loaded in mono sound. If False,
        if the samples are in stereo, they will be loaded in stereo sound.
    dur_sec : int, float, optional
        The length in seconds of the audio signal.
    use_scipy : bool 
        If False, librosa will be used to load the audiofile. If True, 
        scipy.io.wavfile and/or soundfile will be used. If the sound file 
        is not compatible with scipy.io.wavfile.read, this functions converts 
        the file to .wav format and/or changes the bit depth to be compatible. 
        (default False)
        
    Returns
    -------
    data : nd.array [size=(num_samples,) or (num_samples, num_channels)]
        The normalized (between 1 and -1) sample data returned 
        according to the specified settings.
    sr : int 
        The sample rate of the loaded samples.
        
    See Also
    --------
    pysoundtool.files.prep4scipywavfile
        Prepares audio file for scipy.io.wavfile.read.
        
    pysoundtool.files.convert_audiofile
        Converts audio file to .wav format.
    
    pysoundtool.files.newbitdepth
        Converts audio file to specified bitdepth.
        
    pysoundtool.dsp.resample_audio
        Resampe audio data to a specified sample rate.
        
    pysoundtool.files.list_possibleformats
        Lists the possible formats to load with pysoundtool.loadsound
        
    librosa.load
        The package used to load sound data by default. See `librosa`.
        
    scipy.io.wavfile.read
        The package used to load sound if `use_scipy` is set to True.
        See `scipy`.
        
        
    Todo
    ----
    Make librosa data and scipy.io.wavfile data more similar
        https://stackoverflow.com/questions/54482346/reading-a-wav-file-with-scipy-and-librosa-in-python
    '''
    if not use_scipy:
        # the sample data will be a litle different from scipy.io.wavfile
        # as librosa does a litle extra work with the data
        data, sr = librosa.load(filename, sr=sr, mono=mono, duration=dur_sec)
        if mono is False and len(data.shape) > 1: 
            if data.shape[0] < data.shape[1]:
                # change shape from (channels, samples) to (samples, channels)
                data = data.T
        return data, sr
    sr2, data = read(filename)
    if sr:
        if sr2 != sr:
            data, sr2 = pyst.dsp.resample_audio(data, 
                                        sr_original = sr2, 
                                        sr_desired = sr)
            assert sr2 == sr
    else:
        sr = sr2
    # scipy loads data in shape (num_samples, num_channels)
    # don't need to transpose as for librosa
    if mono and len(data.shape) > 1:
        if data.shape[1] > 1:
            data = pyst.dsp.stereo2mono(data)
    # scale samples to be between -1 and 1
    data = pyst.dsp.scalesound(data, max_val= 1, min_val=-1)
    if dur_sec:
        numsamps = int(dur_sec * sr)
        data = pyst.dsp.set_signal_length(data, numsamps)
    return data, sr

def savesound(audiofile_name, signal_values, sr, overwrite=False, use_scipy=True,
              **kwargs):
    """saves the wave at designated path

    Parameters
    ----------
    audiofile_name : str or pathlib.PosixPath
        path and name the audio is to be saved under. (.wav format)
    signal_values : ndarray
        values of real signal to be saved
    sr : int 
        sample rate of the audio samples.
    **kwargs : additional keyword arguments
        The keyword arguments for soundfile.write:
        https://pysoundfile.readthedocs.io/en/latest/index.html?highlight=write#soundfile.write

    Returns
    -------
    audiofile_name : pathlib.PosixPath
        The new audiofile name
        
    See Also
    --------
    scipy.io.wavfile.write
    
    pysoundtool.files.conversion_formats
        Lists the possible formats to save audio files if `use_scipy` is False.
    """
    audiofile_name = pyst.utils.string2pathlib(audiofile_name)
    if os.path.exists(audiofile_name) and overwrite is False:
        raise FileExistsError('Filename {} already exists.'.format(audiofile_name)+\
            '\nSet `overwrite` to True in function savesound() to overwrite.')
    directory = audiofile_name.parent
    directory = pyst.utils.check_dir(directory, make=True)
    if use_scipy:
        write(audiofile_name, sr, signal_values)
    else: 
        raise pyst.VersionError()
    return audiofile_name

def get_file_format(audiofile):
    raise pyst.VersionError()

def list_possibleformats(use_scipy=True):
    if not use_scipy:
        return(['.wav', '.aiff', '.flac', '.ogg','.m4a','.mp3'])
    else:
        return(['.wav', '.aiff', '.flac', '.ogg'])

def list_audioformats():
    msg = '\nPySoundTool can work with the following file types: '+\
        ', '.join(get_compatible_formats())+ \
            '\nSo far, functionality does not work with the following types: '+\
                ', '.join(get_incompatible_formats())
    return msg

# TODO finish
def audiofiles_present(directory, recursive=False):
    '''Checks to see if audio files are present. 
    
    Parameters
    ----------
    directory : str or pathlib.PosixPath
        The directory to look for audio.
        
    recursive : bool
        If True, all nested directories will be checked as well. (default False)
        
    Returns
    -------
    bool 
        True if audio is present; otherwise False.
    '''
    directory = pyst.utils.string2pathlib(directory)

def collect_audiofiles(directory, hidden_files = False, wav_only=False, recursive=False):
    '''Collects all files within a given directory.
    
    This includes the option to include hidden_files in the collection.
    
    Parameters
    ----------
    directory : str or pathlib.PosixPath
        The path to where desired files are located.
    hidden_files : bool 
        If True, hidden files will be included. If False, they won't.
        (default False)
    wav_only : bool 
        If True, only .wav files will be included. Otherwise, no limit
        on file type. 
    
    Returns
    -------
    paths_list : list of pathlib.PosixPath objects
        Sorted list of file pathways.
    '''
    if not isinstance(directory, pathlib.PosixPath):
        directory = pathlib.Path(directory)
    paths_list = []
    # allow all data types to be collected (not only .wav)
    if wav_only:
        filetype = '*.wav'
    else: 
        filetype = '*'
    if recursive:
        filetype = '**/' + filetype
    for item in directory.glob(filetype):
        paths_list.append(item)
    # pathlib.glob collects hidden files as well - remove them if they are there:
    if not hidden_files:
        paths_list = [x for x in paths_list if x.stem[0] != '.']
    # ensure only audiofiles:
    paths_list = pyst.files.ensure_only_audiofiles(paths_list)
    return paths_list


def collect_zipfiles(directory, hidden_files = False, ext='tgz', recursive=False):
    '''Collects all zipfiles within a given directory.
    
    This includes the option to include hidden_files in the collection.
    
    Parameters
    ----------
    directory : str or pathlib.PosixPath
        The path to where desired files are located.
    hidden_files : bool 
        If True, hidden files will be included. If False, they won't.
        (default False)
    wav_only : bool 
        If True, only .wav files will be included. Otherwise, no limit
        on file type. 
    
    Returns
    -------
    paths_list : list of pathlib.PosixPath objects
        Sorted list of file pathways.
    '''
    if not isinstance(directory, pathlib.PosixPath):
        directory = pathlib.Path(directory)
    paths_list = []
    # allow all data types to be collected (not only .wav)
    if ext[0] == '.':
        filetype = '*' + ext
    else: 
        filetype = '*.' + ext
    if recursive:
        filetype = '**/' + filetype
    for item in directory.glob(filetype):
        paths_list.append(item)
    # pathlib.glob collects hidden files as well - remove them if they are there:
    if not hidden_files:
        paths_list = [x for x in paths_list if x.stem[0] != '.']
    return paths_list
    
def ensure_only_audiofiles(audiolist):
    possible_extensions = pyst.files.list_possibleformats(use_scipy=True)
    audiolist_checked = [x for x in audiolist if pathlib.Path(x).suffix in possible_extensions]
    if len(audiolist_checked) < len(audiolist):
        import warnings
        message = 'Some files did not match those acceptable by this program. '+\
            '(i.e. non-audio files) The number of files removed: '+\
                '{}'.format(len(audiolist)-len(audiolist_checked))
        warnings.warn(message)
    return audiolist_checked

def prep4scipywavfile(filename, overwrite=False):
    raise pyst.VersionError()

def conversion_formats():
    raise pyst.VersionError()

def convert_audiofile(filename, format_type=None, sr=None, new_dir=False, overwrite=False, use_scipy=True, **kwargs):
    raise pyst.VersionError()

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
    file_newext : pathlib.PosixPath
        The filename with the new extension
    '''
    if isinstance(filename, str):
        import pathlib
        filename = pathlib.Path(filename)
    #filestring = str(filename)[:len(str(filename))-len(filename.suffix)]
    f_dir = filename.parent
    f_name = filename.stem
    if extension[0] != '.':
        extension = '.'+extension
    new_filename = f_dir.joinpath(f_name + extension)
    return new_filename

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
    raise pyst.VersionError()

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

def extract(tar_url, extract_path='.'):
    tar = tarfile.open(tar_url, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])

def delete_dir_contents(directory, remove_dir = False):
    '''
    https://stackoverflow.com/a/28834214
    '''
    d = pyst.utils.string2pathlib(directory)
    for sub in d.iterdir():
        if sub.is_dir():
            delete_dir_contents(sub)
        else:
            sub.unlink()
    if remove_dir:
        d.rmdir()
        
def matching_filenames(list1, list_of_lists):
    list1_files = []
    if isinstance(list1[0], tuple):
        for item in list1:
            if len(item) != 2:
                # ensures expected length of 2: (encoded_label, pathway)
                raise ValueError('Expected a list of tuple pairs: encoded '+\
                    'label and associated pathway. Received tuple of length ', len(item))
            # checks to ensure encoded label comes first
            if isinstance(item[0], int) or isinstance(item[0], str) and item[0].isdigit():
                list1_files.append(item[1])
    elif isinstance(list1[0], str) or isinstance(list1[0], pathlib.PosixPath) or isinstance(list1[0], pathlib.PurePath):
        list1_files = list1
    other_lists_files = []
    for l in list_of_lists:
        if isinstance(l[0], tuple):
            for item in l:
                if len(item) != 2:
                    # ensures expected length of 2: (encoded_label, pathway)
                    raise ValueError('Expected a list of tuple pairs: encoded '+\
                        'label and associated pathway. Received tuple of length ', len(item))
                # checks to ensure encoded label comes first
                if isinstance(item[0], int) or isinstance(item[0], str) and item[0].isdigit():
                    # ensure pathway is string, not pathlib (for iteration purposes)
                    other_lists_files.append(str(item[1]))
        elif isinstance(l[0], str) or isinstance(l[0], pathlib.PosixPath) or isinstance(l[0], pathlib.PurePath):
            other_lists_files.append(l)
    # ensure list of lists is flat list
    flatten = lambda l: [item for sublist in l for item in sublist]
    other_lists_files = flatten(other_lists_files)
    contanimated_files = []
    for item in list1_files:
        if isinstance(item, str):
            item = pyst.string2pathlib(item)
        fname = item.stem
        fname_parts = fname.split('-')
        fname_head = fname_parts[:-1]
        fname_head = '-'.join(fname_head)
        for i in other_lists_files:
            if fname_head in i:
                contanimated_files.append(item)
                break
    return contanimated_files
        
def remove_contaminated_files(list1, contaminated_files):
    # ensure files are strings not pathlib.PosixPath objects
    contaminated_files = [str(x) for x in contaminated_files]
    remove_idx = []
    if isinstance(list1[0], tuple):
        for i, tuple_pair in enumerate(list1):
            pathway = str(tuple_pair[1])
            if pathway in contaminated_files:
                remove_idx.append(i)
    list_uncont = [x for j, x in enumerate(list1) if j not in remove_idx]
    return list_uncont
