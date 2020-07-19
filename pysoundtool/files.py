'''The files module contains functions related to handling audio data files, for example loading audio files, saving audio files, and examing and reformatting audio files.
'''
import numpy as np
import random
import collections
import math 
import pathlib
from scipy.io.wavfile import write, read
from scipy.signal import resample
import soundfile as sf
import librosa

import os, sys, tarfile
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)

import pysoundtool as pyst



def loadsound(filename, sr=None, mono=True, dur_sec = None, use_scipy=False):
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
    try:
        sr2, data = read(filename)
        if sr:
            if sr2 != sr:
                data, sr2 = pyst.dsp.resample_audio(data, 
                                          sr_original = sr2, 
                                          sr_desired = sr)
                assert sr2 == sr
        else:
            sr = sr2
    except ValueError:
        print("Converting {} to wavfile".format(filename))
        try:
            filename = pyst.files.convert_audiofile(filename, overwrite=False)
        except RuntimeError as e:
            raise RuntimeError('Try setting `use_scipy` to False in pysoundtool.loadsound().')
        try:
            data, sr = loadsound(filename, sr=sr, mono=mono, dur_sec=dur_sec)
            print("File saved as {}".format(filename))
        except ValueError:
            print("Ensure bitdepth is compatible with scipy library")
            filename = pyst.files.newbitdepth(filename, overwrite=False)
            data, sr = loadsound(filename, sr=sr, mono=mono, dur_sec=dur_sec)
    
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

def savesound(audiofile_name, signal_values, sr, overwrite=False, use_scipy=False,
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
        # check to see if filename extension and format match:
        # if not, warning
        if 'format' in kwargs:
            if kwargs['format'].lower() != audiofile_name.suffix[1:]:
                audiofile_name_orig = audiofile_name
                audiofile_name = replace_ext(audiofile_name, kwargs['format'].lower())
                import warnings
                message = 'The desired format does not match the new file name: '+\
                    '\nDesired format: {}'.format(kwargs['format'])+\
                        '\nFilename: {}'.format(audiofile_name_orig) +\
                            '\nExtension will be adjusted to the format setting: '+\
                                '\n{}'.format(audiofile_name)
                warnings.warn(message)
        sf.write(audiofile_name, signal_values, sr, **kwargs)
    return audiofile_name

def get_file_format(audiofile):
    '''Use soundfile to get file format.
    '''
    so = sf.SoundFile(audiofile)
    return so.format

def list_possibleformats(use_scipy=False):
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
    possible_extensions = pyst.files.list_possibleformats(use_scipy=False)
    audiolist_checked = [x for x in audiolist if pathlib.Path(x).suffix in possible_extensions]
    if len(audiolist_checked) < len(audiolist):
        import warnings
        message = 'Some files did not match those acceptable by this program. '+\
            '(i.e. non-audio files) The number of files removed: '+\
                '{}'.format(len(audiolist)-len(audiolist_checked))
        warnings.warn(message)
    return audiolist_checked

def prep4scipywavfile(filename, overwrite=False):
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
            filename = convert_audiofile(filename, overwrite=overwrite)
            print("Saved file as {}".format(filename))
        elif 'bitdepth' not in str(filename):
            print("Ensuring bitdepth is compatible with scipy library")
            filename = newbitdepth(filename, overwrite=overwrite)
            print("Saved file as {}".format(filename))
        else:
            #some other error
            raise e
        filename = prep4scipywavfile(filename, overwrite=overwrite)
    return filename

def conversion_formats():
    '''Lists the formats available for conversion. 
    
    PySoundTool uses soundfile to convert files; therefore, whatever
    available formats soundfile has will be listed here.
    
    Examples
    --------
    >>> conversion_formats
    {'AIFF': 'AIFF (Apple/SGI)',
    'AU': 'AU (Sun/NeXT)',
    'AVR': 'AVR (Audio Visual Research)',
    'CAF': 'CAF (Apple Core Audio File)',
    'FLAC': 'FLAC (Free Lossless Audio Codec)',
    'HTK': 'HTK (HMM Tool Kit)',
    'SVX': 'IFF (Amiga IFF/SVX8/SV16)',
    'MAT4': 'MAT4 (GNU Octave 2.0 / Matlab 4.2)',
    'MAT5': 'MAT5 (GNU Octave 2.1 / Matlab 5.0)',
    'MPC2K': 'MPC (Akai MPC 2k)',
    'OGG': 'OGG (OGG Container format)',
    'PAF': 'PAF (Ensoniq PARIS)',
    'PVF': 'PVF (Portable Voice Format)',
    'RAW': 'RAW (header-less)',
    'RF64': 'RF64 (RIFF 64)',
    'SD2': 'SD2 (Sound Designer II)',
    'SDS': 'SDS (Midi Sample Dump Standard)',
    'IRCAM': 'SF (Berkeley/IRCAM/CARL)',
    'VOC': 'VOC (Creative Labs)',
    'W64': 'W64 (SoundFoundry WAVE 64)',
    'WAV': 'WAV (Microsoft)',
    'NIST': 'WAV (NIST Sphere)',
    'WAVEX': 'WAVEX (Microsoft)',
    'WVE': 'WVE (Psion Series 3)',
    'XI': 'XI (FastTracker 2)'}
    '''
    return sf.available_formats()

def convert_audiofile(filename, format_type=None, sr=None, new_dir=False, overwrite=False, use_scipy=False, **kwargs):
    '''Converts and saves soundfile as .wav type in same or new directory.
    
    Parameters
    ----------
    filename : str or pathlib.PosixPath
        The filename of the audiofile to be converted to .wav type
        
    format_type : str 
        The format to convert the audio file to. See pysoundtool.files.conversion_formats. 
        (defaults to 'wav')
    
    new_dir : str, pathlib.PosixPath, optional 
        If False, the converted files will be saved in same directory as originals.
        If a path is provided, the converted files will be saved there. If no such directory
        exists, one will be created.
    
    sr : int, optional
        The sample rate to be applied to the signal. If none supplied, the sample rate 
        of the original file will be used.
        
    **kwargs : additional keyword arguments
        The keyword arguments for soundfile.write:
        https://pysoundfile.readthedocs.io/en/latest/index.html?highlight=write#soundfile.write
    
        
    Returns 
    -------
    f_wavfile : pathlib.PosixPath
        The filename / path where the audio file is saved.
        
    Examples
    --------
    >>> audiofile = './example/audio.wav'
    # in same directory
    >>> audiofile_flac = pyst.files.convert_audiofile(audiofile, format_type='flac')
    >>> audiofile_flac
    PosixPath('example/audio.flac')
    # in new directory
    >>> audiofile_flac = pyst.files.convert_audiofile(audiofile, format_type='flac', 
                                                     new_dir = './examples2/')
    >>> audiofile_flac
    PosixPath('examples2/audio.flac')
    >>> # can establish desired conversion format in `new_dir`
    >>> audiofile_ogg = pyst.files.convert_audiofile(audiofile,
                                                     new_dir = './examples2/audio.ogg')
    >>> audiofile_ogg
    PosixPath('audiodata2/audio.ogg')
    
    See Also
    --------
    pysoundtool.files.conversion_formats
        Lists the possible formats to convert audio files.
        
    pysoundtool.files.list_possibleformats
        Lists the possible formats to load with pysoundtool.loadsound
    '''
    import pathlib
    import os
    try:
        f = pathlib.Path(filename)
    except TypeError:
        raise TypeError('Function convert_audiofile expected input of type string '+\
            'or a pathlib object, not type {}.'.format(type(filename)))
    if not f.suffix:
        raise TypeError('Function convert_audiofile expected a path with an '+\
            'audio extension, not input: \n', filename)
    if not f.suffix in pyst.files.list_possibleformats(use_scipy=False):
        raise TypeError('This software cannot process audio in {}'.format(f.suffix)+\
            ' format. We apologize for the inconvenience.')
    # ensure filename exists:
    if not os.path.exists(filename):
        raise FileNotFoundError('Could not find audio file at the following '+\
            'location\n{}'.format(filename))
    
    # set format_type to match suffix of new pathway (if it is not set already)
    if new_dir:
        new_dir = pathlib.Path(new_dir)
        if new_dir.suffix:
            if format_type is None:
                format_type = new_dir.suffix[1:].upper()
    
    # set default if format_type is None
    if format_type is None:
        format_type = 'WAV'
    
    # establish the path to save updated file.
    if new_dir:
        # check if new_dir is a directory or filename
        if new_dir.suffix:
            new_filename = new_dir
            new_dir = new_dir.parent
            # check for mismatch in file type and file name
            if format_type.lower() != new_filename.suffix[1:].lower():
                import warnings
                message = '\nWARNING: `format_type` {} '.format(format_type) +\
                    'does not match format of `new_dir`: \n{}.\n'.format(new_filename) + \
                        'Saving according to the `format_type`.'
                warnings.warn(message)
                new_filename = replace_ext(new_filename, format_type.lower())
        else:
            new_f = replace_ext(f, format_type.lower())
            # change filename to new directory
            new_filename = new_dir.joinpath(new_f.name)
        # check to make sure new_dir exists
        new_dir = pyst.utils.check_dir(new_dir, make=True)
    else:
        new_filename = replace_ext(f, format_type.lower())
        
    # load audio samples with soundfile, then save them as new audio file.
    try:
        data, sr = sf.read(filename, samplerate=sr)
    except RuntimeError as e:
        data, sr = librosa.load(filename, sr=sr)

    if not overwrite:
        if os.path.exists(new_filename):
            import warnings
            message = '\nWARNING: File {} already exists at this '.format(new_filename)+\
                'location. Set `overwrite` to True to overwrite it. Not overwritten.'
            warnings.warn(message)
            return new_filename
    if not use_scipy:
        format = format_type.upper()
        
    new_filename = pyst.savesound(new_filename, data, sr, 
                                  overwrite=overwrite, use_scipy=use_scipy,
                                  format=format)
    return new_filename

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
        if newname is None:
            newname = wave
        sf.write(newname, data, sr, subtype=newbit)
        savedname = newname
    else:
        if not newname:
            newname = adjustname(wave, adjustment='_bitdepth{}'.format(bitdepth))
            print("No new filename provided. Saved file as '{}'".format(newname))
            sf.write(newname, data, sr, subtype=newbit)
        elif newname:
            #make sure new extension matches original extension
            wave, newname = match_ext(wave, newname)
            sf.write(newname, data, sr, subtype=newbit)
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
