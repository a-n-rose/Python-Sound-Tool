'''Useful functions for PySoundTool but not directly related to sound data.
''' 
import os, sys
import numpy as np
import datetime
import pathlib

def str2path(pathstring):
    '''Turns string path into pathlib.PosixPath object
    
    Parameters
    ----------
    pathstring : str
        The pathway to be turned into a pathlib object
        
    Returns
    -------
    pathlibpath : pathlib.PosixPath
        The pathway as a pathlib object.
    '''
    pathlibpath = pathlib.Path(pathstring)
    return pathlibpath

# TODO make str path into Pathlib.PosixPath
def path_or_samples(input_value):
    '''Checks whether `input_value` is a path or sample data. Does not check path validity.
    
    This is useful for functions that take both pathways to audio as well as 
    pre-loaded audio data.
    
    Parameters
    ----------
    input_value : str, pathlib.PosixPath, or tuple [size= ( (samples,), sr)]
    
    Returns
    -------
    'path' or 'samples' : str 
    
    Examples
    --------
    >>> import numpy as np
    >>> # create some example samples and sample rate
    >>> samples = np.array([1,2,3,2,1,0])
    >>> sr = 5
    >>> path_or_samples( (samples, sr) )
    'samples'
    >>> # expects both audio samples and sr
    >>> path_or_samples(samples)
    TypeError: The input for `path_or_samples` expected a str, pathlib.PosixPath, or tuple with samples and sample rate, not type <class 'numpy.ndarray'>
    >>> # create example string pathway
    >>> path_or_samples('my_audio.wav')
    'path'
    >>> # create pathlib.PosixPath object 
    >>> import pathlib
    >>> path_or_samples(pathlib.Path('my_audio.wav')
    'path'
    '''
    if isinstance(input_value, str):
        return 'path'
    elif isinstance(input_value, pathlib.PosixPath):
        return 'path'
    elif isinstance(input_value, tuple):
        if isinstance(input_value[0], np.ndarray):
            return 'samples'
    else:
        raise TypeError('The input for `path_or_samples` expected a str, '+\
            'pathlib.PosixPath, or tuple with samples and sample rate, '+\
                'not type {}'.format(type(input_value)))
    
def match_dtype(array1, array2):
    '''Match the dtype of the second array to the first.
    
    Parameters
    ----------
    array1 : np.ndarray
        The numpy array with the dataype to be adjusted and returned.
    array2 : np.ndarray
        The numpy array with the orginal or desired datatype.

        
    Returns
    -------
    array1 : np.ndarray 
        The `array1` with the dtype of `array2`
    '''
    array1 = array1.astype(array2.dtype)
    assert array1.dtype == array2.dtype
    return array1
    
# TODO move to dsp?
def shape_samps_channels(data):
    '''Returns data in shape (num_samps, num_channels)
    
    Parameters
    ----------
    data : np.ndarray [size= (num_samples,) or (num_samples, num_channels), or (num_channels, num_samples)]
        The data that needs to be checked for correct format 
    
    Returns
    -------
    data : np.ndarray [size = (num_samples,) or (num_samples, num_channels)]
    '''
    if len(data.shape) == 1:
        return data
    if len(data.shape) > 2:
        raise ValueError('Expected 2 dimensional data: (num_samples, num_channels,) not '+\
            'shape {}'.format(data.shape))
    if data.shape[0] < data.shape[1]:
        # assumes number of samples will be greater than number of channels
        data = data.T
    assert data.shape[0] > data.shape[1] 
    return data
    
def get_date():
    time = datetime.datetime.now()
    time_str = "{}m{}d{}h{}m{}s{}ms".format(time.month,
                                        time.day,
                                        time.hour,
                                        time.minute,
                                        time.second,
                                        int(time.microsecond*0.001))
    return(time_str)

    
def check_dir(directory, make=True, write_into=True):
    '''Checks if directory exists
    
    Parameters
    ----------
    directory : str or pathlib.PosixPath
        The directory of interest
    make : bool 
        Whether or not the directory should be created or just checked to
        ensure it exists. (default True)
    write_into : bool 
        If True, if a directory with the same name exists, new items will be
        saved into the old directory. Otherwise, an error will be raised. 
        (default True)
        
    Returns
    -------
    directory : pathlib.PosixPath
        If a directory could be created or confirmed to exist, the directory
        path will be returned. Otherwise Errors will be raised.
    '''
    import os 
    if not isinstance(directory, pathlib.PosixPath):
        directory = pathlib.Path(directory)
    # check to ensure the pathway does not have an extension
    if directory.suffix:
        raise TypeError('Expected pathway without extension. Did you mean to set \n~ '#\
            +str(directory)+' ~\nas a directory? If so, remove extension.')
    if not os.path.exists(directory):
        if make:
            os.mkdir(directory)
        else:
            raise FileNotFoundError('The following directory does not exist: '+\
                '\n{}'.format(directory))
    else:
        if not write_into:
            raise FileExistsError('The following directory already exists: '+\
                '\n{}'.format(directory)+'\nTo write into this directory, '+\
                    'set `write_into` to True.')
    return directory

def adjust_time_units(time_sec):
    if time_sec >= 60 and time_sec < 3600:
        total_time = time_sec / 60
        units = 'minutes'
    elif time_sec >= 3600:
        total_time = time_sec / 3600
        units = 'hours'
    else:
        total_time = time_sec
        units = 'seconds'
    return total_time, units

def print_progress(iteration, total_iterations, task = None):
    #print on screen the progress
    progress = (iteration+1) / total_iterations * 100 
    if task:
        sys.stdout.write("\r%d%% through {}".format(task) % progress)
    else:
        sys.stdout.write("\r%d%% through current task" % progress)
    sys.stdout.flush()
    

def check_extraction_variables(sr, feature_type,
              window_size_ms, window_shift):
    accepted_features = ['fbank', 'stft', 'mfcc', 'signal']

    if not isinstance(sr, int):
        raise ValueError('Sampling rate (sr) must be of type int, not '+\
            '{} of type {}.'.format(sr, type(sr)))
    if feature_type not in accepted_features:
        raise ValueError('feature_type {} is not supported.'.format(feature_type)+\
            ' Must be one of the following: {}'.format(', '.join(accepted_features)))

    if not isinstance(window_size_ms, int) and not isinstance(window_size_ms, float):
        raise ValueError('window_size_ms must be an integer or float, '+\
            'not {} of type {}.'.format(window_size_ms, type(window_size_ms)))
    if not isinstance(window_shift, int) and not isinstance(window_shift, float):
        raise ValueError('window_shift must be an integer or float, '+\
            'not {} of type {}.'.format(window_shift, type(window_shift)))
    
def collect_audiofiles(directory, hidden_files = False, wav_only=False, recursive=False):
    '''Collects all files within a given directory.
    
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
    return paths_list
    
def check_noisy_clean_match(noisyfilename, cleanfilename):
    clean = os.path.splitext(os.path.basename(cleanfilename))[0]
    noisy = os.path.splitext(os.path.basename(noisyfilename))[0]
    if clean in noisy:
        return True
    else:
        print('{} is not in {}.'.format(clean, noisy))
        return False
    
def check_length_match(cleanfilename, noisyfilename):   
    clean, sr1 = pyst.dsp.loadsound(cleanfilename)
    noisy, sr2 = pyst.dsp.loadsound(noisyfilename)
    assert sr1 == sr2
    if len(clean) != len(noisy):
        print('length of clean speech: ', len(clean))
        print('length of noisy speech: ', len(noisy))
    else:
        print('length matches!')
    return None

def make_number(value):
    '''If possibe, turns a string into an int, float, or None value.

    This is useful when loading values from a dictionary that are 
    supposed to be integers, floats, or None values instead of strings.

    Parameters
    ----------
    value : str
        The string that should become a number

    Returns
    ----------
    Return value : int, float, None or str
        If `value` is an integer of type str, the number is converted to type int.
        If `value` has the structure of a float, it is converted to type float.
        If `value` is an empty string, it will be converted to type None. 
        Otherwise, `value` is returned unaltered.

    Examples
    ----------
    >>> type_int = make_number('5')
    >>> type(type_int) 
    <class 'int'>
    >>> type_int 
    5
    >>> type_float = make_number('0.45')
    >>> type(type_float) 
    <class 'float'>
    >>> type_float 
    0.45
    >>> type_none = make_number('')
    >>> type(type_none) 
    <class 'NoneType'>
    >>> type_none 
    >>>
    >>> type_str = make_number('53d')
    Value cannot be converted to a number.
    >>> type(type_str) 
    <class 'str'>
    '''
    try:
        if isinstance(value, str) and value == '':
            value_num = None
        elif isinstance(value, str) and value.isdigit():
            value_num = int(value)
        elif isinstance(value, str):
            try:
                value_num = float(value)
            except ValueError:
                raise ValueError(
                    'Value cannot be converted to a number.')
        else:
            raise ValueError('Expected string, got {}\
                             \nReturning original value'.format(
                type(value)))
        return value_num
    except ValueError as e:
        print(e)
    return value


if __name__ == '__main__':
    import doctest
    doctest.testmod()
