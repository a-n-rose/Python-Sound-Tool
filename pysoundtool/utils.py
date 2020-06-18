'''Utils module covers functions that are useful for PySoundTool but are not 
directly related to sound data.
''' 
import os, sys
import csv
import numpy as np
import datetime
import pathlib
# for converting string lists back into list:
import ast

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
    '''Get a string containing month, day, hour, minute, second and millisecond.
    
    This is useful for creating a unique filename.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    time_str : str 
        A string containing the date and time.
        
    Examples
    --------
    >>> date = get_date()
    >>> date
    '6m18d1h16m32s295ms'
    '''
    time = datetime.datetime.now()
    time_str = "{}m{}d{}h{}m{}s{}ms".format(time.month,
                                        time.day,
                                        time.hour,
                                        time.minute,
                                        time.second,
                                        int(time.microsecond*0.001))
    return(time_str)

def check_dir(directory, make=True, write_into=True):
    '''Checks if directory exists and creates it if indicated.
    
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
            try:
                os.mkdir(directory)
            except FileNotFoundError:
                # parent directories might not exist
                directory = create_nested_dirs(directory)
        else:
            raise FileNotFoundError('The following directory does not exist: '+\
                '\n{}'.format(directory))
    else:
        if not write_into:
            raise FileExistsError('The following directory already exists: '+\
                '\n{}'.format(directory)+'\nTo write into this directory, '+\
                    'set `write_into` to True.')
    return directory

def create_nested_dirs(directory):
    '''Creates directory even if several parent directories don't exist.
    
    Parameters
    ----------
    directory : str, pathlib.PosixPath
        The directory to be created.
    
    Returns
    -------
    directory : pathlib.PosixPath
        If successful, the directory path that has been created.
        
    Examples
    --------
    >>> # First an unsucessful creation of nested directory
    >>> import os
    >>> new_dir = './testdir/testdir/testdir/'
    >>> os.mkdir(new_dir)
    FileNotFoundError: [Errno 2] No such file or directory: './testdir/testdir/testdir/'
    >>> # try again with create_nested_dirs()
    >>> directory = create_nested_dirs(new_dir)
    >>> directory
    PosixPath('testdir/testdir/testdir')
    '''
    if not isinstance(directory, pathlib.PosixPath):
        directory = pathlib.Path(directory)
    try:
        os.mkdir(directory)
    except FileNotFoundError:
        path_parent = create_nested_dirs(directory.parent)
        local_dirname = directory.name 
        directory = path_parent.joinpath(local_dirname)
        os.mkdir(directory)
    return directory

def string2pathlib(pathway_string):
    '''Turns string path into pathlib.PosixPath object.
    
    This is useful when working with pathways from varying operating 
    systems. Windows, Linux, and Mac have different ways of organizing
    pathways and pathlib turns strings from these different versions 
    into a pathlib object that can be understood by the software 
    regardless of the system. (At least I hope so..)
    
    Parameters
    ----------
    pathway_string : str or pathlib.PosixPath
        The pathway to be turned into a pathlib object, if need be.
        
    Returns
    -------
    pathway_string : pathlib.PosixPath
        The pathway as a pathlib object.
        
    Examples
    --------
    >>> pathway = 'folder/way2go.txt'
    >>> pathlib_pathway = string2pathlib(pathway)
    >>> pathlib_pathway
    PosixPath('folder/way2go.txt')
    '''
    if not isinstance(pathway_string, pathlib.PosixPath):
        try:
            pathway_string = pathlib.Path(pathway_string)
        except TypeError:
            raise TypeError('Function string2pathlib expects a string or '+\
                'pathlib object, not input of type {}'.format(type(pathway_string)))
    return pathway_string

def string2list(list_paths_string):
    '''Take a string of wavfiles list and establishes back to list

    Warning: this is a very specific function. It has not been tested to handle
    a variety of string lists. This handles lists of strings, lists of 
    pathlib.PosixPath objects, and lists of pathlib.PurePosixPath objects that 
    were converted into a type string object.

    Parameters
    ----------
    list_paths_string : str 
        The list that was converted into a string object 

    Returns
    -------
    list_paths : list 
        The list converted back to a list of paths as pathlib.PosixPath objects.

    Examples
    --------
    >>> input_string = "[PosixPath('data/audio/vacuum/vacuum1.wav')]"
    >>> type(input_string)
    <class 'str'>
    >>> typelist = string2list(input_string)
    >>> typelist
    [PosixPath('data/audio/vacuum/vacuum1.wav')]
    >>> type(typelist)
    <class 'list'>
    >>> # Get type of the object
    >>> type(typelist[0])
    pathlib.PosixPath
    >>> # Example with a list of tuples, i.e. label and audio file pairs:
    >>> input_string = "[(2, PosixPath('data/audio/vacuum/vacuum1.wav')), '+\
        '(1, PosixPath('data/audio/vacuum/vacuum2.wav'))]"
    >>> labelaudio_pairs = string2list(input_string)
    >>> labelaudio_pairs
    [(2, PosixPath('data/audio/vacuum/vacuum1.wav')),
    (1, PosixPath('data/audio/vacuum/vacuum2.wav'))]
    >>> type(labelaudio_pairs)
    list
    >>> type(labelaudio_pairs[0])
    tuple
    >>> type(labelaudio_pairs[0][0])
    int
    >>> type(labelaudio_pairs[0][1])
    pathlib.PosixPath
    '''
    try:
        # first use reliable module to turn string list into list
        list_paths = ast.literal_eval(list_paths_string)
    except ValueError:
        # ast doesn't handle lists of pathlib.PosixPath objects
        # TODO further testing
        # remove the string brackets '[' and ']'
        list_remove_brackets = list_paths_string[1:-1]
        if list_remove_brackets[0] == '(' and list_remove_brackets[-1] == ')':
            # list of tuples
            tuple_string = list_remove_brackets.split('), ')
            tuple_list = [tuple(x.split(', ') for x in tuple_string)]
            list_paths = []
            for item in tuple_list:
                item_list = []
                for t in item:
                    if t[0][0] == '(':
                        t[0] = t[0][1:]
                    if t[-1][-1] == ')':
                        t[-1] = t[-1][:-1]
                    item_list.append(t)
                list_paths.append(tuple(item_list))
            # turn into pathlib.PosixPath objects
            list_pathlib = []
            for item in list_paths:
                item_list = []
                for label, path in item:
                    if label.isdigit():
                        label = int(label)
                    if 'PurePosixPath' in path:
                        remove_str = "PurePosixPath('"
                        end_index = -1
                    elif 'PosixPath' in path:
                        remove_str = "PosixPath('"
                        end_index = -1
                    else:
                        remove_str = "('"
                        end_index = -1
                    audiopath = path.replace(remove_str, '')[:end_index]
                    # end of tuple list, extra "'" character. 
                    # get rid of it
                    if audiopath[-1] == "'":
                        audiopath = audiopath[:-1]
                    audiopath = pathlib.Path(audiopath)
                    item_list.append(tuple([label, audiopath]))
                list_pathlib.append(item_list)
            list_paths = list_pathlib[0]
            return list_paths
        
        list_string_red = list_paths_string[1:-1].split(', ')
        if 'PurePosixPath' in list_paths_string:
            remove_str = "PurePosixPath('"
            end_index = -2
        elif 'PosixPath' in list_paths_string:
            remove_str = "PosixPath('"
            end_index = -2
        else:
            remove_str = "('"
            end_index = -2
        # remove unwanted sections of the string items
        list_paths = []
        for path in list_string_red:
            list_paths.append(pathlib.Path(
                path.replace(remove_str, '')[:end_index]))
    return list_paths

def adjust_time_units(time_sec):
    '''Turns seconds into relevant time units.
    
    This is useful if measuring time of a process and that process 
    takes longer than a couple minutes.
    
    Parameters
    ----------
    time_sec : int, float 
        The amount of time measured in seconds.
    
    Returns
    -------
    total_time : int, float 
        The total amount of time 
    units : str 
        The unites of `total_time`: 'seconds', 'minutes', or 'hours'.
        
    Examples
    --------
    >>> adjust_time_units(5)
    (5, 'seconds')
    >>> adjust_time_units(500)
    (8.333333333333334, 'minutes')
    >>> adjust_time_units(5000)
    (1.3888888888888888, 'hours')
    '''
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
    '''Prints the status of a process based on iteration number.
    
    Assumes the iteration starts at 0 rather than 1.
    
    Parameters
    ----------
    iteration : int 
        The iteration of the current process.
    total_iterations : int 
        The total iterations to be completed.
    task : str, optional
        The relevant task of the process.
        
    Returns
    -------
    sys.stdout.flush()
    
    Examples
    --------
    >>> print_progress(4, 10)
    50% through current task
    >>> print_progress(4, 10, task = 'testing')
    50% through testing
    '''
    progress = (iteration+1) / total_iterations * 100 
    if task:
        sys.stdout.write("\r%d%% through {}".format(task) % progress)
    else:
        sys.stdout.write("\r%d%% through current task" % progress)
    sys.stdout.flush()
    
def check_extraction_variables(sr=None, feature_type=None,
              win_size_ms=None, percent_overlap=None):
    '''Checks to ensure extraction variables are compatible.
    
    Parameters
    ----------
    sr : int 
        The sample rate of audio.
    feature_type : str 
        The type of feature to be extracted: 'fbank', 'stft', 'powspec',
        'mfcc', 'signal'.
    win_size_ms : int, float
        The window size to process audio samples.
    percent_overlap : int, float 
        The percent windows should overlap.
        
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If any of the Parameters aren't compatible.
        
    Examples
    --------
    >>> check_extraction_variables(sr=48000, feature_type='signal', win_size_ms=25,percent_overlap=0.5)
    >>> check_extraction_variables(sr='48000', feature_type='sig',win_size_ms='25',percent_overlap='0.5')
    ValueError: Sampling rate (sr) must be of type int, not 48000 of type <class 'str'>.
    '''
    accepted_features = ['fbank', 'stft', 'powspec', 'mfcc', 'signal']

    if not isinstance(sr, int):
        raise ValueError('Sampling rate (sr) must be of type int, not '+\
            '{} of type {}.'.format(sr, type(sr)))
    if feature_type not in accepted_features:
        raise ValueError('feature_type {} is not supported.'.format(feature_type)+\
            ' Must be one of the following: {}'.format(', '.join(accepted_features)))

    if not isinstance(window_size_ms, int) and not isinstance(window_size_ms, float):
        raise ValueError('window_size_ms must be an integer or float, '+\
            'not {} of type {}.'.format(window_size_ms, type(window_size_ms)))
    if not isinstance(percent_overlap, int) and not isinstance(percent_overlap, float):
        raise ValueError('window_shift must be an integer or float, '+\
            'not {} of type {}.'.format(percent_overlap, type(percent_overlap)))

# TODO: add ensure_only_audio
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
    paths_list = pyst.data.ensure_only_audiofiles(paths_list)
    return paths_list
    
def check_noisy_clean_match(noisyfilename, cleanfilename):
    '''Checks if the clean filename is inside of the noisy filename.
    
    This may be helpful to check that two audiofile data sets (a noisy and 
    clean dataset) are aligned. 
    '''
    clean = os.path.splitext(os.path.basename(cleanfilename))[0]
    noisy = os.path.splitext(os.path.basename(noisyfilename))[0]
    if clean in noisy:
        return True
    else:
        print('{} is not in {}.'.format(clean, noisy))
        return False
    
def check_length_match(filename1, filename2):   
    '''Checks that two audiofiles have the same length.
    
    This may be useful if you have clean and noisy audiofiles that 
    should be the same length.
    
    Parameters
    ----------
    filename1 : str or pathlib.PosixPath
        The path to first audio file.
    filename2 : str or pathlib.PosixPath
        The path to second audio file.
    
    Returns
    -------
    bool : True if they match, False if not.
    
    Warning
    -------
    UserWarning 
        If the sample rate of the audio files don't match.
    UserWarning
        If the length of the files don't match.
    '''
    y1, sr1 = pyst.loadsound(filename1)
    y2, sr2 = pyst.loadsound(filename2)
    if sr1 != sr2:
        import Warnings
        message = '\nWARNING: Sample rates do not match: '+\
            '\n{} has sr {}'.format(filename1, sr1)+\
            '\n{} has sr {}.'.format(filename2, sr2)
        warnings.warn(message)
        y2, sr2 = pyst.dsp.resample_audio(y2, sr_original = sr2, sr_desired = sr1)
    assert sr1 == sr2
    if len(y1) != len(y2):
        import warnings
        message = '\nWARNING: audiofile length mismatch. Length '+\
            ' {}: \n{}'.format(filename1, len(y1))+\
                'Length {}: \n{}'.format(filename2, len(y2))
        return False
    else:
        return True

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

def save_dict(dict2save, filename, overwrite=False):
    '''Saves dictionary as csv file to indicated path and filename

    Parameters
    ----------
    dict2save : dict
        The dictionary that is to be saved 
    filename : str 
        The path and name to save the dictionary under. If '.csv' 
        extension is not given, it is added.
    overwrite : bool, optional
        Whether or not the saved dictionary should overwrite a 
        preexisting file (default False)

    Returns
    ----------
    path : pathlib.PosixPath
        The path where the dictionary was saved
    '''
    if not isinstance(filename, pathlib.PosixPath):
        filename = pathlib.Path(filename)
    if filename.parts[-1][-4:] != '.csv':
        filename_str = filename.resolve()
        filename_csv = filename_str+'.csv'
        filename = pathlib.Path(filename_csv)
    if not overwrite:
        if os.path.exists(filename):
            raise FileExistsError(
                'The file {} already exists at this path:\
                \n{}'.format(filename.parts[-1], filename))
    with open(filename, 'w') as f:
        w = csv.writer(f)
        w.writerows(dict2save.items())
    return filename

def load_dict(csv_path):
    '''Loads a dictionary from csv file. Expands csv limit if too large.
    
    Increasing the csv limit helps if loading dicitonaries with very long audio 
    file path lists. For example, see pysoundtool.data.audio2datasets function.
    '''
    try:
        with open(csv_path, mode='r') as infile:
            reader = csv.reader(infile)
            dict_prepped = {rows[0]: rows[1] for rows in reader}
    except csv.Error:
        print('Dictionary values or size is too large.')
        print('Maxing out field size limit for loading this dictionary:')
        print(csv_path)
        print('\nThe new field size limit is:')
        maxInt = sys.maxsize
        print(maxInt)
        csv.field_size_limit(maxInt)
        dict_prepped = load_dict(csv_path)
    except OverflowError as e:
        print(e)
        maxInt = int(maxInt/10)
        print('Reducing field size limit to: ', maxInt)
        dict_prepped = load_dict(csv_path)
    return dict_prepped

if __name__ == '__main__':
    import doctest
    doctest.testmod()
