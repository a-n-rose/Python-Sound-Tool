'''Data module contains functions related to handling audio data files, 
setting up datasets, and loading audio data. 
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


import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)

import pysoundtool as pyst

###############################################################################


def loadsound(filename, sr=None, mono=True, dur_sec = None, use_librosa=True):
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
    use_librosa : bool 
        If True, librosa will be used to load the audiofile. If False, 
        scipy.io.wavfile and/or soundfile will be used. If the sound file 
        is not compatible with scipy.io.wavfile.read, this functions converts 
        the file to .wav format and/or changes the bit depth to be compatible. 
        (default True)
        
    Returns
    -------
    data : nd.array [size=(num_samples,) or (num_samples, num_channels)]
        The normalized (between 1 and -1) sample data returned 
        according to the specified settings.
    sr : int 
        The sample rate of the loaded samples.
        
    See Also
    --------
    pysoundtool.data.prep4scipywavfile
        Prepares audio file for scipy.io.wavfile.read.
        
    pysoundtool.data.convert2wav
        Converts audio file to .wav format.
    
    pysoundtool.data.newbitdepth
        Converts audio file to specified bitdepth.
    '''
    if use_librosa:
        # the sample data will be a litle different from scipy.io.wavfile
        # as librosa does a litle extra work with the data
        data, sr = librosa.load(filename, sr=sr, mono=mono, duration=dur_sec)
        if mono is False and data.shape[0] < data.shape[1]:
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
            filename = pyst.data.convert2wav(filename)
        except RuntimeError as e:
            raise RuntimeError('Try setting `use_librosa` to True in pysoundtool.loadsound().')
        try:
            data, sr = loadsound(filename, sr=sr, mono=mono, dur_sec=dur_sec)
            print("File saved as {}".format(filename))
        except ValueError:
            print("Ensure bitdepth is compatible with scipy library")
            filename = pyst.data.newbitdepth(filename)
            data, sr = loadsound(filename, sr=sr, mono=mono, dur_sec=dur_sec)
    
    if mono and len(data.shape) > 1:
        if data.shape[1] > 1:
            data = pyst.dsp.stereo2mono(data)
    # scale samples to be between -1 and 1
    data = pyst.dsp.scalesound(data, -1, 1)
    if dur_sec:
        numsamps = int(dur_sec * sr)
        data = pyst.dsp.set_signal_length(data, numsamps)
    return data, sr

def savesound(audiofile_name, signal_values, sr, overwrite=False):
    """saves the wave at designated path

    Parameters
    ----------
    audiofile_name : str or pathlib.PosixPath
        path and name the audio is to be saved under. (.wav format)
    signal_values : ndarray
        values of real signal to be saved
    sr : int 
        sample rate of the audio samples.

    Returns
    ----------
    True if successful, otherwise False
    """
    audiofile_name = pyst.utils.string2pathlib(audiofile_name)
    if os.path.exists(audiofile_name) and overwrite is False:
        raise FileExistsError('Filename {} already exists.'.format(audiofile_name)+\
            '\nSet `overwrite` to True in function savesound() to overwrite.')
    directory = audiofile_name.parent
    directory = pyst.utils.check_dir(directory, make=True)
    write(audiofile_name, sr, signal_values)
    return audiofile_name

def list_possibleformats(use_librosa=True):
    if use_librosa:
        return(['.wav', '.aiff', '.flac', '.ogg','.m4a','.mp3'])
    else:
        return(['.wav', '.aiff', '.flac', '.ogg'])

def list_audioformats():
    msg = '\nPySoundTool can work with the following file types: '+\
        ', '.join(get_compatible_formats())+ \
            '\nSo far, functionality does not work with the following types: '+\
                ', '.join(get_incompatible_formats())
    return msg

def create_encodedlabel2audio_dict(dict_encodelabels, paths_list, limit=None, seed=40):
    '''Creates dictionary with audio labels as keys and filename lists as values.

    If no label is found in the filename path, the label is not included
    in the returned dictionary: labels are only included if corresponding
    paths are present.

    Parameters
    ----------
    dict_encodelabels : dict 
        Dictionary containing the labels as keys and their encoded values as values.
    paths_list : set, list 
        List containing pathlib.PosixPath objects (i.e. paths) of all audio 
        files; expected the audio files reside in directories with names 
        matching their audio class
    limit : int, optional
        The integer indicating a limit to number of audiofiles to each class. This
        may be useful if one wants to ensure a balanced dataset (default None)
    seed : int, optional
        The seed for pseudorandomizing the wavfiles, if a limit is requested. 
        If `seed` is set to None, the randomized order of the limited wavfiles cannot
        be repeated. (default 40)

    Returns
    -------
    label_waves_dict : OrderedDict
        A dictionary with encoded audio labels as keys with values being the audio files 
        corresponding to that label

    TODO update:
    Examples
    --------
    >>> from pathlib import Path
    >>> labels = dict([('vacuum',2),('fridge',0),('wind',1)])
    >>> paths = [Path('data/audio/vacuum/vacuum1.wav'), 
    ...         Path('data/audio/fridge/fridge1.wav'), 
    ...         Path('data/audio/vacuum/vacuum2.wav'),
    ...         Path('data/audio/wind/wind1.wav')]
    >>> label_waves_dict = create_encodedlabel2audio_dict(labels, paths)
    >>> label_waves_dict
    OrderedDict([(0, [PosixPath('data/audio/fridge/fridge1.wav')]), \
    (2, [PosixPath('data/audio/vacuum/vacuum1.wav'), \
    PosixPath('data/audio/vacuum/vacuum2.wav')]), \
    (1, [PosixPath('data/audio/wind/wind1.wav')])])
    >>> #to set a limit on number of audiofiles per class:
    >>> create_encodedlabel2audio_dict(labels, paths, limit=1, seed=40)
    OrderedDict([(0, [PosixPath('data/audio/fridge/fridge1.wav')]), \
    (2, [PosixPath('data/audio/vacuum/vacuum2.wav')]), \
    (1, [PosixPath('data/audio/wind/wind1.wav')])])
    >>> #change the limited pathways chosen:
    >>> create_encodedlabel2audio_dict(labels, paths, limit=1, seed=10)
    OrderedDict([(0, [PosixPath('data/audio/fridge/fridge1.wav')]), \
    (2, [PosixPath('data/audio/vacuum/vacuum1.wav')]), \
    (1, [PosixPath('data/audio/wind/wind1.wav')])])
    '''
    if not isinstance(dict_encodelabels, dict):
        raise TypeError(
            'Expected dict_encodelabels to be type dict, not type {}'.format(type(
                dict_encodelabels)))
    if not isinstance(paths_list, set) and not isinstance(paths_list, list):
        raise TypeError(
            'Expected paths list as type set or list, not type {}'.format(type(
                paths_list)))
    label_waves_dict = collections.OrderedDict()
    # get labels from dict_encodelabels:
    labels_set = set(list(dict_encodelabels.keys()))
    for label in sorted(labels_set):
        # expects name of parent directory to match label
        label_paths = [path for path in paths_list if
                       label == path.parent.name]
        if label_paths:
            if isinstance(limit, int):
                if seed:
                    np.random.seed(seed=seed)
                rand_idx = np.random.choice(range(len(label_paths)),
                                            len(label_paths),
                                            replace=False)
                paths_idx = rand_idx[:limit]
                label_paths = list(np.array(label_paths)[paths_idx])
                # encode label in the label_waves_dict
            label_waves_dict[dict_encodelabels[label]] = sorted(label_paths)
    if not label_waves_dict:
        raise ValueError('No matching labels found in paths list.')
    return label_waves_dict

def create_dicts_labelsencoded(labels_class):
    '''Encodes audio class labels and saves in dictionaries.

    The labels are alphabetized and encoded under their index.

    Parameters
    ----------
    labels_class : set, list
        Set or list containing the labels of all audio classes.

    Returns
    -------
    dict_label2int : dict
        Dictionary where the keys are the string labels and the 
        values are the encoded integers
    dict_int2label : dict
        Dictionary where the keys are the encoded integers and the
        values are the string labels

    Examples
    --------
    >>> labels = {'wind','air_conditioner','fridge'}
    >>> label2int, int2label = create_dicts_labelsencoded(labels)
    >>> label2int
    {'air_conditioner': 0, 'fridge': 1, 'wind': 2}
    >>> int2label
    {0: 'air_conditioner', 1: 'fridge', 2: 'wind'}
    '''
    if not isinstance(labels_class, set) and not isinstance(labels_class, list):
        raise TypeError(
            'Expected inputs as type set or list, not type {}'.format(type(
                labels_class)))
    labels_sorted = sorted(set(labels_class))
    dict_label2int = {}
    dict_int2label = {}
    for i, label in enumerate(labels_sorted):
        dict_label2int[label] = i
        dict_int2label[i] = label
    return dict_label2int, dict_int2label

def waves2dataset(audiolist, train_perc=0.8, seed=40):
    '''Organizes audio files list into train, validation and test datasets.

    Parameters
    ----------
    audiolist : list 
        List containing paths to audio files
    train_perc : float, int 
        Percentage of data to be in the training dataset (default 0.8)
    seed : int, None, optional
        Set seed for the generation of pseudorandom train, validation, 
        and test datsets. Useful for reproducing results. (default 40)

    Returns
    -------
    train_waves : list
        List of audio files for the training dataset
    val_waves : list
        List of audio files for the validation dataset
    test_waves : list
        List of audio files for the test dataset

    Examples
    --------
    >>> #Using a list of numbers instead of filenames
    >>> audiolist = [1,2,3,4,5,6,7,8,9,10]
    >>> #default settings:
    >>> waves2dataset(audiolist)
    ([5, 4, 9, 2, 3, 10, 1, 6], [8], [7])
    >>> #train_perc set to 50% instead of 80%:
    >>> waves2dataset(audiolist, train_perc=50)
    ([5, 4, 9, 2, 3, 10], [1, 6], [8, 7])
    >>> #change seed number
    >>> waves2dataset(audiolist, seed=0)
    ([7, 1, 2, 5, 6, 9, 10, 8], [4], [3])
    '''
    if train_perc > 1:
        train_perc *= 0.01
    val_train_perc = (1-train_perc)/2.
    num_waves = len(audiolist)
    num_train = int(num_waves * train_perc)
    num_val_test = int(num_waves * val_train_perc)
    if num_val_test < 1:
        num_val_test += 1
        num_train -= 2
    if num_train + 2*num_val_test < num_waves:
        diff = num_waves - num_train - 2*num_val_test
        num_train += diff
    if seed:
        np.random.seed(seed=seed)
    rand_idx = np.random.choice(range(num_waves),
                                num_waves,
                                replace=False)
    train_idx = rand_idx[:num_train]
    val_test_idx = rand_idx[num_train:]
    val_idx = val_test_idx[:num_val_test]
    test_idx = val_test_idx[num_val_test:]
    train_waves = list(np.array(audiolist)[train_idx])
    val_waves = list(np.array(audiolist)[val_idx])
    test_waves = list(np.array(audiolist)[test_idx])
    assert len(train_waves)+len(val_waves)+len(test_waves) == len(audiolist)
    return train_waves, val_waves, test_waves

def ensure_only_audiofiles(audiolist):
    possible_extensions = pyst.data.list_possibleformats(use_librosa=True)
    audiolist_checked = [x for x in audiolist if pathlib.Path(x).suffix in possible_extensions]
    if len(audiolist_checked) < len(audiolist):
        import warnings
        message = 'Some files did not match those acceptible by this program. '+\
            'The number of files removed: {}'.format(len(audiolist)-len(audiolist_checked))
        warnings.warn(message)
    return audiolist_checked

def audio2datasets(audiodata, perc_train=0.8, limit=None, seed=None):
    '''Organizes all audio in audio class directories into datasets (randomized).

    Parameters
    ----------
    audiodata : str, pathlib.PosixPath, dict, list, or set
        If data has multiple labels, path to the dictionary where audio class 
        labels and the paths of all audio files belonging to each class are or will
        be stored. The dictionary with the labels and their encoded values
        can also directly supplied here. If the data does not have labels, a list or 
        set of audiofiles can be provided to be placed in train, val, and test datasets.
    perc_train : int, float
        The percentage or decimal representing the amount of training
        data compared to the test and validation data (default 0.8)
    seed : int, optional
        A value to allow random order of audiofiles to be predictable. 
        (default None). If None, the order of audiofiles will not be predictable.

    Returns
    -------
    dataset_audio : tuple
        Named tuple including three lists / datasets of audiofiles or 
        label-audiofile pairs: the train, validation, and test lists, respectively. 
        The label-audiofile pairs are saved as tuples within the lists and contain 
        the encoded label integer (e.g. 0 instead of 'air_conditioner') and the 
        audio paths associated to that class and dataset.
    '''
    if seed == 0:
        raise ValueError('Seed equals 0. This will result in unreliable '+\
            'randomization. Either set `seed` to None or to another integer.')
    if perc_train > 1:
        perc_train /= 100.
    if perc_train > 1:
        raise ValueError('The percentage value of train data exceeds 100%')
    perc_valtest = (1-perc_train)/2.
    if perc_valtest*2 > perc_train:
        raise ValueError(
            'The percentage of train data is too small: {}\
            \nPlease check your values.'.format(
                perc_train))
    if isinstance(audiodata, dict) or isinstance(audiodata, list) or \
        isinstance(audiodata, set):
        waves = audiodata
    else:
        waves = pyst.utils.load_dict(audiodata)
    if isinstance(waves, list) or isinstance(waves,set) or len(waves) == 1:
        multiple_labels = False
    else:
        multiple_labels = True
    count = 0
    row = 0
    train = []
    val = []
    test = []
    if multiple_labels:
        for key, value in waves.items():
            if isinstance(value, str):
                audiolist = pyst.utils.string2list(value)
                # check to make sure all audiofiles and none were lost
                audiolist = pyst.data.ensure_only_audiofiles(audiolist)
                key = int(key)
            else:
                audiolist = value
            train_waves, val_waves, test_waves = waves2dataset(sorted(audiolist), seed=seed)
            for i, wave in enumerate(train_waves):
                train.append(tuple([key, wave]))
            for i, wave in enumerate(val_waves):
                val.append(tuple([key, wave]))
            for i, wave in enumerate(test_waves):
                test.append(tuple([key, wave]))
    else:
        # data has all same label, can be in a simple list, not paired with a label
        if isinstance(waves, dict):
            for i, key in enumerate(waves):
                if i >= 1:
                    raise ValueError('Expected only 1 key, not {}.'.format(len(waves)))
                audiolist = waves[key]
                if isinstance(audiolist, str):
                    # check to make sure all audiofiles and none were lost
                    audiolist = pyst.utils.string2list(audiolist)
                    audiolist = pyst.data.ensure_only_audiofiles(audiolist)
        else:
            audiolist = waves
        # sort to ensure a consistent order of audio; otherwise cannot control randomization
        train_waves, val_waves, test_waves = waves2dataset(sorted(audiolist), seed=seed)
        for i, wave in enumerate(train_waves):
            train.append(wave)
        for i, wave in enumerate(val_waves):
            val.append(wave)
        for i, wave in enumerate(test_waves):
            test.append(wave)
        
    # be sure the classes are not in any certain order
    if seed is not None: 
        random.seed(seed)
    random.shuffle(train)
    if seed is not None: 
        random.seed(seed)
    random.shuffle(val)
    if seed is not None: 
        random.seed(seed)
    random.shuffle(test)
    # esure the number of training data is 80% of all available audiodata:
    if len(train) < math.ceil((len(train)+len(val)+len(test))*perc_train):
        raise pyst.errors.notsufficientdata_error(len(train),
                                      len(val),
                                      len(test),
                                      math.ceil(
                                          (len(train)+len(val)+len(test))*perc_train))
    TrainingData = collections.namedtuple('TrainingData',
                                          ['train_data', 'val_data', 'test_data'])
    dataset_audio = TrainingData(
        train_data=train, val_data=val, test_data=test)
    return dataset_audio


def separate_train_val_test_files(list_of_files):
    '''Checks that file(s) exist, then sorts file(s) into train, val, test lists.
    
    If 'nois' or 'clean' are in the filenames, two paths lists per dataset 
    will be generated. Otherwise just one. This paths list is useful if there are multiple
    training files available for training a model (e.g. for large datasets).
    
    Parameters
    ----------
    list_of_files : list, str, or pathlib.PosixPath
        The feature files (format: .npy) for training a model. 
        
    Returns
    -------
    (train_paths_list, val_paths_list, test_paths_list) : tuple
        Tuple comprised of paths lists to train, validation, and test data files.
        If noisy and clean data files found, each tuple item will be a tuple comprised of two lists: a noisy file paths list and a clean file paths list.
        
    Examples
    --------
    >>> features_files = ['train1.npy', 'train2.npy', 'val.npy', 'test.npy']
    >>> datasets = separate_train_val_test_files(features_files)
    >>> datasets.train
    [PosixPath('train1.npy'), PosixPath('train2.npy')]
    >>> datasets.val
    [PosixPath('val.npy')]
    >>> datasets.test
    [PosixPath('test.npy')]
    >>> # try with noisy and clean data
    >>> features_files = ['train_noisy.npy', 'train_clean.npy', 'val_noisy.npy',
        'val_clean.npy', 'test_noisy.npy', 'test_clean.npy']
    >>> datasets = separate_train_val_test_files(features_files)
    >>> datasets.train.noisy
    [PosixPath('train_noisy.npy')]
    >>> datasets.train.clean
    [PosixPath('train_clean.npy')]
    >>> datasets.val.noisy
    [PosixPath('val_noisy.npy')]
    >>> datasets.val.clean
    [PosixPath('val_clean.npy')]
    >>> datasets.test.noisy
    [PosixPath('test_noisy.npy')]
    >>> datasets.test.clean
    [PosixPath('test_clean.npy')]
    '''
    train_data_input = []
    train_data_output = []
    val_data_input = []
    val_data_output = []
    test_data_input = []
    test_data_output = []
    if isinstance(list_of_files, str) or isinstance(list_of_files, pathlib.PosixPath):
        list_of_files = list(list_of_files)
    for f in list_of_files:
        if isinstance(f, str):
            f = pathlib.Path(f)
        # make sure data files exists:
        if not os.path.exists(f):
            raise FileNotFoundError('Feature file {} not found.'.format(f))
        
        if 'train' in f.stem:
            if 'nois' in f.stem:
                train_data_input.append(f)
            elif 'clean' in f.stem:
                train_data_output.append(f)
            else:
                # non noisy vs clean data
                train_data_input.append(f)
        elif 'val' in f.stem:
            if 'nois' in f.stem:
                val_data_input.append(f)
            elif 'clean' in f.stem:
                val_data_output.append(f)
            else:
                # non noisy vs clean data
                val_data_input.append(f)
        elif 'test' in f.stem:
            if 'nois' in f.stem:
                test_data_input.append(f)
            elif 'clean' in f.stem:
                test_data_output.append(f)
            else:
                # non noisy vs clean data
                test_data_input.append(f)
    TrainingData = collections.namedtuple('TrainingData',
                                          ['train', 'val', 'test'])
    NoisyCleanData = collections.namedtuple('NoisyCleanData',
                                            ['noisy', 'clean'])
    
    if train_data_output:
        train_paths_list = NoisyCleanData(noisy = train_data_input, 
                                        clean = train_data_output)
    else:
        train_paths_list = train_data_input
    if val_data_output:
        val_paths_list = NoisyCleanData(noisy = val_data_input, 
                                        clean = val_data_output)
    else:
        val_paths_list = val_data_input
    if test_data_output:
        test_paths_list = NoisyCleanData(noisy = test_data_input, 
                                        clean = test_data_output)
    else:
        test_paths_list = test_data_input
    return TrainingData(train = train_paths_list, 
                        val = val_paths_list, 
                        test = test_paths_list)

# TODO speed this up, e.g. preload noise data?
# TODO randomize sections of noise applied
def create_denoise_data(cleandata_dir, noisedata_dir, trainingdata_dir, limit=None,
                            snr_levels=None, delay_mainsound_sec = None, seed = None, **kwargs):
    '''Applies noise to clean audio; saves clean and noisy audio to `traingingdata_dir`.

    Parameters
    ----------
    cleandata_dir : str, pathlib.PosixPath
        Name of folder containing clean audio data for autoencoder. E.g. 'clean_speech'
    noisedata_dir : str, pathlib.PosixPath
        Name of folder containing noise to add to clean data. E.g. 'noise'
    trainingdata_dir : str, pathlib.PosixPath
        Directory to save newly created train, validation, and test data
    limit : int, optional
        Limit in number of audiofiles used for training data
    snr_levels : list of ints, optional
        List of varying signal-to-noise ratios to apply to noise levels.
        (default None)
    delay_mainsound_sec : int, float, optional
        Amount in seconds the main sound should be delayed. In other words, in seconds how
        long the background sound should play before the clean / main / target audio starts.
        (default None)
    seed : int 
        A value to allow random order of audiofiles to be predictable. 
        (default None). If None, the order of audiofiles will not be predictable.
    **kwargs : additional keyword arguments
        The keyword arguments for pysoundtool.data.loadsound
        

    Returns
    -------
    saveinput_path : pathlib.PosixPath
        Path to where noisy audio files are located
    saveoutput_path : pathlib.PosixPath   
        Path to where clean audio files are located
        
    See Also
    --------
    pysoundtool.data.loadsound
        Loads audiofiles.
    
    pysoundtool.dsp.add_backgroundsound
        Add background sound / noise to signal at a determined signal-to-noise ratio.
    '''
    import math
    import time
    
    start = time.time()

    # if paths are strings, convert to pathlib ojbects
    cleandata_dir = pyst.utils.string2pathlib(cleandata_dir)
    noisedata_dir = pyst.utils.string2pathlib(noisedata_dir)
    trainingdata_dir = pyst.utils.string2pathlib(trainingdata_dir)
    
    cleandata_folder = 'clean'
    noisedata_folder = 'noisy'
    if limit is not None:
        cleandata_folder += '_limit'+str(limit)
        noisedata_folder += '_limit'+str(limit)
    
    newdata_clean_dir = trainingdata_dir.joinpath(cleandata_folder)
    newdata_noisy_dir = trainingdata_dir.joinpath(noisedata_folder)
    
    # create directory to save new data (if not exist)
    newdata_clean_dir = pyst.utils.check_dir(newdata_clean_dir, make = True)
    newdata_noisy_dir = pyst.utils.check_dir(newdata_noisy_dir, make = True)
   
    # collect audiofiles (not limited to .wav files)
    cleanaudio = sorted(pyst.utils.collect_audiofiles(cleandata_dir,
                                                      hidden_files = False,
                                                      wav_only = False,
                                                      recursive = False))
    noiseaudio = sorted(pyst.utils.collect_audiofiles(noisedata_dir,
                                                      hidden_files = False,
                                                      wav_only = False,
                                                      recursive = False))
    
    if seed is not None:
        random.seed(seed)
    random.shuffle(cleanaudio)
    
    if limit is not None:
        cleanaudio = cleanaudio[:limit]
    
    # ensure snr_levels is array-like 
    if snr_levels is not None:
        if not isinstance(snr_levels, list) and not isinstance(snr_levels, np.ndarray):
            snr_levels = list(snr_levels)
    
    for i, wavefile in enumerate(cleanaudio):
        pyst.utils.print_progress(iteration=i, 
                    total_iterations=len(cleanaudio),
                    task='clean and noisy audio data generation')
        # no random seed applied here:
        # each choice would be the same for each iteration
        noise = random.choice(noiseaudio)
        if snr_levels is not None:
            snr = random.choice(snr_levels)
        else:
            snr = None
        clean_stem = wavefile.stem
        noise_stem = noise.stem
        # load clean data to get duration
        clean_data, sr = pyst.loadsound(wavefile, **kwargs)
        clean_seconds = len(clean_data)/sr
        noisy_data, sr, snr_appx = pyst.dsp.add_backgroundsound(audio_main = wavefile, 
                                                      audio_background = noise, 
                                                      snr = snr, 
                                                      delay_mainsound_sec=delay_mainsound_sec, 
                                                      total_len_sec = clean_seconds,
                                                      **kwargs)
        # ensure both noisy and clean files have same beginning to filename (i.e. clean filename)
        noisydata_filename = newdata_noisy_dir.joinpath(clean_stem+'_'+noise_stem\
            +'_snr'+str(snr)+'.wav')
        cleandata_filename = newdata_clean_dir.joinpath(clean_stem+'.wav')     
        write(noisydata_filename, sr, noisy_data)
        write(cleandata_filename, sr, clean_data)

    end = time.time()
    total_time, units = pyst.utils.adjust_time_units(end-start)
    print('Data creation took a total of {} {}.'.format(
        round(total_time,2), 
        units))

    return newdata_noisy_dir, newdata_clean_dir

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

def convert2wav(filename, sr=None, new_dir=False):
    '''Converts and saves soundfile as .wav type in same or new directory.
    
    Parameters
    ----------
    filename : str or pathlib.PosixPath
        The filename of the audiofile to be converted to .wav type
    new_dir : str, pathlib.PosixPath, optional 
        If False, the converted files will be saved in same directory as originals.
        If a path is provided, the converted files will be saved there. If no such directory
        exists, one will be created.
    sr : int, optional
        The sample rate to be applied to the signal. If none supplied, the sample rate 
        of the original file will be used.
        
    Returns 
    -------
    f_wavfile : str or pathlib.PosixPath
        The filename / path where the .wav file is saved.
    '''
    import pathlib
    import os
    try:
        f = pathlib.Path(filename)
    except TypeError:
        raise TypeError('Function convert2wav expected input of type string '+\
            'or a pathlib object, not type {}.'.format(type(filename)))
    if not f.suffix:
        raise TypeError('Function convert2wav expected a path with an '+\
            'audio extension, not input: \n', filename)
    if not f.suffix in pyst.data.list_possibleformats(use_librosa=True):
        raise TypeError('This software cannot process audio in {}'.format(f.suffix)+\
            ' format. We apologize for the inconvenience.')
    # ensure filename exists:
    if not os.path.exists(filename):
        raise FileNotFoundError('Could not find audio file at the following '+\
            'location\n{}'.format(filename))
    # establish the path to save updated file.
    if new_dir:
        # check if new_dir is a directory or filename
        new_dir = pathlib.Path(new_dir)
        if new_dir.suffix:
            new_filename = new_dir.stem
            new_extension = new_dir.suffix
            if new_extension != '.wav':
                import warnings
                warnings.warn('\n\nWARNING: Function convert2wav only converts to '+\
                    '.wav files, not to {} files. '.format(new_extension)+\
                        'Converting to .wav instead.\n\n')
                new_extension = '.wav'
            new_dir = new_dir.parent
        else:
            new_filename = f.stem
            new_extension = '.wav'
        # check to make sure new_dir exists
        new_dir = pyst.utils.check_dir(new_dir, make=True)
        
    else:
        new_filename = f.stem
        new_extension = '.wav'
        new_dir = f.parent
    new_filename = new_dir.joinpath(new_filename+new_extension)
        
    # load audio samples with soundfile, then save them as wav file.
    try:
        data, sr = sf.read(filename, samplerate=sr)
    except RuntimeError as e:
        data, sr = librosa.load(filename, sr=sr)
    # save the data as .wav file
    sf.write(new_filename, data, sr)
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

def soundfile_limitduration(newfilename, soundfile, sr=None, 
                            dur_sec=None, overwrite=False):
    if sr:
        data, sr = librosa.load(soundfile,sr=sr, duration=dur_sec)
    else:
        data, sr = librosa.load(soundfile, duration=dur_sec)
    sf.write(newfilename, data, sr)
    
    
# better here or in dsp module?
def zeropad_features(feats, desired_shape, complex_vals = False):
    '''Applies zeropadding to a copy of feats. 
    '''
    fts = feats.copy()
    if feats.shape != desired_shape:
        if complex_vals:
            dtype = np.complex_
        else:
            dtype = np.float
        empty_matrix = np.zeros(desired_shape, dtype = dtype)
        try:
            if len(desired_shape) == 1:
                empty_matrix[:feats.shape[0]] += feats
            elif len(desired_shape) == 2:
                empty_matrix[:feats.shape[0], 
                            :feats.shape[1]] += feats
            elif len(desired_shape) == 3:
                empty_matrix[:feats.shape[0], 
                            :feats.shape[1],
                            :feats.shape[2]] += feats
            elif len(desired_shape) == 4:
                empty_matrix[:feats.shape[0], 
                            :feats.shape[1],
                            :feats.shape[2],
                            :feats.shape[3]] += feats
            elif len(desired_shape) == 5:
                empty_matrix[:feats.shape[0], 
                            :feats.shape[1],
                            :feats.shape[2],
                            :feats.shape[3],
                            :feats.shape[4]] += feats
            else:
                raise TypeError('Zeropadding columns requires a matrix with '+\
                    'a minimum of 1 dimension and maximum of 5 dimensions.')
            fts = empty_matrix
        except ValueError as e:
            print(e)
            raise ValueError('The desired shape is smaller than the original shape.'+ \
                ' No zeropadding necessary.')
        except IndexError as e:
            print(e)
            raise IndexError('The dimensions do not align. Zeropadding '+ \
                'expects same number of dimensions.')
    assert fts.shape == desired_shape
    return fts

def reduce_num_features(feats, desired_shape, complex_vals = False):
    '''Limits number features of a copy of feats. 
    '''
    fts = feats.copy()
    if feats.shape != desired_shape:
        if complex_vals:
            dtype = np.complex_
        else:
            dtype = np.float
        empty_matrix = np.zeros(desired_shape, dtype = dtype)
        try:
            if len(desired_shape) == 1:
                empty_matrix += feats[:empty_matrix.shape[0]]
            elif len(desired_shape) == 2:
                empty_matrix += feats[:empty_matrix.shape[0], 
                            :empty_matrix.shape[1]]
            elif len(desired_shape) == 3:
                empty_matrix += feats[:empty_matrix.shape[0], 
                            :empty_matrix.shape[1],
                            :empty_matrix.shape[2]]
            elif len(desired_shape) == 4:
                empty_matrix += feats[:empty_matrix.shape[0], 
                            :empty_matrix.shape[1],
                            :empty_matrix.shape[2],
                            :empty_matrix.shape[3]]
            elif len(desired_shape) == 5:
                empty_matrix += feats[:empty_matrix.shape[0], 
                            :empty_matrix.shape[1],
                            :empty_matrix.shape[2],
                            :empty_matrix.shape[3],
                            :empty_matrix.shape[4]]
            else:
                raise TypeError('Reducing items in columns requires a matrix with'+\
                    ' a minimum of 1 dimension and maximum of 5 dimensions.')
            fts = empty_matrix
        except ValueError as e:
            print(e)
            raise ValueError('The desired shape is larger than the original shape.'+ \
                ' Perhaps try zeropadding.')
        except IndexError as e:
            print(e)
            raise IndexError('The dimensions do not align. Zeropadding '+ \
                'expects same number of dimensions.')
    assert fts.shape == desired_shape
    return fts

# TODO remove warning for 'operands could not be broadcast together with shapes..'
def adjust_data_shape(data, desired_shape):
    if len(data.shape) != len(desired_shape):
        raise ValueError('Cannot adjust data to a different number of '+\
            'dimensions.\nOriginal data shape: '+str(data.shape)+ \
                '\nDesired shape: '+str(desired_shape))
    # attempt to zeropad data:
    try:
        data_prepped = pyst.data.zeropad_features(data, 
                                                  desired_shape = desired_shape)
    # if zeropadding is smaller than data.shape/features:
    except ValueError:
        # remove extra data/columns to match desired_shape:
        data_prepped = pyst.data.reduce_num_features(data, 
                                                 desired_shape = desired_shape)
    return data_prepped

def section_data(dataset_dict, dataset_paths_dict, divide_factor=None):
    '''Expects keys of these two dictionaries to match
    
    Examples
    --------
    >>> import pathlib
    >>> # train is longer than val and test
    >>> d = {'train': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'val': [1, 2, 3, 4, 5],
            'test': [1, 2, 3, 4, 5]}
    >>> # dictionary: paths to where extracted data will be saved
    >>> dp = {'train': pathlib.PosixPath('train_data.npy'),
              'val': pathlib.PosixPath('val_data.npy'),
              'test': pathlib.PosixPath('test_data.npy')}
    >>> d2, dp2 = section_data(d, dp, divide_factor = 3)
    >>> # val and train not touched (too small)
    >>> d2
    {'train__1': [1, 2, 3, 4, 5],
    'train__2': [6, 7, 8, 9, 10],
    'train__3': [11, 12, 13, 14, 15],
    'val': [1, 2, 3, 4, 5],
    'test': [1, 2, 3, 4, 5]}
    >>> dp2
    {'train__1': PosixPath('train_data__1.npy'),
    'train__2': PosixPath('train_data__2.npy'),
    'train__3': PosixPath('train_data__3.npy'),
    'val': PosixPath('val_data.npy'),
    'test': PosixPath('test_data.npy')}
    >>> # repeat: now val and test as long as train
    >>> # default divide_factor is 2
    >>> d3, dp3 = section_data(d2, dp2)
    >>> d3
    {'train__1': PosixPath('train_data__1.npy'),
    'train__2': PosixPath('train_data__2.npy'),
    'train__3': PosixPath('train_data__3.npy'),
    'train__4': PosixPath('train_data__4.npy'),
    'train__5': PosixPath('train_data__5.npy'),
    'train__6': PosixPath('train_data__6.npy'),
    'val__1': PosixPath('val_data__1.npy'),
    'val__2': PosixPath('val_data__2.npy'),
    'test__1': PosixPath('test_data__1.npy'),
    'test__2': PosixPath('test_data__2.npy')}
    >>> dp3
    {'train__1': PosixPath('train_data__1.npy'),
    'train__2': PosixPath('train_data__2.npy'),
    'train__3': PosixPath('train_data__3.npy'),
    'train__4': PosixPath('train_data__4.npy'),
    'train__5': PosixPath('train_data__5.npy'),
    'train__6': PosixPath('train_data__6.npy'),
    'val__1': PosixPath('val_data__1.npy'),
    'val__2': PosixPath('val_data__2.npy'),
    'test__1': PosixPath('test_data__1.npy'),
    'test__2': PosixPath('test_data__2.npy')}
    '''
    if divide_factor is None:
        divide_factor = 2
    # find max length:
    maxlen = 0
    for key, value in dataset_dict.items():
        if len(value) > maxlen:
            maxlen = len(value)
    # the length the maximum list will have
    # if other value lists are shorter, don't need to be sectioned.
    new_max_len = int(maxlen/divide_factor)
    try:
        new_key_list = []
        updated_dataset_dict = dict()
        updated_dataset_paths_dict = dict()
        for key, value in dataset_dict.items():
            if len(value) <= new_max_len: 
                updated_dataset_dict[key] = dataset_dict[key]
                updated_dataset_paths_dict[key] = dataset_paths_dict[key]
            else:
                # separate value into sections
                divided_values = {}
                len_new_values = int(len(value)/divide_factor)
                if len_new_values < 1:
                    len_new_values = 1
                index = 0
                for i in range(divide_factor):
                    if i == divide_factor - 1:
                        # to ensure all values are included
                        vals = value[index:]
                    else:
                        vals = value[index:index+len_new_values]
                    divided_values[i] = vals
                    index += len_new_values
                
                # assign new keys for each divided section of data
                divided_values_keys = {}
                # assign new paths to each section of data:
                divided_values_paths = {}
                path_orig = dataset_paths_dict[key]
                for i in range(divide_factor):
                    if not key[-1].isdigit():
                        key_stem = key
                        version_num = 1
                        new_key = key_stem + '__' + str(version_num)
                    else:
                        key_stem, version_num = key.split('__')
                        version_num = int(version_num)
                        new_key = key
                    unique_key = False
                    while unique_key is False:
                        if new_key not in new_key_list:
                            unique_key = True
                            new_key_list.append(new_key)
                            divided_values_keys[i] = new_key
                            break
                        else:
                            version_num += 1
                            new_key = key_stem + '__'+ str(version_num)
                    
                    if not isinstance(path_orig, pathlib.PosixPath):
                        path_orig = pathlib.Path(path_orig)
                        # convert to pathlib.PosixPath
                        dataset_paths_dict[key] = path_orig
                    stem_orig = path_orig.stem
                    if stem_orig[-1].isdigit():
                        stem, vers = stem_orig.split('__')
                    else:
                        stem = stem_orig
                    new_stem = stem + '__' + str(version_num)
                    new_path = path_orig.parent.joinpath(new_stem+path_orig.suffix)
                    divided_values_paths[i] = new_path
                
                # apply newly divided data and keys to new dictionaries 
                for i in range(divide_factor):
                    # only if the list of divided values has values in it
                    if len(divided_values[i]) > 0:
                        new_key = divided_values_keys[i]
                        updated_dataset_dict[new_key] = divided_values[i]
                        updated_dataset_paths_dict[new_key] = divided_values_paths[i]
    except ValueError:
        raise ValueError('Expect only one instance of "__" to '+\
            'be in the dictionary keys. Multiple found.')
    return updated_dataset_dict, updated_dataset_paths_dict


if __name__ == '__main__':
    import doctest
    doctest.testmod()
