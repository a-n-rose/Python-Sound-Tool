'''Functions related to handling audio data files. This ranges from making 
audio data match in format to creating training, validation, and test datasets.
'''
import numpy as np
import random
import collections
import math 
import csv
import pathlib
from scipy.io.wavfile import write, read
from scipy.signal import resample
import soundfile as sf
import librosa


import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)

import pysoundtool as pyst

###############################################################################


def loadsound(filename, sr=None, mono=True, dur_sec = None, use_librosa=True):
    '''Loads sound file with scipy.io.wavfile.read or librosa.load (default scipy)
    
    If the sound file is not compatible with scipy's read module
    this functions converts the file to .wav format and/or
    changes the bit depth to be compatible. 
    
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
        scipy.io.wavfile and/or soundfile will be used. (default True)
        
    Returns
    -------
    data : nd.array [size=(num_samples,) or (num_samples, num_channels)]
        The normalized (between 1 and -1) sample data returned 
        according to the specified settings.
    sr : int 
        The sample rate of the loaded samples.
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

def get_incompatible_formats():
    '''According to SoundFile
    '''
    return(['.m4a','.mp3'])

def get_compatible_formats():
    '''According to SoundFile
    '''
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


def save_dict(dict2save, filename, replace=False):
    '''Saves dictionary as csv file to indicated path and filename

    Parameters
    ----------
    dict2save : dict
        The dictionary that is to be saved 
    filename : str 
        The path and name to save the dictionary under. If '.csv' 
        extension is not given, it is added.
    replace : bool, optional
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
    if not replace:
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

def audio2datasets(label_wavfiles, perc_train=0.8, limit=None, seed=None):
    '''Organizes all audio in audio class directories into datasets (randomized).

    Parameters
    ----------
    label_wavfiles : str, pathlib.PosixPath, or dict
        path to the dictionary where audio class labels and the 
        paths of all audio files belonging to each class are or will
        be stored. The dictionary with the labels and their encoded values
        can also directly supplied here.
    perc_train : int, float
        The percentage or decimal representing the amount of training
        data compared to the test and validation data (default 0.8)
    seed : int, optional
        A value to allow random order of audiofiles to be predictable. 
        (default None). If None, the order of audiofiles will not be predictable.

    Returns
    -------
    dataset_audio : tuple
        Named tuple including three lists of tuples: the train, validation, 
        and test lists, respectively. The tuples within the lists contain
        the encoded label integer (e.g. 0 instead of 'air_conditioner') and
        the audio paths associated to that class and dataset.
    '''
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
    if isinstance(label_wavfiles, dict):
        class_waves_dict = label_wavfiles
    else:
        class_waves_dict = pyst.data.load_dict(label_wavfiles)
    count = 0
    row = 0
    train = []
    val = []
    test = []
    for key, value in class_waves_dict.items():
        if isinstance(value, str):
            audiolist = pyst.paths.string2list(value)
        else:
            audiolist = value
        train_waves, val_waves, test_waves = waves2dataset(audiolist)

        for i, wave in enumerate(train_waves):
            train.append(tuple([key, wave]))

        for i, wave in enumerate(val_waves):
            val.append(tuple([key, wave]))

        for i, wave in enumerate(test_waves):
            test.append(tuple([key, wave]))
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


# TODO speed this up, e.g. preload noise data?
def create_autoencoder_data(cleandata_path, noisedata_path, trainingdata_dir,
                               perc_train=0.8, perc_val=0.2,  limit=None,
                               noise_scales=[0.3,0.2,0.1], sr = 22050,
                               seed = None):
    '''Organizes all audio in audio class directories into datasets.
    
    Expects the name/id of files/data in the output_folder to be included 
    in those in the input_folder. For example, take the output file 'wav1.wav' 
    and the input file 'wav1_noisy.wav'. 'wav1' is in 'wav1_noisy' but 'wav1_noisy' 
    is not in 'wav1'. (Autencoders tend to take in noisy data to produce cleaner
    versions of that data.)

    Parameters
    ----------
    cleandata_path : str, pathlib.PosixPath
        Name of folder containing clean audio data for autoencoder. E.g. 'clean_speech'
    noisedata_path : str, pathlib.PosixPath
        Name of folder containing noise to add to clean data. E.g. 'noise'
    trainingdata_dir : str, pathlib.PosixPath
        Directory to save newly created train, validation, and test data
    perc_train : int, float
        The percentage or decimal representing the amount of training
        data compared to the test data (default 0.8)
    perc_val : int, float
        The percentage or decimal representing the amount of training data to 
        reserve for validation (default 0.2)
    limit : int, optional
        Limit in number of audiofiles used for training data
    noise_scales : list of floats
        List of varying scales to apply to noise levels, for example, to 
        allow for varying amounts of noise. (default [0.3,0.2,0.1]) The noise
        sample will be multiplied by the scales.
    seed : int 
        A value to allow random order of audiofiles to be predictable. 
        (default None). If None, the order of audiofiles will not be predictable.
        

    Returns
    -------
    saveinput_path : pathlib.PosixPath
        Path to where noisy train, validation, and test audio data are located
    saveoutput_path : pathlib.PosixPath   
        Path to where clean train, validation, and test audio data are located
    '''
    import math
    import time
    
    start = time.time()
    if perc_train > 1:
        perc_train /= 100.
    if perc_val > 1:
        perc_val /= 100.
    if perc_train > 1:
        raise ValueError('The percentage value of train data exceeds 100%')
    if perc_val > 1:
        raise ValueError('The percentage value of validation data exceeds 100%')

    # change string path to pathlib
    cleandata_path = pyst.paths.str2path(cleandata_path)
    noisedata_path = pyst.paths.str2path(noisedata_path)
    trainingdata_dir = pyst.paths.str2path(trainingdata_dir)
    
    cleandata_folder = 'clean'
    noisedata_folder = 'noisy'
    if limit is not None:
        cleandata_folder += '_limit'+str(limit)
        noisedata_folder += '_limit'+str(limit)
    
    newdata_clean_dir = trainingdata_dir.joinpath(cleandata_folder)
    newdata_noisy_dir = trainingdata_dir.joinpath(noisedata_folder)
    
    # create directory to save new data (if not exist)
    newdata_clean_dir = pyst.paths.prep_path(newdata_clean_dir, create_new = True)
    newdata_noisy_dir = pyst.paths.prep_path(newdata_noisy_dir, create_new = True)
   
    # TODO test for existence of directories/files
    # for example: q.exists() or q.is_dir() (pathlib objects) 
    clean_datapaths = []
    noisy_datapaths = []
    
    for dataset in ['train', 'val', 'test']:
        saveinput_dataset = newdata_noisy_dir.joinpath(dataset)
        noisy_datapaths.append(saveinput_dataset)
        
        saveoutput_dataset = newdata_clean_dir.joinpath(dataset)
        clean_datapaths.append(saveoutput_dataset)
    
    # TODO expand available file types
    # pathlib includes hidden files... :(
    cleanaudio = sorted(cleandata_path.glob('*.wav'))
    noiseaudio = sorted(noisedata_path.glob('*.wav'))
    
    # remove hidden files
    cleanaudio = [x for x in cleanaudio if x.parts[-1][0] != '.']
    noiseaudio = [x for x in noiseaudio if x.parts[-1][0] != '.']
    
    if seed is not None:
        random.seed(seed)
    random.shuffle(cleanaudio)
    
    if limit is not None:
        cleanaudio = cleanaudio[:limit]
    
    percentage_training = math.floor(perc_train * len(cleanaudio))
    train_audio, test_audio = cleanaudio[:percentage_training], \
        cleanaudio[percentage_training:]

    percentage_val = math.floor((1 - perc_val) * len(train_audio))
    train_audio, val_audio = train_audio[:percentage_val], train_audio[percentage_val:]
    
    for j, dataset_path in enumerate(clean_datapaths):
        # ensure directory exists:
        pyst.paths.prep_path(dataset_path, create_new=True)
        pyst.paths.prep_path(noisy_datapaths[j], create_new=True)
        
        if 'train' in dataset_path.parts[-1]:
            print('\nProcessing train data...')
            audiopaths = train_audio
        elif 'val' in dataset_path.parts[-1]:
            print('\nProcessing val data...')
            audiopaths = val_audio
        elif 'test' in dataset_path.parts[-1]:
            print('\nProcessing test data...')
            audiopaths = test_audio
        for i, wavefile in enumerate(audiopaths):
            pyst.utils.print_progress(iteration=i, 
                        total_iterations=len(audiopaths),
                        task='clean and noisy audio data generation')
            # no random seed applied here:
            # each choice would be the same for each iteration
            noise = random.choice(noiseaudio)
            scale = random.choice(noise_scales)
            clean_stem = wavefile.stem
            noise_stem = noise.stem
            noise_data, sr = pyst.soundprep.loadsound(
                noise, sr=sr)
            clean_data, sr2 = pyst.soundprep.loadsound(
                wavefile, sr=sr)
            clean_seconds = len(clean_data)/sr2
            noisy_data, sr = pyst.soundprep.add_sound_to_signal(
                wavefile, noise, scale = scale, delay_target_sec=0, total_len_sec = clean_seconds
                )
            noisydata_filename = noisy_datapaths[j].joinpath(clean_stem+'_'+noise_stem\
                +'_scale'+str(scale)+'.wav')
            cleandata_filename = dataset_path.joinpath(clean_stem+'.wav')     
            write(noisydata_filename, sr, noisy_data)
            write(cleandata_filename, sr, clean_data)

        print('Finished processing {}'.format(dataset_path))
        print('Finished processing {}'.format(noiseaudio[j]))
    end = time.time()
    total_time, units = pyst.utils.adjust_time_units(end-start)
    print('Dataset creation took a total of {} {}.'.format(
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

def convert2wav(filename, sr=None, new_path=False):
    '''Converts and saves soundfile as .wav type in same or new directory.
    
    Parameters
    ----------
    filename : str or pathlib.PosixPath
        The filename of the audiofile to be converted to .wav type
    new_path : str, pathlib.PosixPath, optional 
        If False, the converted files will be saved in same directory as originals.
        If a path is provided, the converted files will be saved there. If no such directory
        exists, one will be created.
    sr : int, optional
        The sample rate to be applied to the signal. If none supplied, the sample rate 
        of the original file will be used.
        
    Returns 
    -------
    f_wavfile : str or pathlib.PosixPath
        The filename and path where the .wav file is saved.
    '''
    import pathlib
    f = pathlib.Path(filename)
    extension_orig = f.suffix
    stem_orig = f.stem
    if not extension_orig:
        f = str(f)
    else:
        f = str(f)[:len(str(f))-len(extension_orig)]
    if new_path:
        f_dir = pyst.paths.prep_path(new_path)
        f_wavfile = new_path + stem_orig + '.wav'        
    else:
        f_wavfile = f+'.wav'
    # soundfile can load several datatypes
    try:
        data, sr = sf.read(filename, samplerate=sr)
    except RuntimeError as e:
        print(e)
        raise RuntimeError('Audioformat `{}` is not compatible with soundfile.'.format(
            os.path.splitext(filename)[1]))
    # save the data at wavfile
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




if __name__ == '__main__':
    import doctest
    doctest.testmod()
