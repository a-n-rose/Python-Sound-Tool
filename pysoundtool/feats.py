 


###############################################################################

import numpy as np
from random import shuffle
import random
import collections
import math 
from scipy.io.wavfile import write

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)

import pysoundtool as pyst





def setup_audioclass_dicts(audio_classes_dir, encoded_labels_path, label_waves_path,
                           limit=None):
    '''Saves dictionaries containing encoded label and audio class wavfiles.

    Parameters
    ----------
    audio_classes_dir : str, pathlib.PosixPath
        Directory path to where all audio class folders are located.
    encoded_labels_path : str, pathlib.PosixPath
        path to the dictionary where audio class labels and their 
        encoded integers are stored or will be stored.
    label_waves_path : str, pathlib.PosixPath
        path to the dictionary where audio class labels and the 
        paths of all audio files belonging to each class are or will
        be stored.
    limit : int, optional
        The integer indicating a limit to number of audiofiles to each class. This
        may be useful if one wants to ensure a balanced dataset (default None)

    Returns
    -------
    label2int_dict : dict
        Dictionary containing the string labels as keys and encoded 
        integers as values.
    '''
    paths, labels = pyst.paths.collect_audio_and_labels(audio_classes_dir)
    label2int_dict, int2label_dict = create_dicts_labelsencoded(set(labels))
    __ = pyst.paths.save_dict(int2label_dict, encoded_labels_path)
    label2audiofiles_dict = create_label2audio_dict(
        set(labels), paths, limit=limit)
    __ = pyst.paths.save_dict(label2audiofiles_dict, label_waves_path)
    return label2int_dict

def create_label2audio_dict(labels_set, paths_list, limit=None, seed=40):
    '''Creates dictionary with audio labels as keys and filename lists as values.

    If no label is found in the filename path, the label is not included
    in the returned dictionary: labels are only included if corresponding
    paths are present.

    Parameters
    ----------
    labels_set : set, list
        Set containing the labels of all audio training classes
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
        A dictionary with audio labels as keys with values being the audio files 
        corresponding to that label

    Examples
    --------
    >>> from pathlib import Path
    >>> labels = set(['vacuum','fridge','wind'])
    >>> paths = [Path('data/audio/vacuum/vacuum1.wav'), 
    ...         Path('data/audio/fridge/fridge1.wav'), 
    ...         Path('data/audio/vacuum/vacuum2.wav'),
    ...         Path('data/audio/wind/wind1.wav')]
    >>> label_waves_dict = create_label2audio_dict(labels, paths)
    >>> label_waves_dict
    OrderedDict([('fridge', [PosixPath('data/audio/fridge/fridge1.wav')]), \
('vacuum', [PosixPath('data/audio/vacuum/vacuum1.wav'), \
PosixPath('data/audio/vacuum/vacuum2.wav')]), \
('wind', [PosixPath('data/audio/wind/wind1.wav')])])
    >>> #to set a limit on number of audiofiles per class:
    >>> create_label2audio_dict(labels, paths, limit=1, seed=40)
    OrderedDict([('fridge', [PosixPath('data/audio/fridge/fridge1.wav')]), \
('vacuum', [PosixPath('data/audio/vacuum/vacuum2.wav')]), \
('wind', [PosixPath('data/audio/wind/wind1.wav')])])
    >>> #change the limited pathways chosen:
    >>> create_label2audio_dict(labels, paths, limit=1, seed=10)
    OrderedDict([('fridge', [PosixPath('data/audio/fridge/fridge1.wav')]), \
('vacuum', [PosixPath('data/audio/vacuum/vacuum1.wav')]), \
('wind', [PosixPath('data/audio/wind/wind1.wav')])])
    '''
    if not isinstance(labels_set, set) and not isinstance(labels_set, list):
        raise TypeError(
            'Expected labels list as type set or list, not type {}'.format(type(
                labels_set)))
    if not isinstance(paths_list, set) and not isinstance(paths_list, list):
        raise TypeError(
            'Expected paths list as type set or list, not type {}'.format(type(
                paths_list)))
    label_waves_dict = collections.OrderedDict()
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
            label_waves_dict[label] = sorted(label_paths)
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

def audio2datasets(audio_classes_dir, encoded_labels_path,
                   label_wavfiles_path, perc_train=0.8, limit=None):
    '''Organizes all audio in audio class directories into datasets.

    If they don't already exist, dictionaries with the encoded labels of the 
    audio classes as well as the wavfiles belonging to each class are saved. 

    Parameters
    ----------
    audio_classes_dir : str, pathlib.PosixPath
        Directory path to where all audio class folders are located.
    encoded_labels_path : str, pathlib.PosixPath
        path to the dictionary where audio class labels and their 
        encoded integers are stored or will be stored.
    label_wavfiles_path : str, pathlib.PosixPath
        path to the dictionary where audio class labels and the 
        paths of all audio files belonging to each class are or will
        be stored.
    perc_train : int, float
        The percentage or decimal representing the amount of training
        data compared to the test and validation data (default 0.8)

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
    if os.path.exists(encoded_labels_path) and \
            os.path.exists(label_wavfiles_path):
        print('Loading preexisting encoded labels file: {}'.format(
            encoded_labels_path))
        print('Loading preexisting label and wavfiles file: {}'.format(
            label_wavfiles_path))
        encodelabels_dict_inverted = pyst.paths.load_dict(encoded_labels_path)
        label2int = {}
        for key, value in encodelabels_dict_inverted.items():
            label2int[value] = key
        class_waves_dict = pyst.paths.load_dict(label_wavfiles_path)
    else:
        kwargs = {'audio_classes_dir': audio_classes_dir,
                  'encoded_labels_path': encoded_labels_path,
                  'label_waves_path': label_wavfiles_path,
                  'limit': limit}
        label2int = setup_audioclass_dicts(**kwargs)
        # load the just created label to wavfiles dictionary
        class_waves_dict = pyst.paths.load_dict(label_wavfiles_path)
    count = 0
    row = 0
    train = []
    val = []
    test = []
    for key, value in class_waves_dict.items():
        audiolist = pyst.paths.string2list(value)
        train_waves, val_waves, test_waves = waves2dataset(audiolist)

        for i, wave in enumerate(train_waves):
            train.append(tuple([label2int[key], wave]))

        for i, wave in enumerate(val_waves):
            val.append(tuple([label2int[key], wave]))

        for i, wave in enumerate(test_waves):
            test.append(tuple([label2int[key], wave]))
    # be sure the classes are not in any certain order
    shuffle(train)
    shuffle(val)
    shuffle(test)
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
                               noise_scales=[0.3,0.2,0.1], sr = 22050):
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
            pyst.tools.print_progress(iteration=i, 
                        total_iterations=len(audiopaths),
                        task='clean and noisy audio data generation')
            noise = random.choice(noiseaudio)
            scale = random.choice(noise_scales)
            clean_stem = wavefile.stem
            noise_stem = noise.stem
            noise_data, sr = pyst.soundprep.loadsound(
                noise, samplerate=sr)
            clean_data, sr2 = pyst.soundprep.loadsound(
                wavefile, samplerate=sr)
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
    total_time, units = pyst.tools.adjust_time_units(end-start)
    print('Dataset creation took a total of {} {}.'.format(
        round(total_time,2), 
        units))

    return newdata_noisy_dir, newdata_clean_dir

def extract_autoencoder_data():
    pass





class AcousticData:
    '''Base class for handling acoustic data and machine learning.
    
    Holds attributes relevant to acosutic data and how it is handled. Goal is to 
    be able to prepare acoustic data for just about any model, and any format.
    
    Attributes
    ----------
    feature_type : str 
        Options: 'signal' for raw samples, 'stft' for magnitude/power spectrum 
        related data, 'fbank' for mel filterbank energies, and 'mfcc' for mel 
        frequency cepstral coefficients. (default 'fbank') # TODO add DCT.
    sr : int 
        Sample rate of audio data, especially relevant for dealing with samples. 
        (default 48000)
    n_melfilters : int 
        The number of mel filters to apply, especially 
        relevant for 'fbank' features. (default 40)
    n_mfcc : int 
        The number of mel cepstral coefficients. For speech, around 13 is normal. 
        Others have used 22 and 40 coefficients. (default None)
    win_size : int, float, optional
        window in milliseconds to process acoustic data. (default 25 ms)
    percent_overlap : float, optional
        Percent of overlapping samples between windows. For example, if `win_size` 
        is 25 ms, a `percent_overlap` of 0.5 would result in 12.5 ms of 
        overlapping samples of consecutive windows. (default 0.5)
    window_type : 'str', optional
        Window type applied to each processing window. Options: 'hamming', 
        'hann'. (default 'hamming')
    real_signal : bool 
        IF True, half of the fft will be used in feature extraction. If false, 
        the full FFT will be used, which is symmetrical. For acoustic data, using 
        half the FFT may be beneficial for increasing efficiency.
    '''
    def __init__(self,
                 feature_type = 'fbank',
                 sr= 48000, 
                 n_melfilters = 40,
                 n_mfcc = None,
                 win_size = 25,
                 percent_overlap = 0.5,
                 window_type = 'hamming',
                 real_signal = True,
                 feature_shape = None,
                 ):
        self.feature_type = feature_type
        self.sr = sr
        self.n_melfilters = n_melfilters
        self.n_mfcc = n_mfcc
        self.win_size = win_size
        self.percent_overlap = percent_overlap
        self.win_shift = self.win_size * self.percent_overlap
        self.window_type = window_type
        self.frame_length = pyst.dsp.calc_frame_length(self.window_size,
                                                  self.sr)
        self.fft_bins = self.frame_length
        self.real_signal = real_signal
        
class FeatPrep_SpeechRecognition(AcousticData):
    pass

class FeatPrep_DenoiseCNN(AcousticData):
    '''FeatPrep_DenoiseAutoencoder, FeatPrep_DenoiseImage, FeatPrep_DenoiseWavelet?
    '''
    pass

class FeatPrep_EmotionRecognition(AcousticData):
    pass

class FeatPrep_ClinicalDiagnosis(AcousticData):
    pass

class FeatPrep_LanguageClassifier(AcousticData):
    pass

class FeatPrep_SpeakerRecognition(AcousticData):
    pass

class FeatPrep_SoundClassifier(AcousticData):

    def __init__(self,
                 feature_type='fbank',
                 sr=48000,
                 n_melfilters=40,
                 n_mfcc=None,
                 win_size=25,
                 percent_overlap=0.5,
                 training_segment_ms=1000,
                 num_columns=None,
                 num_images_per_audiofile=None,
                 num_waves=None,
                 feature_sets=None,
                 window_type=None,
                 augment_data=False
                 ):
        '''
        Set defaults for feature extraction according to the methodology of 
        the paper by Sehgal and Kehtarnavaz (2018).


        Parameters
        ----------
        feature_type : str
            Which type of features extracted. Options: 'mfcc', 'fbank'. 
            (default 'fbank')
        sr : int, default=48000
            The audio files will be processed by this sampling rate.
        n_melfilters : int, default=40
            Number of mel filters applied when calculating the melspectrogram
        win_size : int or float, default=25
            In ms, the window size to calculate melspectrogram
        win_shift : int or float, default=12.5
            The amount of overlap of melspectrogram calculations. Default is
            50% overlap
        training_segment_ms : int or float, default=62.5
            In ms, the size of melspectrogram 'image' to feed to CNN.
        '''
        AcousticData.__init__(self,
                              feature_type = feature_type,
                              sr = sr, 
                              n_melfilters = n_melfilters, 
                              n_mfcc = n_mfcc,
                              win_size = win_size,
                              percent_overlap = percent_overlap, 
                              window_type = window_type)

        self.training_segment_ms = training_segment_ms
        self.num_columns = num_columns or self.num_filters
        if 'mfcc' in feature_type:
            if self.num_mfcc is None:
                self.num_mfcc = self.num_filters
            self.num_columns = num_columns or self.num_mfcc
        self.num_images_per_audiofile = num_images_per_audiofile or 1
        self.num_waves = num_waves
        if 'fbank' in self.feature_type or 'mfcc' in self.feature_type:
            self.feature_sets = feature_sets or self.calc_filter_image_sets()
        self.augment_data = augment_data

    def calc_filter_image_sets(self):
        '''
        calculates how many feature sets create a full image, given window size, 
        window shift, and desired image length in milliseconds.
        '''
        sets = pyst.dsp.calc_num_subframes(self.training_segment_ms,
                                      self.window_size,
                                      self.window_shift)
        return sets

    def get_max_samps(self, filter_features, num_sets):
        '''
        calculates the maximum  number of samples of a particular wave's 
        features that would also create a full image
        '''
        max_num_samps = (filter_features.shape[0]//num_sets) * num_sets
        return int(max_num_samps)

    def samps2feats(self, y, augment_data=None):
        '''Gets features from section of samples, at varying volumes.
        '''
        # when applying this to new data, dont't want to augment it.
        if not augment_data and self.augment_data:
            samples = pyst.augmentdata.spread_volumes(y)
        else:
            samples = (y,)
        feat_matrix = pyst.dsp.create_empty_matrix(
            (int(self.num_images_per_audiofile*len(samples)),
             self.feature_sets,
             self.num_columns))
        row = 0
        for sample_set in samples:
            feats, frame_length, window_size_ms = pyst.dsp.collect_features(
                feature_type=self.feature_type,
                samples=y,
                sr=self.sr,
                window_size_ms=self.window_size,
                window_shift_ms=self.window_shift,
                num_filters=self.num_filters,
                num_mfcc=self.num_mfcc,
                window_function=self.window_type)
            assert frame_length == self.fft_bins
            feat_matrix[row:row+len(feats)] = feats[:self.feature_sets]
            row += len(feats)
        return feat_matrix

    def extractfeats(self, sounddata, dur_sec=None, augment_data=None):
        '''Organizes feat extraction of each audiofile according to class attributes.
        '''
        sounddata = pyst.paths.str2path(sounddata)
        if pyst.paths.is_audio_ext_allowed(sounddata):
            y, sr = pyst.dsp.load_signal(sounddata,
                                    sampling_rate=self.sr,
                                    dur_sec=dur_sec)
        else:
            print('The following datatype for sounddata is not understood:')
            print(type(sounddata))
            print('Data presented: ', sounddata)
            sys.exit()
        if augment_data is not None and augment_data != self.augment_data:
            feats = self.samps2feats(y, augment_data=augment_data)
        else:
            feats = self.samps2feats(y)
        assert feats.shape[1] == self.feature_sets
        assert feats.shape[2] == self.num_columns
        feats = feats[:self.feature_sets]
        return feats

    def get_feats(self, list_waves, dur_sec=None):
        '''collects fbank or mfcc features of entire wavfile list
        '''
        tot_waves = len(list_waves)
        num_images = 1
        if self.augment_data:
            num_versions_samples = 3  # three different volume augmentaions
        else:
            num_versions_samples = 1
        # for training scene classifier, just want 1 'image' per scen
        tot_len_matrix = int(tot_waves * num_images * num_versions_samples)
        # create empty matrix to fill features into
        # +1 is for the label column
        shape = (tot_len_matrix, self.feature_sets, self.num_columns+1)
        feats_matrix = pyst.dsp.create_empty_matrix(shape, complex_vals=False)
        row = 0  # keep track of where we are in filling the empty matrix
        for i, label_wave in enumerate(list_waves):
            # collect label information
            # list_waves contains tuples with (soundclass_int, wavfile)
            label = int(label_wave[0])
            wave = label_wave[1]
            if not os.path.exists(wave):
                print('Not Found: ', wave)
            else:
                feats = self.extractfeats(wave, dur_sec=dur_sec)
                # add label column - need label to stay with the features!
                label_col = np.full(
                    (feats.shape[0], self.feature_sets, 1), label)
                feats = np.concatenate((feats, label_col), axis=2)
                # fill the matrix with the features and labels
                feats_matrix[row:row+feats.shape[0]] = feats
                # actualize the row for the next set of features to fill it with
                row += feats.shape[0]
                # print on screen the progress
                progress = row / tot_len_matrix * 100
                sys.stdout.write("\r%d%% through current dataset" % progress)
                sys.stdout.flush()
                if not progress < 100:
                    print("\nCompleted feature extraction.")
        if row < tot_len_matrix:
            diff = tot_len_matrix - row
            print('A total of {} wavfiles were not found. \
                  \nExpected amount: {} total waves.'.format(
                      diff, tot_len_matrix))
            feats_matrix = feats_matrix[:row]
        # randomize rows
        np.random.shuffle(feats_matrix)
        return feats_matrix

    def get_save_feats(self, wave_list, directory4features, filename):
        if self.num_waves is None:
            self.num_waves = len(wave_list)
        else:
            self.num_waves += len(wave_list)
        feats = self.get_feats(
            wave_list, dur_sec=self.training_segment_ms/1000.0)
        save2file = directory4features.joinpath(filename)
        pyst.paths.save_feature_data(save2file, feats)
        return None

    def save_class_settings(self, path, replace=False):
        '''saves class settings to dictionary
        '''
        class_settings = self.__dict__
        filename = 'settings_{}.csv'.format(self.__class__.__name__)
        featuresettings_path = path.joinpath(filename)
        pyst.paths.save_dict(
            class_settings, featuresettings_path, replace=replace)
        return None


def prepfeatures(filter_class, feature_type='mfcc', num_filters=40,
                 segment_dur_ms=1000, limit=None, augment_data=False,
                 sampling_rate=48000):
    '''Pulls info from 'filter_class' instance to then extract, save features

    Parameters
    ----------
    filter_class : class
        The class instance holding attributes relating to path structure
        and filenames necessary for feature extraction
    feature_type : str, optional
        Acceptable inputs: 'mfcc' and 'fbank'. These are the features that
        will be extracted from the audio and saved (default 'mfcc')
    num_filters : int, optional
        The number of mel filters used during feature extraction. This number 
        ranges for 'mfcc' extraction between 13 and 40 and for 'fbank'
        extraction between 20 and 128. The higher the number, the greater the 
        computational load and memory requirement. (default 40)
    segment_dur_ms : int, optional
        The length in milliseconds of the acoustic data to extract features
        from. If 1000 ms, 1 second of acoustic data will be processed; 1 sec 
        of feature data will be extracted. If not enough audio data is present,
        the feature data will be zero padded. (default 1000)

    Returns
    ----------
    feats_class : class
        The class instance holding attributes relating to the current 
        feature extraction session
    filter_class : class
        The updated class instance holding attributes relating to path
        structure
    '''
    # extract features
    # create namedtuple with train, val, and test wavfiles and labels
    datasetwaves = pyst.feats.audio2datasets(filter_class.audiodata_dir,
                                          filter_class.labels_encoded_path,
                                          filter_class.labels_waves_path,
                                          limit=limit)

    # TODO make baseclass PrepFeatures for ClassifierFeats, AutoencoderFeats
    feats_class = FeatPrep_SoundClassifier(feature_type=feature_type,
                               num_filters=num_filters,
                               training_segment_ms=segment_dur_ms,
                               augment_data=augment_data)
    # incase an error occurs; save this before extraction starts
    feats_class.save_class_settings(filter_class.features_dir)
    for i, dataset in enumerate(datasetwaves._fields):
        feats_class.get_save_feats(datasetwaves[i],
                                   filter_class.features_dir,
                                   '{}.npy'.format(dataset))
    # save again with added information, ie the total number of wavfiles
    feats_class.save_class_settings(filter_class.features_dir, replace=True)
    filter_class.features = filter_class.features_dir
    return feats_class, filter_class


def prepfeatures_autoencoder(filter_class, feature_type='mfcc', num_filters=40,
                 segment_dur_ms=1000, limit=None, augment_data=False,
                 sampling_rate=48000):
    '''Pulls info from 'filter_class' instance to then extract, save features

    Parameters
    ----------
    filter_class : class
        The class instance holding attributes relating to path structure
        and filenames necessary for feature extraction
    feature_type : str, optional
        Acceptable inputs: 'mfcc' and 'fbank'. These are the features that
        will be extracted from the audio and saved (default 'mfcc')
    num_filters : int, optional
        The number of mel filters used during feature extraction. This number 
        ranges for 'mfcc' extraction between 13 and 40 and for 'fbank'
        extraction between 20 and 128. The higher the number, the greater the 
        computational load and memory requirement. (default 40)
    segment_dur_ms : int, optional
        The length in milliseconds of the acoustic data to extract features
        from. If 1000 ms, 1 second of acoustic data will be processed; 1 sec 
        of feature data will be extracted. If not enough audio data is present,
        the feature data will be zero padded. (default 1000)

    Returns
    ----------
    feats_class : class
        The class instance holding attributes relating to the current 
        feature extraction session
    filter_class : class
        The updated class instance holding attributes relating to path
        structure
    '''
    # extract features
    # create namedtuple with train, val, and test wavfiles and labels
    # TODO update filter_class with new autoencoder attributes
    datasetwaves = pyst.feats.audio2datasets(filter_class.audiodata_dir,
                                          filter_class.inputdata_folder,
                                          filter_class.outputdata_folder,
                                          filter_class.features_dir,
                                          limit=limit)

    # TODO make baseclass PrepFeatures for ClassifierFeats, AutoencoderFeats
    feats_class = PrepFeatures(feature_type=feature_type,
                               num_filters=num_filters,
                               training_segment_ms=segment_dur_ms,
                               augment_data=augment_data)
    # incase an error occurs; save this before extraction starts
    feats_class.save_class_settings(filter_class.features_dir)
    for i, dataset in enumerate(datasetwaves._fields):
        feats_class.get_save_feats(datasetwaves[i],
                                   filter_class.features_dir,
                                   '{}.npy'.format(dataset))
    # save again with added information, ie the total number of wavfiles
    feats_class.save_class_settings(filter_class.features_dir, replace=True)
    filter_class.features = filter_class.features_dir
    return feats_class, filter_class

def getfeatsettings(feature_info):
    '''Loads prev extracted feature settings into new feature class instance

    This is useful if one wants to extract new features that match the
    dimensions and settings of previously extracted features.

    Parameters
    ----------
    feature_info : dict, class
        Either a dictionary or a class instance that holds the path
        attribute to a dictionary. 

    Returns
    -------
    feats_class : class 
        Feature extraction class instance with the same settings as the 
        settings dictionary
    '''
    if isinstance(feature_info, dict):
        feature_settings = feature_info
    else:
        featuresettings_path = pyst.paths.load_settings_file(
            feature_info.features_dir)
        feature_settings = pyst.paths.load_dict(featuresettings_path)
    sr = pyst.tools.make_number(feature_settings['sr'])
    window_size = pyst.tools.make_number(feature_settings['window_size'])
    window_shift = pyst.tools.make_number(feature_settings['window_shift'])
    feature_sets = pyst.tools.make_number(feature_settings['feature_sets'])
    feature_type = feature_settings['feature_type']
    num_columns = pyst.tools.make_number(feature_settings['num_columns'])
    num_images_per_audiofile = pyst.tools.make_number(
        feature_settings['num_images_per_audiofile'])
    training_segment_ms = pyst.tools.make_number(
        feature_settings['training_segment_ms'])
    if 'fbank' in feature_type.lower():
        feature_type = 'fbank'
        num_filters = num_columns
        num_mfcc = None
    elif 'mfcc' in feature_type.lower():
        feature_type = 'mfcc'
        if num_columns != 40:
            num_filters = 40
        else:
            num_filters = num_columns
        num_mfcc = num_columns
    feats_class = PrepFeatures(feature_type=feature_type,
                               sampling_rate=sr,
                               num_filters=num_filters,
                               num_mfcc=num_mfcc,
                               window_size=window_size,
                               window_shift=window_shift,
                               training_segment_ms=training_segment_ms,
                               num_images_per_audiofile=num_images_per_audiofile)
    assert feature_sets == feats_class.feature_sets
    return feats_class


################################### TODO Consolidate with other functions ################
# TODO Clean up   
# TODO add graph for each channel? For all feature types?
def visualize_feats(feature_matrix, feature_type, 
                    save_pic=False, name4pic=None, scale=None,
                    title=None, sample_rate=None):
    '''Visualize feature extraction; frames on x axis, features on y axis. Uses librosa to scale the data if scale applied.
    
    Parameters
    ----------
    feature_matrix : np.ndarray [shape=(num_samples,), (num_samples, num_channels), or (num_features, num_frames), dtype=np.float].
        Matrix of features. If the features are not of type 'signal' and the
        shape is 1 D, one dimension will be added to be plotted with a colormesh.
    feature_type : str
        Options: 'signal', 'stft', 'mfcc', or 'fbank' features, or 
        what user would like to name the feature set.
        signal: the 1 D samples of sound.
        STFT: short-time Fourier transform
        MFCC: mel frequency cepstral coefficients.
        FBANK: mel-log filterbank energies (default 'fbank').
    save_pic : bool
        True to save image as .png; False to just plot it.
    name4pic : str, optional
        If `save_pic` set to True, the name the image should be saved under.
    scale : str, optional
        If features need to be adjusted, e.g. from power to decibels. 
        Default is None.
    title : str, optional
        The title for the graph. If None, `feature_type` is used.
    sample_rate : int, optional
        Useful for plotting a signal type feature matrix. Allows x-axis to be
        presented as time in seconds.
    '''
    # ensure real numbers
    if isinstance(feature_matrix[0], np.complex):
        feature_matrix = feature_matrix.real
    # features presented via colormesh need 2D format.
    if len(feature_matrix.shape) == 1:
        feature_matrix = np.expand_dims(feature_matrix, axis=1)
    if 'fbank' in feature_type:
        axis_feature_label = 'Num Mel Filters'
    elif 'mfcc' in feature_type:
        axis_feature_label = 'Num Mel Freq Cepstral Coefficients'
    elif 'stft' in feature_type:
        axis_feature_label = 'Number of frames'
    elif 'signal' in feature_type:
        axis_feature_label = 'Amplitude'
    else:
        axis_feature_label = 'Energy'
    if scale is None or feature_type == 'signal':
        energy_label = 'energy'
        pass
    elif scale == 'power_to_db':
        feature_matrix = librosa.power_to_db(feature_matrix)
        energy_label = 'decicels'
    elif scale == 'db_to_power':
        feature_matrix = librosa.db_to_power(feature_matrix)
        energy_label = 'power'
    elif scale == 'amplitude_to_db':
        feature_matrix = librosa.amplitude_to_db(feature_matrix)
        energy_label = 'decibels'
    elif scale == 'db_to_amplitude':
        feature_matrix = librosa.db_to_amplitude(feature_matrix)
        energy_label = 'amplitude'
    plt.clf()
    if feature_type != 'signal':
        x_axis_label = 'Frequency bins'
    else:
        x_axis_label = 'Samples over time'
    if feature_type == 'signal':
        # transpose matrix if second dimension is larger - probably 
        # because channels are in first dimension. Expect in second dimension
        if not feature_matrix.shape[0] > feature_matrix.shape[1]:
            feature_matrix = feature_matrix.T
        if sample_rate is not None:
            x_axis_label = 'Time (sec)'
            dur_sec = feature_matrix.shape[0] / sample_rate
            time_sec = pyst.dsp.get_time_points(dur_sec, sample_rate)
            for channel in range(feature_matrix.shape[1]):
                data = feature_matrix[:,channel]
                # overlay the channel data
                plt.plot(time_sec, data)
        else:
            for channel in range(feature_matrix.shape[1]):
                data = feature_matrix[:,channel]
                # overlay the channel data
                plt.plot(data)
                ##display.waveplot(data,sr=sample_rate)
        x_axis_label += ' across {} channel(s)'.format(channel+1)
    else:
        plt.pcolormesh(feature_matrix.T)
        ## display.specshow(feature_matrix.T, sr=sample_rate)
        plt.colorbar(label=energy_label)
    plt.xlabel(x_axis_label)
    plt.ylabel(axis_feature_label)
    # if feature_matrix has multiple frames, not just one
    if feature_matrix.shape[1] > 1 and 'signal' not in feature_type:
        # the xticks basically show time but need to be multiplied by 0.01
        plt.xlabel('Time (sec)') 
        locs, labels = plt.xticks()
        new_labels=[str(round(i*0.01,1)) for i in locs]
        plt.xticks(ticks=locs,labels=new_labels)
        plt.ylabel('Frequency bins')
    if title is None:
        plt.title('{} Features'.format(feature_type.upper()))
    else:
        plt.title(title)
    if save_pic:
        outputname = name4pic or 'visualize{}feats'.format(feature_type.upper())
        plt.savefig('{}.png'.format(outputname))
    else:
        plt.show()

def get_feats(sound, 
              features='fbank', 
              win_size_ms = 20, 
              win_shift_ms = 10,
              num_filters=40,
              num_mfcc=None, 
              samplerate=None, 
              limit=None,
              mono=None):
    '''Feature extraction depending on set parameters; frames on y axis, features x axis
    
    Parameters
    ----------
    sound : str or numpy.ndarray
        If str, wavfile (must be compatible with scipy.io.wavfile). Otherwise 
        the samples of the sound data. Note: in the latter case, `samplerate`
        must be declared.
    features : str
        Either 'mfcc' or 'fbank' features. MFCC: mel frequency cepstral
        coefficients; FBANK: mel-log filterbank energies (default 'fbank')
    win_size_ms : int or float
        Window length in milliseconds for Fourier transform to be applied
        (default 20)
    win_shift_ms : int or float 
        Window overlap in milliseconds; default set at 50% window size 
        (default 10)
    num_filters : int
        Number of mel-filters to be used when applying mel-scale. For 
        'fbank' features, 20-128 are common, with 40 being very common.
        (default 40)
    num_mfcc : int
        Number of mel frequency cepstral coefficients. First coefficient
        pertains to loudness; 2-13 frequencies relevant for speech; 13-40
        for acoustic environment analysis or non-linguistic information.
        Note: it is not possible to choose only 2-13 or 13-40; if `num_mfcc`
        is set to 40, all 40 coefficients will be included.
        (default None). 
    samplerate : int, optional
        The sample rate of the sound data or the desired sample rate of
        the wavfile to be loaded. (default None)
    limit : float, optional
        Time in seconds to limit in loading a signal. (default None)
    mono: bool, optional
        For loading an audiofile, True will result in only one channel of 
        data being loaded; False will allow additional channels be loaded. 
        (default None, which results in mono channel data)
        
    Returns
    -------
    feats : tuple (num_samples, sr) or np.ndarray [size (num_frames, num_filters) dtype=np.float or np.complex]
        Feature data. If `feature_type` is 'signal', returns a tuple containing samples and sampling rate. If `feature_type` is of another type, returns np.ndarray with shape (num_frames, num_filters/features)
    '''
    if isinstance(sound, str):
        if mono is None:
            mono = True
        data, sr = librosa.load(sound, sr=samplerate, duration=limit, mono=mono)
        if mono is False and len(data.shape) > 1:
            index_samples = np.argmax(data.shape)
            index_channels = np.argmin(data.shape)
            num_channels = data.shape[index_channels]
            # transpose data to be (samples, num_channels) rather than (num_channels, samples)
            if index_channels == 0:
                data = data.T 
            # remove additional channel for 'stft', 'fbank' etc. feature
            # extraction
            if 'signal' not in features and num_channels > 1:
                data = data[:,0]
    else:
        if samplerate is None:
            raise ValueError('No samplerate given. Either provide '+\
                'filename or appropriate samplerate.')
        data, sr = sound, samplerate
    try:
        if 'fbank' in features:
            feats = librosa.feature.melspectrogram(
                data,
                sr = sr,
                n_fft = int(win_size_ms * sr // 1000),
                hop_length = int(win_shift_ms*0.001*sr),
                n_mels = num_filters).T
            # have found better results if not conducted:
            # especially if recreating audio signal later
            #feats -= (np.mean(feats, axis=0) + 1e-8)
        elif 'mfcc' in features:
            if num_mfcc is None:
                num_mfcc = num_filters
            feats = librosa.feature.mfcc(
                data,
                sr = sr,
                n_mfcc = num_mfcc,
                n_fft = int(win_size_ms * sr // 1000),
                hop_length = int(win_shift_ms*0.001*sr),
                n_mels = num_filters).T
            #feats -= (np.mean(feats, axis=0) + 1e-8)
        elif 'stft' in features:
            feats = librosa.stft(
                data,
                n_fft = int(win_size_ms * sr // 1000),
                hop_length = int(win_shift_ms*0.001*sr)).T
            #feats -= (np.mean(feats, axis=0) + 1e-8)
        elif 'signal' in features:
            feats = (data, sr)
    except librosa.ParameterError as e:
        feats = get_feats(np.asfortranarray(data),
                          features = features,
                          win_size_ms = win_size_ms,
                          win_shift_ms = win_shift_ms,
                          num_filters = num_filters,
                          num_mfcc = num_mfcc,
                          samplerate = sr,
                          limit = limit,
                          mono = mono)
    return feats

def visualize_audio(audiodata, feature_type='fbank', win_size_ms = 20, \
    win_shift_ms = 10, num_filters=40, num_mfcc=40, samplerate=None,\
        save_pic=False, name4pic=None, power_scale=None, mono=None):
    '''Visualize feature extraction depending on set parameters. Does not use Librosa.
    
    Parameters
    ----------
    audiodata : str or numpy.ndarray
        If str, wavfile (must be compatible with scipy.io.wavfile). Otherwise 
        the samples of the sound data. Note: in the latter case, `samplerate`
        must be declared.
    feature_type : str
        Options: 'signal', 'mfcc', or 'fbank' features. 
        MFCC: mel frequency cepstral
        coefficients; FBANK: mel-log filterbank energies (default 'fbank')
    win_size_ms : int or float
        Window length in milliseconds for Fourier transform to be applied
        (default 20)
    win_shift_ms : int or float 
        Window overlap in milliseconds; default set at 50% window size 
        (default 10)
    num_filters : int
        Number of mel-filters to be used when applying mel-scale. For 
        'fbank' features, 20-128 are common, with 40 being very common.
        (default 40)
    num_mfcc : int
        Number of mel frequency cepstral coefficients. First coefficient
        pertains to loudness; 2-13 frequencies relevant for speech; 13-40
        for acoustic environment analysis or non-linguistic information.
        Note: it is not possible to choose only 2-13 or 13-40; if `num_mfcc`
        is set to 40, all 40 coefficients will be included.
        (default 40). 
    samplerate : int, optional
        The sample rate of the sound data or the desired sample rate of
        the wavfile to be loaded. (default None)
    mono : bool, optional
        When loading an audiofile, True will limit number of channels to
        one; False will allow more channels to be loaded. (default None, 
        which results in mono channel loading.)
    '''
    feats = get_feats(audiodata, features=feature_type, 
                      win_size_ms = win_size_ms, win_shift_ms = win_shift_ms,
                      num_filters=num_filters, num_mfcc = num_mfcc, samplerate=samplerate,
                      mono=mono)
    if 'signal' in feature_type:
        feats, sr = feats
    else:
        sr = None
    visualize_feats(feats, feature_type=feature_type, sample_rate=sr,
                    save_pic = save_pic, name4pic=name4pic, scale=power_scale)



if __name__ == "__main__":
    import doctest
    doctest.testmod()
