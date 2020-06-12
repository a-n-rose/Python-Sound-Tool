'''feats module includes functions related to handling, extracting, and 
prepping features for working with audio in the digital domain (machine 
learning, filtering)
''' 


###############################################################################
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)

import numpy as np
import math
import librosa
import pathlib
from python_speech_features import logfbank, mfcc
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib.pyplot as plt
import pysoundtool as pyst





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

    def extractfeats(self, sounddata, dur_sec=None, augment_data=None):
        '''Organizes feat extraction of each audiofile according to class attributes.
        '''
        sounddata = pyst.paths.str2path(sounddata)
        if pyst.paths.is_audio_ext_allowed(sounddata):
            y, sr = pyst.loadsound(sounddata,
                                    sr=self.sr,
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

    def get_feats(self, list_waves, dur_sec=None, seed=None):
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
        if seed is not None:
            np.random.seed(seed)
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
                 sr=48000):
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
                 sr=48000):
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
    sr = pyst.utils.make_number(feature_settings['sr'])
    window_size = pyst.utils.make_number(feature_settings['window_size'])
    window_shift = pyst.utils.make_number(feature_settings['window_shift'])
    feature_sets = pyst.utils.make_number(feature_settings['feature_sets'])
    feature_type = feature_settings['feature_type']
    num_columns = pyst.utils.make_number(feature_settings['num_columns'])
    num_images_per_audiofile = pyst.utils.make_number(
        feature_settings['num_images_per_audiofile'])
    training_segment_ms = pyst.utils.make_number(
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
                               sr=sr,
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
def plot(feature_matrix, feature_type, 
                    save_pic=False, name4pic=None, scale=None,
                    title=None, sr=None, win_size_ms=None, percent_overlap=None,
                    x_label=None, y_label=None):
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
    sr : int, optional
        Useful for plotting a signal type feature matrix. Allows x-axis to be
        presented as time in seconds.
    '''
    # ensure real numbers
    if isinstance(feature_matrix[0], np.complex_):
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
        if sr is not None:
            x_axis_label = 'Time (sec)'
            dur_sec = feature_matrix.shape[0] / sr
            time_sec = pyst.dsp.get_time_points(dur_sec, sr)
            for channel in range(feature_matrix.shape[1]):
                data = feature_matrix[:,channel]
                # overlay the channel data
                plt.plot(time_sec, data)
        else:
            for channel in range(feature_matrix.shape[1]):
                data = feature_matrix[:,channel]
                # overlay the channel data
                plt.plot(data)
                ##display.waveplot(data,sr=sr)
        x_axis_label += ' across {} channel(s)'.format(channel+1)
    else:
        plt.pcolormesh(feature_matrix.T)
        ## display.specshow(feature_matrix.T, sr=sr)
        plt.colorbar(label=energy_label)
    plt.xlabel(x_axis_label)
    plt.ylabel(axis_feature_label)
    # if feature_matrix has multiple frames, not just one
    if feature_matrix.shape[1] > 1: 
        if 'signal' not in feature_type \
            and win_size_ms is not None and percent_overlap is not None:
            # the xticks basically show time but need to be multiplied by 0.01
            plt.xlabel('Time (sec)') 
            locs, labels = plt.xticks()
            new_labels=[str(round(i*0.001*win_size_ms*percent_overlap,1)) for i in locs]
            plt.xticks(ticks=locs,labels=new_labels)
        else:
            plt.xlabel('Number frames')
        plt.ylabel('Frequency bins')
    if title is None:
        plt.title('{} Features'.format(feature_type.upper()))
    else:
        plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if save_pic:
        outputname = name4pic or 'visualize{}feats'.format(feature_type.upper())
        plt.savefig('{}.png'.format(outputname))
    else:
        plt.show()

# TODO test duration limit on all settings
def get_feats(sound,
              sr=None, 
              features='fbank', 
              win_size_ms = 20, 
              percent_overlap = 0.5,
              window = 'hann',
              num_filters=40,
              num_mfcc=None, 
              duration=None,
              mono=None):
    '''Collects raw signal data, stft, fbank, or mfcc features via librosa.
    
    Parameters
    ----------
    sound : str or numpy.ndarray [size=(num_samples,) or (num_samples, num_channels)]
        If str, wavfile (must be compatible with scipy.io.wavfile). Otherwise 
        the samples of the sound data. Note: in the latter case, `sr`
        must be declared.
    sr : int, optional
        The sample rate of the sound data or the desired sample rate of
        the wavfile to be loaded. (default None)
    features : str
        Options include 'signal', 'stft', 'fbank', or 'mfcc' data (default 'fbank').
        signal: energy/amplitude measurements along time
        STFT: short-time fourier transform
        FBANK: mel-log filterbank energies 
        MFCC: mel frequency cepstral coefficients 
    win_size_ms : int or float
        Window length in milliseconds for Fourier transform to be applied
        (default 20)
    percent_overlap : int or float 
        Amount of overlap between processing windows. For example, if `percent_overlap`
        is set at 0.5, the overlap will be half that of `win_size_ms`. (default 0.5) 
        If an integer is provided, it will be converted to a float between 0 and 1. 
    window : str or np.ndarray [size (n_fft, )]
        The window function to be applied to each window. (Default 'hann')
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
    duration : float, optional
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
    # load data
    if isinstance(sound, str) or isinstance(sound, pathlib.PosixPath):
        if mono is None:
            mono = True
        data, sr = librosa.load(sound, sr=sr, duration=duration, mono=mono)
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
        if sr is None:
            raise ValueError('No samplerate given. Either provide '+\
                'filename or appropriate samplerate.')
        data, sr = sound, sr
        if duration:
            data = data[:int(sr*duration)]
    # ensure percent overlap is between 0 and 1
    percent_overlap = check_percent_overlap(percent_overlap)
    win_shift_ms = win_size_ms * percent_overlap
    try:
        if 'fbank' in features:
            feats = librosa.feature.melspectrogram(
                data,
                sr = sr,
                n_fft = int(win_size_ms * sr // 1000),
                hop_length = int(win_shift_ms*0.001*sr),
                n_mels = num_filters, window=window).T
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
                n_mels = num_filters,
                window=window).T
            #feats -= (np.mean(feats, axis=0) + 1e-8)
        elif 'stft' in features:
            feats = librosa.stft(
                data,
                n_fft = int(win_size_ms * sr // 1000),
                hop_length = int(win_shift_ms*0.001*sr),
                window=window).T
            #feats -= (np.mean(feats, axis=0) + 1e-8)
        elif 'signal' in features:
            feats = (data, sr)
    except librosa.ParameterError as e:
        # potential error in handling fortran array
        feats = get_feats(np.asfortranarray(data),
                          features = features,
                          win_size_ms = win_size_ms,
                          percent_overlap = percent_overlap,
                          num_filters = num_filters,
                          num_mfcc = num_mfcc,
                          sr = sr,
                          duration = duration,
                          mono = mono)
    return feats

# TODO possibly remove?
def get_mfcc_fbank(samples, feature_type='mfcc', sr=48000, win_size_ms=20,
                     percent_overlap=0.5, num_filters=40, num_mfcc=40,
                     window_function=None):
    '''Collects fbank or mfcc features via python speech features.
    '''
    if not window_function:
        # default for python_speech_features:
        def window_function(x): return np.ones((x,))
    else:
        if 'hamming' in window_function:
            window_function = hamming
        elif 'hann' in window_function:
            window_function = hann
        else:
            # default for python_speech_features:
            def window_function(x): return np.ones((x,))
    if len(samples)/sr*1000 < win_size_ms:
        win_size_ms = len(samples)/sr*1000
    frame_length = pyst.dsp.calc_frame_length(win_size_ms, sr)
    percent_overlap = check_percent_overlap(percent_overlap)
    window_shift_ms = win_size_ms * percent_overlap
    if 'fbank' in feature_type:
        feats = logfbank(samples,
                         sr=sr,
                         winlen=win_size_ms * 0.001,
                         winstep=window_shift_ms * 0.001,
                         nfilt=num_filters,
                         nfft=frame_length,
                         winfunc=window_function)
    elif 'mfcc' in feature_type:
        feats = mfcc(samples,
                     sr=sr,
                     winlen=win_size_ms * 0.001,
                     winstep=window_shift_ms * 0.001,
                     nfilt=num_filters,
                     numcep=num_mfcc,
                     nfft=frame_length,
                     winfunc=window_function)
    return feats, frame_length, win_size_ms


def plotsound(audiodata, feature_type='fbank', win_size_ms = 20, \
    percent_overlap = 0.5, num_filters=40, num_mfcc=40, sr=None,\
        save_pic=False, name4pic=None, power_scale=None, mono=None):
    '''Visualize feature extraction depending on set parameters. Does not use Librosa.
    
    Parameters
    ----------
    audiodata : str or numpy.ndarray
        If str, wavfile (must be compatible with scipy.io.wavfile). Otherwise 
        the samples of the sound data. Note: in the latter case, `sr`
        must be declared.
    feature_type : str
        Options: 'signal', 'mfcc', or 'fbank' features. 
        MFCC: mel frequency cepstral
        coefficients; FBANK: mel-log filterbank energies (default 'fbank')
    win_size_ms : int or float
        Window length in milliseconds for Fourier transform to be applied
        (default 20)
    percent_overlap : int or float 
        Amount of overlap between processing windows. For example, if `percent_overlap`
        is set at 0.5, the overlap will be half that of `win_size_ms`. (default 0.5) 
        If an integer is provided, it will be converted to a float between 0 and 1.
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
    sr : int, optional
        The sample rate of the sound data or the desired sample rate of
        the wavfile to be loaded. (default None)
    mono : bool, optional
        When loading an audiofile, True will limit number of channels to
        one; False will allow more channels to be loaded. (default None, 
        which results in mono channel loading.)
    '''
    percent_overlap = check_percent_overlap(percent_overlap)
    win_shift_ms = win_size_ms * percent_overlap
    feats = pyst.feats.get_feats(audiodata, features=feature_type, 
                      win_size_ms = win_size_ms, win_shift_ms = win_shift_ms,
                      num_filters=num_filters, num_mfcc = num_mfcc, sr=sr,
                      mono=mono)
    if 'signal' in feature_type:
        feats, sr = feats
    else:
        sr = None
    pyst.feats.plot(feats, feature_type=feature_type, sr=sr,
                    save_pic = save_pic, name4pic=name4pic, scale=power_scale)

def check_percent_overlap(percent_overlap):
    if percent_overlap > 1:
        percent_overlap *= 0.01
        if percent_overlap > 1:
            raise ValueError('The percent overlap value '+str(percent_overlap)+\
                ' is too large. Please use a value between 0 and 1 or 0 and 100.')
    return percent_overlap


def separate_dependent_var(matrix):
    '''Separates matrix into features and labels. Expects 3D array.

    Assumes the last column of the last dimension of the matrix constitutes
    the dependent variable (labels), and all other columns the indpendent variables
    (features). Additionally, it is assumed that for each block of data, 
    only one label is needed; therefore, just the first label is taken for 
    each block.

    Parameters
    ----------
    matrix : numpy.ndarray [size = (num_samples, num_frames, num_features)]
        The `matrix` holds the numerical data to separate. num_features is
        expected to be at least 2.

    Returns
    -------
    X : numpy.ndarray [size = (num_samples, num_frames, num_features -1)]
        A matrix holding the (assumed) independent variables
    y : numpy.ndarray, numpy.int64, numpy.float64 [size = (num_samples,)]
        A vector holding the labels assigned to the independent variables.
        If only one value in array, just the value inside is returned

    Examples
    --------
    >>> import numpy as np
    >>> #vector
    >>> separate_dependent_var(np.array([1,2,3,4]))
    (array([1, 2, 3]), 4)
    >>> #simple matrix
    >>> matrix = np.arange(4).reshape(2,2)
    >>> matrix
    array([[0, 1],
           [2, 3]])
    >>> X, y = separate_dependent_var(matrix)
    >>> X
    array([[0],
           [2]])
    >>> y 
    1
    >>> #more complex matrix
    >>> matrix = np.arange(20).reshape((2,2,5))
    >>> matrix
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9]],
    <BLANKLINE>
           [[10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19]]])
    >>> X, y = separate_dependent_var(matrix)
    >>> X
    array([[[ 0,  1,  2,  3],
            [ 5,  6,  7,  8]],
    <BLANKLINE>
           [[10, 11, 12, 13],
            [15, 16, 17, 18]]])
    >>> y
    array([ 4, 14])
    '''
    # get last column
    if matrix.shape[-1] == 1:
        raise ValueError('Expects input matrix to be size (num_samples, num_frames, ' + \
                         'num_features). Number of features must exceed 1 in order ' + \
                         'to separate into X and y arrays.')
    y_step1 = np.take(matrix, -1, axis=-1)
    # because the label is the same for each block of data, just need the first
    # row,  not all the rows, as they are the same label.
    y = np.take(y_step1, 0, axis=-1)
    # get features:
    X = np.delete(matrix, -1, axis=-1)
    return X, y

# TODO: perhaps remove - just use np.expand_dims() 
# TODO: https://github.com/biopython/biopython/issues/1496
# Fix numpy array repr for Doctest. 
def add_tensor(matrix):
    '''Adds tensor / dimension to input ndarray (e.g. features).

    Keras requires an extra dimension at some layers, which represents 
    the 'tensor' encapsulating the data. 

    Further clarification taking the example below. The input matrix has 
    shape (2,3,4). Think of it as 2 different events, each having
    3 sets of measurements, with each of those having 4 features. So, 
    let's measure differences between 2 cities at 3 different times of
    day. Let's take measurements at 08:00, 14:00, and 19:00 in... 
    Magic City and Never-ever Town. We'll measure.. 1) tempurature, 
    2) wind speed 3) light level 4) noise level.

    How I best understand it, putting our measurements into a matrix
    with an added dimension/tensor, this highlights the separate 
    measurements, telling the algorithm: yes, these are 4 features
    from the same city, BUT they occur at different times. Or it's 
    just how Keras set up the code :P 

    Parameters
    ----------
    matrix : numpy.ndarray
        The `matrix` holds the numerical data to add a dimension to.

    Returns
    -------
    matrix : numpy.ndarray
        The `matrix` with an additional dimension.

    Examples
    --------
    >>> import numpy as np
    >>> matrix = np.arange(24).reshape((2,3,4))
    >>> matrix.shape
    (2, 3, 4)
    >>> matrix
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    <BLANKLINE>
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
    >>> matrix_2 = add_tensor(matrix)
    >>> matrix_2.shape
    (2, 3, 4, 1)
    >>> matrix_2
    array([[[[ 0],
             [ 1],
             [ 2],
             [ 3]],
    <BLANKLINE>
            [[ 4],
             [ 5],
             [ 6],
             [ 7]],
    <BLANKLINE>
            [[ 8],
             [ 9],
             [10],
             [11]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[12],
             [13],
             [14],
             [15]],
    <BLANKLINE>
            [[16],
             [17],
             [18],
             [19]],
    <BLANKLINE>
            [[20],
             [21],
             [22],
             [23]]]])
    '''
    if isinstance(matrix, np.ndarray) and len(matrix) > 0:
        matrix = matrix.reshape(matrix.shape + (1,))
        return matrix
    elif isinstance(matrix, np.ndarray):
        raise ValueError('Input matrix is empty.')
    else:
        raise TypeError('Expected type numpy.ndarray, recieved {}'.format(
            type(matrix)))
    
# TODO improve / remove.. move to data module?
def scale_X_y(matrix, is_train=True, scalars=None):
    '''Separates and scales data into X and y arrays. Adds dimension for keras.
    
    Assumes the last column of the last dimension is the y or label data.
    
    Parameters
    ----------
    matrix : np.ndarray [size = (num_samples, num_frames, num_features)]
        Matrix with X and y data
    is_train : bool
        Relevant for the `scalars` parameter. If the data is training
        data (i.e. True), the `scalars` will be created. If the data
        is test data (i.e. False), the function expects `scalars` to 
        be provided. (default True)
    scalars : dict, optional
        Dictionary with scalars to be applied to non-training data.
        
    Returns
    -------
    X : np.ndarray [size = (num_sampls, num_frames, num_features-1, 1)]
        Scaled features with extra dimension
    y : np.ndarray [size = (num_samples, 1, 1)]
        Scaled independent variable with extra dimension
    scalars : dict
        The scalars either created or previously loaded.
    '''
    X, y = pyst.feats.separate_dependent_var(matrix)
    if is_train:
        scalars = {}
    elif scalars is None:
        raise TypeError('If non-train data, `scalars` cannot be of type None.')
    if len(X.shape) != 3:
        raise ValueError('Expected 3d input, not input of shape {}.'.format(
            matrix.shape))
    if X.dtype == np.complex_:
        # convert stft to power spectrum
        print('\nTaking absolute value and power of complex data..'+\
            '\ni.e. Removing complex values.')
        X = np.abs(X)**2
    for j in range(X.shape[2]):
        if is_train:
            scalars[j] = StandardScaler()
            X[:, :, j] = scalars[j].fit_transform(X[:, :, j])
        else:
            X[:, :, j] = scalars[j].transform(X[:, :, j])
        X[:, :, j] = normalize(X[:, :, j])
    # Keras needs an extra dimension as a tensor / holder of data
    X = pyst.feats.add_tensor(X)
    y = pyst.feats.add_tensor(y)
    return X, y, scalars

# TODO move to data module??
def normalize(data):
    # need to take power - complex values in stft
    if data.dtype == np.complex_:
        # take power of absoulte value of stft
        data = np.abs(data)**2
    data = (data - np.min(data)) / \
        (np.max(data) - np.min(data))
    return data

# TODO test for all these features:
def list_available_features():
    return ['stft', 'powspec', 'fbank', 'mfcc', 'signal']

# TODO apply keyword arguments
def save_features_datasets(datasets_dict, datasets_path2save_dict, dur_sec,
                                    feature_type='fbank', num_feats=None, sr=22050, 
                                    win_size_ms=20, percent_overlap=0.5, n_fft = None,
                                    window='hann',frames_per_sample=None,labeled_data=False, 
                                    subsection_data=False, divide_factor=5,
                                    visualize=False, vis_every_n_frames=50, 
                                    use_librosa=True, center=True, mode='reflect', 
                                    log_settings=True):
    '''Extracts and saves audio features, sectioned into datasets, to indicated locations.
    
    If MemoryError, the provided dataset dicts will be adjusted to allow data to be subsectioned.
    
    Parameters
    ----------
    datasets_dict : dict 
        Dictionary with keys representing datasets and values the audifiles making up that dataset.
        E.g. {'train':['1.wav', '2.wav', '3.wav'], 'val': ['4.wav'], 'test':['5.wav']} for unlabled
        data or  {'train':[(0, '1.wav'), (1, '2.wav'), (0, '3.wav')], 'val': [(1, '4.wav')], 
        'test':[(0, '5.wav')]} for labeled data.
    datasets_path2save_dict : dict
        Dictionary with keys representing datasets and values the pathways of where extracted 
        features of that dataset will be saved.
        E.g. {'train': './data/train.npy', 'val': './data/val.npy', 'test': './data/test.npy'}
    feature_type : str 
        String including only one of the following: 'stft', 'powspec', 'fbank', and 'mfcc'.
        'powspec' and 'stft' are basically the same; 'powspec' is the 'stft' except without 
        complex values and squared. E.g 'mfcc_noisy' or 'stft_train'.
    sr : int 
        The sample rate the audio data should be loaded with.
    n_fft : int 
        The number of frequency bins used for the Fast Fourier Transform (fft)
    dur_sec : int or float
        The desired duration of how long the audio data should be. This is used to calculate 
        size of feature data and is therefore necessary, as audiofiles tend to differe in length.
        If audiofiles are longer or shorter, they will be cut or zeropadded respectively.
    num_feats : int 
        The number of mfcc coefficients (mfcc), mel filters (fbank), or frequency bins (stft).
    win_size_ms : int 
        The desired window size in milliseconds to process audio samples.
    percent_overlap : float
        The amount audio samples should overlap as each window is processed.
    window : str 
        The window applied to processing audio data, e.g. 'hann' or 'hamming' (default 'hann')
    frames_per_sample : int, optional 
        If you want to section each audio file feature data into smaller frames. This might be 
        useful for speech related contexts. (Can avoid this by simply reshaping data later)
    labeled_data : bool 
        If True, expects each audiofile to be accompanied by an integer label. See example 
        given for `datasets_dict`.
    subsection_data : bool 
        If you have a large dataset, you may want to divide it into subsections. See 
        pysoundtool.data.subsection_data. If datasets are large enough to raise a MemoryError, 
        this will be applied automatically.
    divide_factor : int 
        The number of subsections to divide data into. Only large enough sections will be divided.
        If smaller datasets (i.e. validation and test datasets) are as large or smaller than 
        the new subsectioned larger dataset(s) (i.e. train), they will be left unchanged.
    visualize : bool
        If True, periodic plots of the features will be saved throughout the extraction process. (default False)
    vis_every_n_frames : int 
        How often visuals should be made: every 10 samples, every 100, etc. (default 50)
    use_librosa : bool 
        If True, librosa is used to load and extract features. As of now, no other option is 
        available. TODO: add other options. :P I just wanted to be clear that some elements
        of this function are unique to using librosa. (default True)
    center : bool 
        Relevant for librosa and feature extraction. (default True)
    mode : str 
        Relevant for librosa and feature extraction. (default 'reflect')
    log_settings : bool
        If True, a .csv file will be saved in the feature extraction directory with 
        most of the feature settings saved. (default True)
    
    Returns
    -------
    datasets_dict : dict 
        The final dataset dictionary used in feature extraction. The datasets may 
        have been subdivided.
    datasets_path2save_dict : dict
        The final dataset feature pathway dict. The pathways will have been 
        adjusted if the datasets have been subdivided.
    '''
    # if dataset is large, may want to divide it into sections
    if subsection_data:
        datasets_dict, datasets_path2save_dict = pyst.data.section_data(
            datasets_dict,
            datasets_path2save_dict,
            divide_factor=divide_factor)
    try:
        # depending on which packages one uses, shape of data changes.
        # for example, Librosa centers/zeropads data automatically
        # TODO see which shapes result from python_speech_features
        total_samples = pyst.dsp.calc_frame_length(dur_sec*1000, sr=sr)
        # if using Librosa:
        if use_librosa:
            frame_length = pyst.dsp.calc_frame_length(win_size_ms, sr)
            hop_length = int(win_size_ms*percent_overlap*0.001*sr)
            if n_fft is None:
                n_fft = frame_length
            # librosa centers samples by default, sligthly adjusting total 
            # number of samples
            if center:
                y_zeros = np.zeros((total_samples,))
                y_centered = np.pad(y_zeros, int(n_fft // 2), mode=mode)
                total_samples = len(y_centered)
            # each audio file 
            total_rows_per_wav = int(1 + (total_samples - n_fft)//hop_length)
            
            # set defaults to num_feats if set as None:
            if num_feats is None:
                if 'mfcc' in feature_type or 'fbank' in feature_type:
                    num_feats = 40
                elif 'powspec' in feature_type or 'stft' in feature_type:
                    num_feats = int(1+n_fft/2)
                else:
                    raise ValueError('Feature type "{}" '.format(feature_type)+\
                        'not understood.\nMust include one of the following: \n'+\
                            ', '.join(list_available_features()))
            
            # adjust shape for model
            if frames_per_sample is not None:
                # want smaller windows, e.g. autoencoder denoiser
                batch_size = math.ceil(total_rows_per_wav/frames_per_sample)
                input_shape = (batch_size, frames_per_sample, num_feats)
                # if extracted data is too short, shape to zeropad:
                desired_shape = (input_shape[0]*input_shape[1],
                                        input_shape[2])
            else:
                if labeled_data:
                    input_shape = (int(total_rows_per_wav), num_feats + 1)
                    desired_shape = (input_shape[0], input_shape[1]-1)
                else:
                    input_shape = (int(total_rows_per_wav), num_feats)
                    desired_shape = input_shape
            # set whether or not features will include complex values:
            if 'stft' in feature_type:
                complex_vals = True
            else:
                complex_vals = False
            if 'mfcc' in feature_type:
                feat_type = 'mfcc'
            elif 'fbank' in feature_type:
                feat_type = 'fbank'
            elif 'stft' in feature_type:
                feat_type = 'stft'
            elif 'powspec' in feature_type:
                feat_type = 'stft'
            else:
                raise TypeError('Expected '+', '.join(list_available_features())+\
                    ' to be in `feature_type`, not {}'.format(feature_type))
            for key, value in datasets_dict.items():
                # get parent directory of where data should be saved (i.e. for saving pics)
                datapath = datasets_path2save_dict[key]
                if not isinstance(datapath, pathlib.PosixPath):
                    datapath = pathlib.Path(datapath)
                datadir = datapath.parent
                # when loading a dictionary, the value is a string
                if isinstance(value, str):
                    value = pyst.utils.string2list(value)
                extraction_shape = (len(value),) + input_shape
                feats_matrix = pyst.dsp.create_empty_matrix(
                    extraction_shape, 
                    complex_vals=complex_vals)
                for j, audiofile in enumerate(value):
                    if labeled_data:
                        label, audiofile = int(audiofile[0]), audiofile[1]
                    feats = pyst.feats.get_feats(audiofile,
                                                sr=sr,
                                                features=feat_type,
                                                win_size_ms=win_size_ms,
                                                percent_overlap=percent_overlap,
                                                window=window,
                                                num_filters=num_feats,
                                                num_mfcc=num_feats,
                                                duration=dur_sec)
                    # if power spectrum (remove complex values and squaring features)
                    if 'powspec' in feature_type:
                        feats = np.abs(feats)**2
                    # zeropad feats if too short:
                    feats = pyst.data.zeropad_features(
                        feats, 
                        desired_shape = desired_shape,
                        complex_vals = complex_vals)
                    
                    if labeled_data:
                        # create label column
                        label_col = np.zeros((len(feats),1)) + label
                        feats = np.concatenate([feats,label_col], axis=1)

                    if visualize:
                        # visualize features:
                        if 'mfcc' in feature_type:
                            scale = None
                        else:
                            scale = 'power_to_db'
                        #visualize features only every n num frames
                        if j % vis_every_n_frames == 0:
                            save_pic_path = datadir.joinpath(
                                'images',key,'{}_sample{}'.format(
                                    feature_type, j))
                            # make sure this directory exists
                            save_pic_dir = pyst.utils.check_dir(save_pic_path.parent, make=True)
                            pyst.feats.plot(feats, 
                                            feature_type = feature_type,
                                            win_size_ms = win_size_ms,
                                            percent_overlap = percent_overlap,
                                            scale=scale,
                                            title='{} {} features'.format(
                                                key, feature_type.upper()),
                                            save_pic=visualize, 
                                            name4pic=save_pic_path)
                    feats = feats.reshape(extraction_shape[1:])
                    # fill in empty matrix with features from each audiofile

                    feats_matrix[j] = feats
                    pyst.utils.print_progress(iteration = j, 
                                            total_iterations = len(value),
                                            task = '{} {} feature extraction'.format(
                                                key, feature_type))
                # save data:
                np.save(datasets_path2save_dict[key], feats_matrix)
                print('\nFeatures saved at {}\n'.format(datasets_path2save_dict[key]))
            if log_settings:
                log_filename = datadir.joinpath('log_extraction_settings.csv')
                feat_settings = dict(dur_sec=dur_sec,
                                     feature_type=feature_type,
                                     feat_type=feat_type,
                                     complex_vals=complex_vals,
                                     sr=sr,
                                     num_feats=num_feats,
                                     n_fft=n_fft,
                                     win_size_ms=win_size_ms,
                                     frame_length=frame_length,
                                     percent_overlap=percent_overlap,
                                     window=window,
                                     frames_per_sample=frames_per_sample,
                                     labeled_data=labeled_data,
                                     visualize=visualize,
                                     total_samples=total_samples,
                                     input_shape=input_shape,
                                     use_librosa=use_librosa,
                                     center=center,
                                     mode=mode,
                                     subsection_data=subsection_data,
                                     divide_factor=divide_factor,
                                     )
                feat_settings_path = pyst.utils.save_dict(feat_settings,
                                                          log_filename,
                                                          overwrite=True)
        else:
            raise ValueError('Sorry, this functionality is not yet supported. '+\
                'Set `use_librosa` to True.')
    except MemoryError as e:
        print('MemoryError: ',e)
        print('\nSectioning data and trying again.\n')
        datasets_dict, datasets_path2save_dict = pyst.data.section_data(
            datasets_dict, datasets_path2save_dict, divide_factor=divide_factor)
        datasets_dict, datasets_path2save_dict = save_features_datasets(
            datasets_dict = datasets_dict, 
            datasets_path2save_dict = datasets_path2save_dict,
            feature_type = feature_type, 
            sr = sr, 
            n_fft = n_fft, 
            dur_sec = dur_sec,
            num_feats = num_feats,
            win_size_ms = win_size_ms,
            percent_overlap = percent_overlap,
            use_librosa = use_librosa, 
            window = window,
            center = center,
            mode = mode,
            frames_per_sample = frames_per_sample,
            visualize = visualize, 
            vis_every_n_frames = vis_every_n_frames, 
            labeled_data = labeled_data,
            log_settings = log_settings)
    return datasets_dict, datasets_path2save_dict

if __name__ == "__main__":
    import doctest
    doctest.testmod()
