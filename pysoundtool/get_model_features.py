import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)

import time
import pysoundtool as pyst


def envclassifier_feats(
    data_dir = '../audiodata/minidatasets/background_noise/',
    data_features_dir = '../audiodata/features_scene_classifier/',
    feature_type = 'fbank',
    **kwargs):
    '''Environment Classifier: feature extraction of scene audio into train, val, & test datasets.
    
    Saves extracted feature datasets (train, val, test datasets) as well as 
    feature extraction settings in the directory `data_features_dir`.
    
    Parameters
    ----------
    data_dir : str or pathlib.PosixPath
        The directory with scene subfolders (e.g. 'air_conditioner', 'traffic') that 
        contain audio files belonging to that scene (e.g. 'air_conditioner/ac1.wav',
        'air_conditioner/ac2.wav', 'traffic/t1.wav')
        (default './audiodata/minidatasets/background_noise/')
    
    data_features_dir : str or pathlib.PosixPath
        The directory where feature extraction related to the dataset will be stored. 
        Within this directory, a unique subfolder will be created each time features are
        extracted. This allows several versions of extracted features on the same dataset
        without overwriting files.
        (default './audiodata/features_scene_classifier/')
    
    feature_type : str 
        The type of features to be extracted. Options: 'stft', 'powspec', 'mfcc', 'fbank'. (default 'fbank')
    
    kwargs : additional keyword arguments
        Keyword arguments for `pysoundtool.feats.save_features_datasets` and 
        `pysoundtool.feats.get_feats`.
        
    Returns
    -------
    feat_extraction_dir : pathlib.PosixPath
        The pathway to where all feature extraction files can be found, including datasets.
        
    See Also
    --------
    pysoundtool.feats.get_feats
        Extract features from audio file or audio data.
        
    pysoundtool.feats.save_features_datasets
        Preparation of acoustic features in train, validation and test datasets.
    '''
    if data_dir is None:
        data_dir = './audiodata/minidatasets/background_noise/'
    if data_features_dir is None:
        data_features_dir = './audiodata/features_scene_classifier/'
    if feature_type is None:
        feature_type = 'fbank'
    if 'signal' in feature_type:
        raise ValueError('Feature type "signal" is not yet supported for CNN training.')

    feat_extraction_dir = 'features_'+feature_type + '_' + pyst.utils.get_date()

    # collect labels 
    labels = []
    data_dir = pyst.utils.string2pathlib(data_dir)
    for label in data_dir.glob('*/'):
        labels.append(label.stem)
    labels = set(labels)

    # create paths for what we need to save:
    data_features_dir = pyst.utils.check_dir(data_features_dir)
    feat_extraction_dir = data_features_dir.joinpath(feat_extraction_dir)
    feat_extraction_dir = pyst.utils.check_dir(feat_extraction_dir, make=True)

    # dictionaries containing encoding and decoding labels:
    dict_encode_path = feat_extraction_dir.joinpath('dict_encode.csv')
    dict_decode_path = feat_extraction_dir.joinpath('dict_decode.csv')
    # dictionary for which audio paths are assigned to which labels:
    dict_encdodedlabel2audio_path = feat_extraction_dir.joinpath('dict_encdodedlabel2audio.csv')

    # designate where to save train, val, and test data
    data_train_path = feat_extraction_dir.joinpath('{}_data_{}.npy'.format('train',
                                                                        feature_type))
    data_val_path = feat_extraction_dir.joinpath('{}_data_{}.npy'.format('val',
                                                                        feature_type))
    data_test_path = feat_extraction_dir.joinpath('{}_data_{}.npy'.format('test',
                                                                        feature_type))

    # create and save encoding/decoding labels dicts
    dict_encode, dict_decode = pyst.data.create_dicts_labelsencoded(labels)
    dict_encode_path = pyst.utils.save_dict(dict_encode, 
                                        filename = dict_encode_path,
                                        overwrite=False)
    dict_decode_path = pyst.utils.save_dict(dict_encode, 
                                        filename = dict_decode_path,
                                        overwrite=False)

    # save audio paths to each label in dict 
    paths_list = pyst.utils.collect_audiofiles(data_dir, recursive=True)
    paths_list = sorted(paths_list)

    dict_encodedlabel2audio = pyst.data.create_encodedlabel2audio_dict(dict_encode,
                                                        paths_list)
    dict_encdodedlabel2audio_path = pyst.utils.save_dict(dict_encodedlabel2audio, 
                                            filename = dict_encdodedlabel2audio_path, overwrite=False)
    # assign audiofiles into train, validation, and test datasets
    train, val, test = pyst.data.audio2datasets(dict_encdodedlabel2audio_path,
                                                perc_train=0.8,
                                                limit=None,
                                                seed=40)

    # save audiofiles for each dataset to dict and save
    dataset_dict = dict([('train',train),('val', val),('test',test)])
    dataset_dict_path = feat_extraction_dir.joinpath('dataset_audiofiles.csv')
    dataset_dict_path = pyst.utils.save_dict(dataset_dict, dataset_dict_path, 
                                            overwrite=True)
    # save paths to where extracted features of each dataset will be saved to dict w same keys
    datasets_path2save_dict = dict([('train',data_train_path),
                                    ('val', data_val_path),
                                    ('test',data_test_path)])

    # extract features

    start = time.time()

    dataset_dict, datasets_path2save_dict = pyst.feats.save_features_datasets(
        datasets_dict = dataset_dict,
        datasets_path2save_dict = datasets_path2save_dict,
        labeled_data = True,
        feature_type = feature_type,
        **kwargs)

    end = time.time()
    
    total_dur_sec = end-start
    total_dur, units = pyst.utils.adjust_time_units(total_dur_sec)
    print('\nFinished! Total duration: {} {}.'.format(round(total_dur,2), units))

    # save which audiofiles were extracted for each dataset
    # save where extracted data were saved
    # save how long feature extraction took
    dataprep_settings = dict(dataset_dict=dataset_dict,
                            datasets_path2save_dict=datasets_path2save_dict,
                            total_dur_sec = total_dur_sec)
    dataprep_settings_path = pyst.utils.save_dict(
        dataprep_settings,
        feat_extraction_dir.joinpath('dataset_audio_assignments.csv'))
    
    return feat_extraction_dir


def denoiser_feats(
    data_clean_dir = '../audiodata/minidatasets/denoise/cleanspeech_IEEE_small/',
    data_noisy_dir = '../audiodata/minidatasets/denoise/noisyspeech_IEEE_small/',
    data_features_dir = '../audiodata/features_denoiser/',
    feature_type = 'fbank',
    **kwargs):
    '''Autoencoder Denoiser: feature extraction of clean & noisy audio into train, val, & test datasets.
    
    Saves extracted feature datasets (train, val, test datasets) as well as 
    feature extraction settings in the directory `data_features_dir`.
    
    Parameters
    ----------
    data_clean_dir : str or pathlib.PosixPath
        The directory with clean audio files. If no directory is given, the sample data
        available with PySoundTool will be used.
        (default './audiodata/minidatasets/denoise/cleanspeech_IEEE_small/')
    
    data_noisy_dir : str or pathlib.PosixPath
        The directory with noisy audio files. These should be the same as the clean audio,
        except noise has been added. If no directory is given, the sample data available 
        with PySoundTool will be used.
        (default './audiodata/minidatasets/denoise/noisyspeech_IEEE_small/')
    
    data_features_dir : str or pathlib.PosixPath
        The directory where feature extraction related to the dataset will be stored. 
        Within this directory, a unique subfolder will be created each time features are
        extracted. This allows several versions of extracted features on the same dataset
        without overwriting files.
        (default './audiodata/features_denoiser/')
    
    feature_type : str 
        The type of features to be extracted. Options: 'stft', 'powspec', 'mfcc', 'fbank' or
        'signal'. (default 'fbank')
    
    kwargs : additional keyword arguments
        Keyword arguments for `pysoundtool.feats.save_features_datasets` and 
        `pysoundtool.feats.get_feats`.
        
    Returns
    -------
    feat_extraction_dir : pathlib.PosixPath
        The pathway to where all feature extraction files can be found, including datasets.
        
    See Also
    --------
    pysoundtool.feats.get_feats
        Extract features from audio file or audio data.
        
    pysoundtool.feats.save_features_datasets
        Preparation of acoustic features in train, validation and test datasets.
    '''
    # create unique directory for feature extraction session:
    feat_extraction_dir = 'features_'+feature_type + '_' + pyst.utils.get_date()

    # 1) Ensure clean and noisy data directories exist
    audio_clean_path = pyst.utils.check_dir(data_clean_dir, make=False)
    audio_noisy_path = pyst.utils.check_dir(data_noisy_dir, make=False)

    # 2) create paths for what we need to save:
    denoise_data_path = pyst.utils.check_dir(data_features_dir, make=True)
    feat_extraction_dir = denoise_data_path.joinpath(feat_extraction_dir)
    feat_extraction_dir = pyst.utils.check_dir(feat_extraction_dir, make=True)
    # Noisy and clean train, val, and test data paths:
    data_train_noisy_path = feat_extraction_dir.joinpath('{}_data_{}_{}.npy'.format('train',
                                                                            'noisy',
                                                                        feature_type))
    data_val_noisy_path = feat_extraction_dir.joinpath('{}_data_{}_{}.npy'.format('val',
                                                                        'noisy',
                                                                        feature_type))
    data_test_noisy_path = feat_extraction_dir.joinpath('{}_data_{}_{}.npy'.format('test',
                                                                        'noisy',
                                                                        feature_type))
    data_train_clean_path = feat_extraction_dir.joinpath('{}_data_{}_{}.npy'.format('train',
                                                                            'clean',
                                                                        feature_type))
    data_val_clean_path = feat_extraction_dir.joinpath('{}_data_{}_{}.npy'.format('val',
                                                                        'clean',
                                                                        feature_type))
    data_test_clean_path = feat_extraction_dir.joinpath('{}_data_{}_{}.npy'.format('test',
                                                                        'clean',
                                                                        feature_type))

    # 3) collect audiofiles and divide them into train, val, and test datasets
    # noisy data
    noisyaudio = pyst.utils.collect_audiofiles(audio_noisy_path, 
                                                    hidden_files = False,
                                                    wav_only = False,
                                                    recursive = False)
    # sort audio (can compare if noisy and clean datasets are compatible)
    noisyaudio = sorted(noisyaudio)

    # clean data
    cleanaudio = pyst.utils.collect_audiofiles(audio_clean_path, 
                                                    hidden_files = False,
                                                    wav_only = False,
                                                    recursive = False)
    cleanaudio = sorted(cleanaudio)


    # check if they match up: (expects clean file name to be in noisy file name)
    for i, audiofile in enumerate(noisyaudio):
        if not pyst.utils.check_noisy_clean_match(audiofile, cleanaudio[i]):
            raise ValueError('The noisy and clean audio datasets do not appear to match.')

    # save collected audiofiles for noisy and clean datasets to dictionary
    noisy_audio_dict = dict([('noisy', noisyaudio)])
    clean_audio_dict = dict([('clean', cleanaudio)])
    
    noisy_audio_dict_path = feat_extraction_dir.joinpath('noisy_audio.csv')
    noisy_audio_dict_path = pyst.utils.save_dict(noisy_audio_dict, noisy_audio_dict_path,
                                                overwrite=False)
    clean_audio_dict_path = feat_extraction_dir.joinpath('clean_audio.csv')
    clean_audio_dict_path = pyst.utils.save_dict(clean_audio_dict, clean_audio_dict_path,
                                                overwrite=False)
    # separate into datasets
    train_noisy, val_noisy, test_noisy = pyst.data.audio2datasets(noisy_audio_dict_path,
                                                                seed=40)
    train_clean, val_clean, test_clean = pyst.data.audio2datasets(clean_audio_dict_path,
                                                                seed=40)

    # save train, val, test dataset assignments to dict
    dataset_dict_noisy = dict([('train', train_noisy),('val', val_noisy),('test', test_noisy)])
    dataset_dict_clean = dict([('train', train_clean),('val', val_clean),('test', test_clean)])

    # keep track of paths to save data
    dataset_paths_noisy_dict = dict([('train',data_train_noisy_path),
                                    ('val', data_val_noisy_path),
                                    ('test',data_test_noisy_path)])
    dataset_paths_clean_dict = dict([('train',data_train_clean_path),
                                    ('val', data_val_clean_path),
                                    ('test',data_test_clean_path)])

    path2_noisy_datasets = feat_extraction_dir.joinpath('audiofiles_datasets_noisy.csv')
    path2_clean_datasets = feat_extraction_dir.joinpath('audiofiles_datasets_clean.csv')

    # save dicts to .csv files
    path2_noisy_datasets = pyst.utils.save_dict(dataset_dict_noisy,
                                                    path2_noisy_datasets,
                                                    overwrite=False)
    path2_clean_datasets = pyst.utils.save_dict(dataset_dict_clean,
                                                    path2_clean_datasets,
                                                    overwrite=False)

    # 5) extract features
            
    # ensure the noisy and clean values match up:
    for key, value in dataset_dict_noisy.items():
        for j, audiofile in enumerate(value):
            if not pyst.utils.check_noisy_clean_match(audiofile,
                                                    dataset_dict_clean[key][j]):
                raise ValueError('There is a mismatch between noisy and clean audio. '+\
                    '\nThe noisy file:\n{}'.format(dataset_dict_noisy[key][i])+\
                        '\ndoes not seem to match the clean file:\n{}'.format(audiofile))

    start = time.time()

    # first clean data
    dataset_dict_clean, dataset_paths_clean_dict = pyst.feats.save_features_datasets(
        datasets_dict = dataset_dict_clean,
        datasets_path2save_dict = dataset_paths_clean_dict,
        feature_type = feature_type + ' clean',
        **kwargs)
        
    # then noisy data
    dataset_dict_noisy, dataset_paths_noisy_dict = pyst.feats.save_features_datasets(
        datasets_dict = dataset_dict_noisy,
        datasets_path2save_dict = dataset_paths_noisy_dict,
        feature_type = feature_type + ' noisy',
        **kwargs)

    end = time.time()

    total_dur_sec = round(end-start,2)
    total_dur, units = pyst.utils.adjust_time_units(total_dur_sec)
    print('\nFinished! Total duration: {} {}.'.format(total_dur, units))
    # save which audiofiles were extracted for each dataset
    # save where extracted data were saved
    # save total duration of feature extraction
    dataprep_settings = dict(dataset_dict_noisy = dataset_dict_noisy,
                            dataset_paths_noisy_dict = dataset_paths_noisy_dict,
                            dataset_dict_clean = dataset_dict_clean,
                            dataset_paths_clean_dict = dataset_paths_clean_dict,
                            total_dur_sec = total_dur_sec)
    dataprep_settings_path = pyst.utils.save_dict(
        dataprep_settings,
        feat_extraction_dir.joinpath('dataset_audio_assignments.csv'))
    return feat_extraction_dir
