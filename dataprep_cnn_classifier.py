
import time
import pysoundtool as pyst


def dataprep_sceneclassifier(
    data_dir = './audiodata/minidatasets/background_noise/',
    data_features_dir = './audiodata/features_scene_classifier/',
    feature_type = 'fbank',
    **kwargs):
    '''Extract features from scene dataset into train, val, and test datasets.
    
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

if __name__ == '__main__':
    feature_file_directory = dataprep_sceneclassifier(feature_type = 'mfcc', 
                                                      sr=16000,
                                                      win_size_ms=20,
                                                      percent_overlap=0.5,
                                                      dur_sec=1,
                                                      num_feats=None,
                                                      visualize=True,
                                                      vis_every_n_frames=50,
                                                      subsection_data=False,
                                                      divide_factor=None,
                                                      window='hann')
    print('\nFeature datasets and related information can be found here: '+\
        '\n{}'.format(feature_file_directory))
