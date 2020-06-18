'''Example code for preparing audio data for training a convolutional neural network 
as a scene or noise environment classifier.
'''
import time
import pysoundtool as pyst


# TODO specifiy keyword arguments
def dataprep_sceneclassifier(
    data_dir = './audiodata/minidatasets/background_noise/',
    data_features_dir = './audiodata/features_scene_classifier/',
    feature_type = 'fbank', 
    sr=16000,
    win_size_ms=20,
    percent_overlap=0.5,
    dur_sec=1,
    num_feats=None,
    labeled_data=True,
    visualize=False,
    vis_every_n_frames=50,
    subsection_data=False,
    divide_factor=None
                             ):
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
        The type of features to be extracted. Options: 'stft', 'powspec', 'mfcc', 'fbank'.
        (default 'fbank')
    sr : int 
        The sample rate to load the audio data with. The higher the sample rate, the larger
        the dataset will be (more sample data to work with). Keep this in mind for limiting 
        computational cost. (default 16000)
    win_size_ms : int, float 
        The size of window in milliseconds to process audio data. (default 20 ms)
    percent_overlap : int, float
        The amount of overlap between windows. If set to 0.5, that results in 50% overlap.
        Value should be between 0 and 1.
        (default 0.5)
    dur_sec : int
        The duration in seconds of audio from each audio file to be used in feature extraction.
        If audio files are not long enough, they will be zeropadded. (default 1)
    num_feats : int, optional
        If None, default values will be set according to `feature_type`. 'fbank' : 40, 'mfcc' : 40,
        'stft' : frame_length (calcluated so: int(`win_size_ms` * `sr` // 1000)),
        'powspec' : frame_length.
        (default None)
    labeled_data : bool 
        If True, expects labels to be associated with every audio file. Labels will be collected
        from the subfolders in the `data_dir`. (default True)
    visualize : bool 
        If True, plots will be generated and saved through the feature extraction process. This 
        is useful if you want to know what the features look like or to ensure it looks the way
        you intended.
    vis_every_n_frames : int
        If `visualize` is set to True, this limits the number of plots made. E.g. if 
        `vis_every_n_frames` is set to 50, plots will be created every 50 windows / frames.
        (default 50)
    subsection_data : bool 
        If True, datasets will be sectioned into smaller sections. This is useful to control 
        memory usage during feature extraction and if datasets are particularly large. If 
        a MemoryError is raised, feature extraction will be attempted again, but with 
        `subsection_data` set to True. (default False)
    divide_factor : int 
        If `subsection_data` set to True, the number of subsections to divide datasets into. 
        
    Returns
    -------
    feat_extraction_dir : pathlib.PosixPath
        The pathway to where all feature extraction files can be found, including datasets.
    '''
    if 'signal' in feature_type:
        raise ValueError('Feature type "signal" is not yet supported for CNN training.')

    feat_extraction_dir = 'features_'+feature_type + '_' + pyst.utils.get_date()

    # 1) collect labels 
    labels = []
    data_dir = pyst.utils.string2pathlib(data_dir)
    for label in data_dir.glob('*/'):
        labels.append(label.stem)
    labels = set(labels)

    # 2) create paths for what we need to save:
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

    # 3) create and save encoding/decoding labels dicts
    dict_encode, dict_decode = pyst.data.create_dicts_labelsencoded(labels)
    dict_encode_path = pyst.utils.save_dict(dict_encode, 
                                        filename = dict_encode_path,
                                        overwrite=False)
    dict_decode_path = pyst.utils.save_dict(dict_encode, 
                                        filename = dict_decode_path,
                                        overwrite=False)

    # 4) save audio paths to each label in dict 
    # ensure only audiofiles included
    paths_list = pyst.utils.collect_audiofiles(data_dir, recursive=True)
    #paths_list = sorted(pyst.data.ensure_only_audiofiles(paths_list))
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

    # 5) extract features


    # load the dataset_dict:
    dataset_dict = pyst.utils.load_dict(dataset_dict_path)
    # ensure only audiofiles in dataset_dict:

    start = time.time()

    dataset_dict, datasets_path2save_dict = pyst.feats.save_features_datasets(
        datasets_dict = dataset_dict,
        datasets_path2save_dict = datasets_path2save_dict,
        labeled_data = labeled_data,
        feature_type = feature_type,
        sr = sr,
        dur_sec = dur_sec,
        num_feats = num_feats,
        win_size_ms = win_size_ms, 
        percent_overlap = percent_overlap,
        visualize = visualize, 
        vis_every_n_frames = vis_every_n_frames,
        subsection_data = subsection_data,
        divide_factor = divide_factor)

    end = time.time()
    
    total_dur_sec = end-start
    total_dur, units = pyst.utils.adjust_time_units(total_dur_sec)
    print('\nFinished! Total duration: {} {}.'.format(total_dur, units))

    # save which audiofiles were extracted for each dataset
    # save where extracted data were saved
    dataprep_settings = dict(dataset_dict=dataset_dict,
                            datasets_path2save_dict=datasets_path2save_dict,
                            total_dur_sec = total_dur_sec)
    dataprep_settings_path = pyst.utils.save_dict(
        dataprep_settings,
        feat_extraction_dir.joinpath('dataset_audio_assignments.csv'))
    
    return feat_extraction_dir

if __name__ == '__main__':
    feature_type = 'signal'
    dataprep_sceneclassifier(feature_type = feature_type)
 
