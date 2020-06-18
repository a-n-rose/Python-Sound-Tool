import pysoundtool as pyst

# which features will we extract?
feature_type = 'stft'  # 'fbank', 'stft', 'mfcc', 'powspec'
feat_extraction_dir = 'features_'+feature_type + '_' + pyst.utils.get_date()

# 1) collect labels 
labels = []
data_scene_dir = pyst.utils.string2pathlib('./audiodata/minidatasets/background_noise/')
for label in data_scene_dir.glob('*/'):
    labels.append(label.stem)
labels = set(labels)

# 2) create paths for what we need to save:
classify_data_path = pyst.utils.check_dir('./audiodata/test_classifier/')
feat_extraction_dir = classify_data_path.joinpath(feat_extraction_dir)
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
paths_list = pyst.utils.collect_audiofiles(data_scene_dir, recursive=True)
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

'''When extracting features, need to first create empty matrix to fill.
This means we must know the final shape of all features put together:
'''

# decide settings, which will influence the size of data:
sr = 22050
win_size_ms = 25
percent_overlap = 0.5
dur_sec = 1
num_feats = None # allow for default settings depending of feature_type

# load the dataset_dict:
dataset_dict = pyst.utils.load_dict(dataset_dict_path)
# ensure only audiofiles in dataset_dict:

dataset_dict, datasets_path2save_dict = pyst.feats.save_features_datasets(
    datasets_dict = dataset_dict,
    datasets_path2save_dict = datasets_path2save_dict,
    labeled_data = True,
    feature_type = feature_type,
    sr = sr,
    dur_sec = dur_sec,
    num_feats = num_feats,
    win_size_ms = win_size_ms, 
    percent_overlap = percent_overlap,
    visualize=True, 
    vis_every_n_frames=30)

# save which audiofiles were extracted for each dataset
# save where extracted data were saved
dataprep_settings = dict(dataset_dict=dataset_dict,
                         datasets_path2save_dict=datasets_path2save_dict)
dataprep_settings_path = pyst.utils.save_dict(
    dataprep_settings,
    feat_extraction_dir.joinpath('dataset_audio_assignments.csv'))
