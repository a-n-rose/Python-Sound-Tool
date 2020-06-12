
import pysoundtool as pyst
import numpy as np

# classifier:
feature_type = 'stft'  # 'fbank', 'stft', 'mfcc'
feat_extraction_dir = 'features_'+feature_type + '_' + pyst.utils.get_date()

# 1) collect labels 
labels = []
data_scene_dir = pyst.utils.str2path('./audiodata/minidatasets/background_noise/')
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
## TODO
## log feature settings as well!
#feature_settings_path = feat_extraction_dir.joinpath('feature_settings.csv')


# 3) create and save encoding/decoding labels dicts
dict_encode, dict_decode = pyst.data.create_dicts_labelsencoded(labels)

try:
    dict_encode_path = pyst.utils.save_dict(dict_encode, 
                                       filename = dict_encode_path,
                                       overwrite=False)
except FileExistsError:
    pass
try:
    dict_decode_path = pyst.utils.save_dict(dict_encode, 
                                       filename = dict_decode_path,
                                       overwrite=False)
except FileExistsError:
    pass

# 4) save audio paths to each label in dict 
# ensure only audiofiles included
paths_list = pyst.utils.collect_audiofiles(data_scene_dir, recursive=True)
paths_list = sorted(pyst.data.ensure_only_audiofiles(paths_list))

dict_encodedlabel2audio = pyst.data.create_encodedlabel2audio_dict(dict_encode,
                                                     paths_list)
try:
    dict_encdodedlabel2audio_path = pyst.utils.save_dict(dict_encodedlabel2audio, 
                                            filename = dict_encdodedlabel2audio_path, overwrite=False)
except FileExistsError:
    pass

train, val, test = pyst.data.audio2datasets(dict_encdodedlabel2audio_path,
                                             perc_train=0.8,
                                             limit=None,
                                             seed=40)

# save audiofiles for each dataset:
dataset_dict = dict([('train',train),('val', val),('test',test)])
datasets_path2save_dict = dict([('train',data_train_path),
                                ('val', data_val_path),
                                ('test',data_test_path)])

dataset_dict_path = feat_extraction_dir.joinpath('dataset_audiofiles.csv')

try:
    dataset_dict_path = pyst.utils.save_dict(dataset_dict, dataset_dict_path, 
                                            overwrite=True)
except FileExistsError:
    pass


# clear out variables
variables2remove = [dataset_dict, train, val, test, paths_list,
                    dict_encdodedlabel2audio_path, dict_encodedlabel2audio]

for var in variables2remove:
    del var

# 5) extract features

'''When extracting features, need to first create empty matrix to fill.
This means we must know the final shape of all features put together:
'''

# decide settings, which will influence the size of data:
sr = 22050
win_size_ms = 25
percent_overlap = 0.5
dur_sec = 1
# decide number of features:
# for mfcc, this is the number of mel cepstral coefficients (common: 13, 40)
# for bank, the number of filters applied (common: 22, 24, 29, 40, 128 (etc.) )
num_feats = 40 
complex_vals = False # complex values will be present if extracting stft features
# for mfcc, fbank, and stft features, Fast Fourier Transform is applied (fft)
# must determin how many fft bins (frequency bins) will be used
# default number of fft bins is the frame length (num samples in each processing window)
n_fft = win_size_ms * sr // 1000
# same as:
frame_length = pyst.dsp.calc_frame_length(win_size_ms, sr)

# number of features for stft data may be different if not using Librosa.
if 'stft' in feature_type:
    # using Librosa for extracing stft features:
    # according to Librosa 
    # num features is the number of fft bins divided by 2 + 1
    num_feats = int(1+n_fft/2)
    complex_vals = True
    
   
# which variables to include?
local_variables = locals()
global_variables = globals()

pyst.utils.save_dict(local_variables, 
                    feat_extraction_dir.joinpath('local_variables_{}.csv'.format(
                        feature_type)),
                    overwrite=True)
pyst.utils.save_dict(global_variables,
                    feat_extraction_dir.joinpath('global_variables_{}.csv'.format(
                        feature_type)),
                    overwrite = True)
    
# load the dataset_dict:
dataset_dict = pyst.utils.load_dict(dataset_dict_path)
# ensure only audiofiles in dataset_dict:

pyst.feats.save_features_datasets_dicts(
    datasets_dict = dataset_dict,
    datasets_path2save_dict = datasets_path2save_dict,
    labeled_data = True,
    feature_type = feature_type,
    sr = sr,
    n_fft = n_fft,
    dur_sec = dur_sec,
    num_feats = num_feats,
    use_librosa=True, 
    win_size_ms = win_size_ms, 
    percent_overlap = percent_overlap,
    window='hann', 
    center=True, 
    mode='reflect',
    frames_per_sample=None, 
    complex_vals=complex_vals,
    visualize=True, 
    vis_every_n_frames=30)
