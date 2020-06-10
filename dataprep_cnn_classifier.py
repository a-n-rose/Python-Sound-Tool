
import pysoundtool as pyst
import numpy as np

# classifier:

# 1) collect labels 
labels = []
data_scene_dir = pyst.utils.str2path('./audiodata/minidatasets/background_noise/')
for label in data_scene_dir.glob('*/'):
    labels.append(label.stem)
labels = set(labels)

# 2) create paths for what we need to save:
classify_data_path = pyst.utils.check_dir('./audiodata/test_classifier/')

# dictionaries containing encoding and decoding labels:
dict_encode_path = classify_data_path.joinpath('dict_encode.csv')
dict_decode_path = classify_data_path.joinpath('dict_decode.csv')
# dictionary for which audio paths are assigned to which labels:
dict_encdodedlabel2audio_path = classify_data_path.joinpath('dict_encdodedlabel2audio.csv')
# train, val, and test data
# which features will we extract?
feature_type = 'mfcc'  # 'fbank', 'stft', 'mfcc'
data_train_path = classify_data_path.joinpath('{}_data_{}.npy'.format('train',
                                                                      feature_type))
data_val_path = classify_data_path.joinpath('{}_data_{}.npy'.format('val',
                                                                      feature_type))
data_test_path = classify_data_path.joinpath('{}_data_{}.npy'.format('test',
                                                                      feature_type))
# TODO
# log feature settings as well!
feature_settings_path = classify_data_path.joinpath('feature_settings.csv')


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
paths_list = pyst.utils.collect_audiofiles(data_scene_dir, recursive=True)
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
dataset_dict_path = classify_data_path.joinpath('dataset_audiofiles.csv')

try:
    dataset_dict_path = pyst.utils.save_dict(dataset_dict, dataset_dict_path, 
                                            overwrite=True)
except FileExistsError:
    pass

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
    
# depending on which packages one uses, shape of data changes.
# for example, Librosa centers/zeropads data automatically
# TODO see which shapes result from python_speech_features
total_samples = pyst.dsp.calc_frame_length(dur_sec*1000, sr=sr)
use_librosa= True
# if using Librosa:
if use_librosa:
    hop_length = int(win_size_ms*percent_overlap*0.001*sr)
    center=True
    mode='reflect'
    # librosa centers samples by default, sligthly adjusting total 
    # number of samples
    if center:
        y_zeros = np.zeros((total_samples,))
        y_centered = np.pad(y_zeros, int(n_fft // 2), mode=mode)
        total_samples = len(y_centered)
    # each audio file 
    total_rows_per_wav = int(1 + (total_samples - n_fft)//hop_length)
    
    dataset_paths = [data_train_path, data_val_path, data_test_path]
    for i, dataset in enumerate([train, val, test]):
        # +1 for the label column
        feats_matrix = pyst.dsp.create_empty_matrix(
            (len(dataset), int(total_rows_per_wav), num_feats + 1), 
            complex_vals=complex_vals)


        for j, audioset in enumerate(dataset):
            label, audiofile = int(audioset[0]), audioset[1]
            feats = pyst.feats.get_feats(audiofile,
                                        sr=sr,
                                        features=feature_type,
                                        win_size_ms=win_size_ms,
                                        percent_overlap=percent_overlap,
                                        window='hann',
                                        num_filters=num_feats,
                                        num_mfcc=num_feats,
                                        duration=dur_sec)
            # add label:
            label_col = np.zeros((len(feats),1)) + label
            feats = np.concatenate([feats,label_col], axis=1)
            # fill in empty matrix with features from each audiofile
            feats_matrix[j] = feats
            
            # visualize features:
            if 'mfcc' in feature_type:
                scale = None
            else:
                scale = 'power_to_db'
            # visualize features only every n num frames
            every_n_frames = 50
            if j % every_n_frames == 0:
                pyst.feats.plot(feats, feature_type = feature_type, scale=scale)
        # must be 2 D to visualize
        pyst.feats.plot(feats_matrix.reshape((feats_matrix.shape[0] * feats_matrix.shape[1], feats_matrix.shape[2])), feature_type=feature_type,
                        scale=scale, x_label='number of audio files')
        # save data:
        np.save(dataset_paths[i], feats_matrix)
