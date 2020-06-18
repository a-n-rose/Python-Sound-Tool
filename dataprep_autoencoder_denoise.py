import pysoundtool as pyst
import time

# which features will we extract?
feature_type = 'stft'  # 'fbank', 'stft', 'mfcc', 'powspec'
feat_extraction_dir = 'features_'+feature_type + '_' + pyst.utils.get_date()

## datasets:
## clean
# uwnu_clean       (large)
# clean_speech_IEEE       (medium)
# cleanspeech_IEEE_small   

## noisy
# uwnu_noisy_varied       (large)
# noisy_speech_varied_IEEE    (medium)
# noisyspeech_IEEE_small

# 1) specify clean and noisy audio directories, and ensure they exist
audio_clean_path = pyst.utils.check_dir(\
    '/home/airos/Projects/Data/denoising_sample_data/uwnu_clean/', 
                                         make=False)
audio_noisy_path = pyst.utils.check_dir(\
    '/home/airos/Projects/Data/denoising_sample_data/uwnu_noisy_varied/',
                                         make=False)

# 2) create paths for what we need to save:
denoise_data_path = pyst.utils.check_dir('./audiodata/denoiser/', make=True)
feat_extraction_dir = denoise_data_path.joinpath(feat_extraction_dir)
feat_extraction_dir = pyst.utils.check_dir(feat_extraction_dir, make=True)
# train, val, and test data
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
# sort audio
#noisyaudio = sorted(pyst.data.ensure_only_audiofiles(noisyaudio))
noisyaudio = sorted(noisyaudio)

# clean data
cleanaudio = pyst.utils.collect_audiofiles(audio_clean_path, 
                                                  hidden_files = False,
                                                  wav_only = False,
                                                  recursive = False)
#cleanaudio = sorted(pyst.data.ensure_only_audiofiles(cleanaudio))
cleanaudio = sorted(cleanaudio)


# check if they match up: (expects clean file name to be in noisy file name)
for i, audiofile in enumerate(noisyaudio):
    if not pyst.utils.check_noisy_clean_match(audiofile, cleanaudio[i]):
        raise ValueError('The noisy and clean audio datasets do not appear to match.')

# save to csv:
noisy_audio_dict = dict([('noisy', noisyaudio)])
noisy_audio_dict_path = feat_extraction_dir.joinpath('noisy_audio.csv')
noisy_audio_dict_path = pyst.utils.save_dict(noisy_audio_dict, noisy_audio_dict_path,
                                             overwrite=False)
clean_audio_dict = dict([('clean', cleanaudio)])
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

path2_noisy_datasets = pyst.utils.save_dict(dataset_dict_noisy,
                                                 path2_noisy_datasets,
                                                 overwrite=False)
path2_clean_datasets = pyst.utils.save_dict(dataset_dict_clean,
                                                 path2_clean_datasets,
                                                 overwrite=False)

# 5) extract features
# decide settings, which will influence the size of data:
sr = 22050
win_size_ms = 16
percent_overlap = 0.5
dur_sec = 3
frames_per_sample = 11
num_feats = None # use defaults for each feature_type
visualize = True

# load audiofile dicts and convert value to list instead of string
dataset_dict_clean = pyst.utils.load_dict(path2_clean_datasets)
for key, value in dataset_dict_clean.items():
    if isinstance(value, str):
        dataset_dict_clean[key] = pyst.utils.string2list(value)
dataset_dict_noisy = pyst.utils.load_dict(path2_noisy_datasets)
for key, value in dataset_dict_noisy.items():
    if isinstance(value, str):
        dataset_dict_noisy[key] = pyst.utils.string2list(value)
        
# ensure the noisy and clean values match up:
for key, value in dataset_dict_noisy.items():
    for j, audiofile in enumerate(value):
        if not pyst.utils.check_noisy_clean_match(audiofile,
                                                dataset_dict_clean[key][j]):
            raise ValueError('There is a mismatch between noisy and clean audio. '+\
                '\nThe noisy file:\n{}'.format(dataset_dict_noisy[key][i])+\
                    '\ndoes not seem to match the clean file:\n{}'.format(audiofile))

start = time.time()

dataset_dict_clean, dataset_paths_clean_dict = pyst.feats.save_features_datasets(
    datasets_dict = dataset_dict_clean,
    datasets_path2save_dict = dataset_paths_clean_dict,
    feature_type = feature_type + ' clean',
    sr = sr,
    dur_sec = dur_sec,
    num_feats = num_feats,
    win_size_ms = win_size_ms, 
    percent_overlap = percent_overlap,
    frames_per_sample=11, 
    visualize=visualize, 
    vis_every_n_frames=100,
    subsection_data=True,
    divide_factor=10)
    
dataset_dict_noisy, dataset_paths_noisy_dict = pyst.feats.save_features_datasets(
    datasets_dict = dataset_dict_noisy,
    datasets_path2save_dict = dataset_paths_noisy_dict,
    feature_type = feature_type + ' noisy',
    sr = sr,
    dur_sec = dur_sec,
    num_feats = num_feats,
    win_size_ms = win_size_ms, 
    percent_overlap = percent_overlap,
    frames_per_sample=11, 
    visualize=visualize, 
    vis_every_n_frames=100,
    subsection_data=True,
    divide_factor=10)

end = time.time()

total_duration_seconds = round(end-start,2)
# save which audiofiles were extracted for each dataset
# save where extracted data were saved
dataprep_settings = dict(dataset_dict_noisy = dataset_dict_noisy,
                         dataset_paths_noisy_dict = dataset_paths_noisy_dict,
                         dataset_dict_clean = dataset_dict_clean,
                         dataset_paths_clean_dict = dataset_paths_clean_dict,
                         total_duration_seconds = total_duration_seconds)
dataprep_settings_path = pyst.utils.save_dict(
    dataprep_settings,
    feat_extraction_dir.joinpath('dataset_audio_assignments.csv'))
