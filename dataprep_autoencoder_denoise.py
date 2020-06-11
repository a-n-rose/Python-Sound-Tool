
import pysoundtool as pyst


# which features will we extract?
feature_type = 'mfcc'  # 'fbank', 'stft', 'mfcc'
feat_extraction_dir = 'features_'+feature_type + '_' + pyst.utils.get_date()


# 1) specify clean and noisy audio directories, and ensure they exist
audio_clean_path = pyst.utils.check_dir(\
    '/home/airos/Projects/Data/denoising_sample_data/clean_speech_IEEE/', 
                                         make=False)
audio_noisy_path = pyst.utils.check_dir(\
    '/home/airos/Projects/Data/denoising_sample_data/noisy_speech_varied_IEEE/',
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

# TODO
# log feature settings as well!
feature_settings_path = feat_extraction_dir.joinpath('feature_settings.csv')


# 3) collect audiofiles and divide them into train, val, and test datasets
# can do this different ways.. this is simply one way I've done it.

noisyaudio = pyst.utils.collect_audiofiles(audio_noisy_path, 
                                                  hidden_files = False,
                                                  wav_only = False,
                                                  recursive = False)
# remove non-audio files
noisyaudio = sorted(pyst.data.ensure_only_audiofiles(noisyaudio))

cleanaudio = pyst.utils.collect_audiofiles(audio_clean_path, 
                                                  hidden_files = False,
                                                  wav_only = False,
                                                  recursive = False)
cleanaudio = sorted(pyst.data.ensure_only_audiofiles(cleanaudio))

# check if they match up:
for i, audiofile in enumerate(noisyaudio):
    if not pyst.utils.check_noisy_clean_match(audiofile, cleanaudio[i]):
        raise ValueError('The noisy and clean audio datasets do not appear to match.')

# save to csv:
noisy_audio_dict = dict([('noisy', noisyaudio)])
clean_audio_dict = dict([('clean', cleanaudio)])

noisy_audio_dict_path = feat_extraction_dir.joinpath('noisy_audio.csv')
noisy_audio_dict_path = pyst.utils.save_dict(noisy_audio_dict, noisy_audio_dict_path,
                                             overwrite=True)

clean_audio_dict_path = feat_extraction_dir.joinpath('clean_audio.csv')
clean_audio_dict_path = pyst.utils.save_dict(clean_audio_dict, clean_audio_dict_path,
                                             overwrite=True)

train_noisy, val_noisy, test_noisy = pyst.data.audio2datasets(noisy_audio_dict_path,
                                                              seed=40)
train_clean, val_clean, test_clean = pyst.data.audio2datasets(clean_audio_dict_path,
                                                              seed=40)

# save train, val, test datasets:

dataset_dict_noisy = dict([('train', train_noisy),('val', val_noisy),('test', test_noisy)])
dataset_dict_clean = dict([('train', train_clean),('val', val_clean),('test', test_clean)])

# keep track of paths to save data
dataset_paths_noisy_dict = dict([('train',data_train_noisy_path),
                                 ('val', data_val_noisy_path),
                                 ('test',data_test_noisy_path)])
dataset_paths_clean_dict = dict([('train',data_train_clean_path),
                                 ('val', data_val_clean_path),
                                 ('test',data_test_clean_path)])

path2_noisy_dataset_waves = feat_extraction_dir.joinpath('audiofiles_datasets_noisy.csv')
path2_clean_dataset_waves = feat_extraction_dir.joinpath('audiofiles_datasets_clean.csv')

path2_noisy_dataset_waves = pyst.utils.save_dict(dataset_dict_noisy,
                                                 path2_noisy_dataset_waves,
                                                 overwrite=True)
path2_clean_dataset_waves = pyst.utils.save_dict(dataset_dict_clean,
                                                 path2_clean_dataset_waves,
                                                 overwrite=True)

variables2remove = [noisyaudio, cleanaudio, train_noisy, val_noisy, test_noisy,
                    train_clean, val_clean, test_clean, dataset_dict_noisy, 
                    dataset_dict_clean]

for var in variables2remove:
    del var


# 5) extract features

'''When extracting features, need to first create empty matrix to fill.
This means we must know the final shape of all features put together:
'''

# decide settings, which will influence the size of data:
sr = 22050
win_size_ms = 16
percent_overlap = 0.5
dur_sec = 3
frames_per_sample = 11
# decide number of features:
# for mfcc, this is the number of mel cepstral coefficients (common: 13, 40)
# for bank, the number of filters applied (common: 22, 24, 29, 40, 128 (etc.) )
num_feats = 40 
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
else:
    complex_vals = False
    

# save settings of features
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

# load audiofile dicts and convert value to list instead of string
dataset_dict_clean = pyst.utils.load_dict(path2_clean_dataset_waves)
for key, value in dataset_dict_clean.items():
    if isinstance(value, str):
        dataset_dict_clean[key] = pyst.utils.string2list(value)
dataset_dict_noisy = pyst.utils.load_dict(path2_noisy_dataset_waves)
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


pyst.feats.save_features_datasets_dicts(
    datasets_dict = dataset_dict_clean,
    datasets_path2save_dict = dataset_paths_clean_dict,
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
    frames_per_sample=11, 
    complex_vals=complex_vals,
    visualize=False, 
    vis_every_n_frames=50)
    
pyst.feats.save_features_datasets_dicts(
    datasets_dict = dataset_dict_noisy,
    datasets_path2save_dict = dataset_paths_noisy_dict,
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
    frames_per_sample=11, 
    complex_vals=complex_vals,
    visualize=False, 
    vis_every_n_frames=50)
