import pysoundtool as pyso 
import matplotlib.pyplot as plt

feature_type = 'fbank'
num_filters = 40
mono = True
rate_of_change = False
rate_of_acceleration = False
subtract_mean = False
use_scipy = False
dur_sec = 1
win_size_ms = 25
percent_overlap = 0.5
sr = 48000
fft_bins = None
center = False 
mode = 'reflect' 

get_feats_kwargs = dict(feature_type = feature_type,
                        num_filters = num_filters,
                        mono = mono,
                        rate_of_change = rate_of_change,
                        rate_of_acceleration = rate_of_acceleration,
                        subtract_mean = subtract_mean,
                        use_scipy = use_scipy,
                        dur_sec = dur_sec,
                        win_size_ms = win_size_ms,
                        percent_overlap = percent_overlap,
                        sr = sr,
                        fft_bins = fft_bins,
                        center = center,
                        mode = mode)


augmentation_none = dict()
augmentation_noise = dict([('add_white_noise',True)])
augmentation_speedup = dict([('speed_increase', True)])
augmentation_speeddown = dict([('speed_decrease', True)])
augmentation_pitchup = dict([('pitch_increase', True)])
augmentation_pitchdown = dict([('pitch_decrease', True)])
augmentation_timeshift = dict([('time_shift', True)])
augmentation_shuffle = dict([('shufflesound', True)])
augmentation_harmondist = dict([('harmonic_distortion', True)])
augmentation_vtlp = dict([('vtlp', True)])
augmentation_all_speedup_pitchup = dict([('add_white_noise',True),
                                         ('speed_increase', True),
                                         ('pitch_increase', True),
                                         ('time_shift', False),
                                         ('harmonic_distortion', True),
                                         ('vtlp', True)
                                         ])
augmentation_all_speedup_pitchdown = dict([('add_white_noise',True),
                                           ('speed_increase', True),
                                           ('pitch_decrease', True),
                                           ('time_shift', False),
                                           ('harmonic_distortion', True),
                                           ('vtlp', True)
                                           ])
augmentation_all_speeddown_pitchup = dict([('add_white_noise',True),
                                           ('speed_decrease', True),
                                           ('pitch_increase', True),
                                           ('time_shift', False),
                                           ('harmonic_distortion', True),
                                           ('vtlp', True)
                                           ])
augmentation_all_speeddown_pitchdown = dict([('add_white_noise',True),
                                             ('speed_decrease', True),
                                             ('pitch_decrease', True),
                                             ('harmonic_distortion', True),
                                             ('vtlp', True)
                                             ])
augmentation_all_speeddown_pitchdown_novtlp = dict([('add_white_noise',True),
                                             ('speed_decrease', True),
                                             ('pitch_decrease', True),
                                             ('harmonic_distortion', True),
                                             ('vtlp', False)
                                             ])



audiodata_path = '../mini-audio-datasets/speech_commands/'
augment_dict_list = [augmentation_all_speeddown_pitchdown_novtlp]
labeled_data = True 
batch_size = 1
use_librosa = True 
frames_per_sample = None 
log_settings = True 
epochs = 5
patience = 15

# get defaults dict of all augmentations:
aug_settings_dict = {}
for key in pyso.augment.get_augmentation_dict().keys():
    aug_settings_dict[key] = pyso.augment.get_augmentation_settings_dict(key)
# if want to change augmentation settings:
# Note: these changes will over-ride the default values of the generator
aug_settings_dict['pitch_decrease']['num_semitones'] = 1 
aug_settings_dict['add_white_noise']['snr'] = [10,15,20]
aug_settings_dict['speed_decrease']['perc'] = 0.1

model_dir, history = pyso.envclassifier_extract_train(
    model_name = 'testing_augment_buildin',
    audiodata_path = audiodata_path,
    augment_dict_list = augment_dict_list,
    augment_settings_dict = aug_settings_dict,
    labeled_data = labeled_data,
    batch_size = batch_size,
    use_librosa = use_librosa,
    frames_per_sample = frames_per_sample,
    epochs = epochs, 
    patience = patience,
    visualize = True,
    vis_every_n_items = 50,
    **get_feats_kwargs)

plt.clf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()
