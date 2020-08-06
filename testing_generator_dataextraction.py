import pysoundtool as pyso 
import matplotlib.pyplot as plt

feature_type = 'stft'
num_filters = 40
mono = True
rate_of_change = False
rate_of_acceleration = False
subtract_mean = False
use_scipy = True
dur_sec = 1
win_size_ms = 25
percent_overlap = 0.5
sr = 48000
fft_bins = None
center = False 
mode = 'reflect'
real_signal = True

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
                        mode = mode,
                        real_signal = real_signal)


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
augmentation_pitchdown_novtlp = dict([('add_white_noise',False),
                                             ('speed_decrease', False),
                                             ('pitch_decrease', True),
                                             ('harmonic_distortion', False),
                                             ('vtlp', False)
                                             ])
augmentation_pitchdown_vtlp = dict([('add_white_noise',False),
                                             ('speed_decrease', False),
                                             ('pitch_decrease', True),
                                             ('harmonic_distortion', False),
                                             ('vtlp', True)
                                             ])

# get defaults dict of all augmentations:
augment_settings_dict = {}
for key in pyso.augment.get_augmentation_dict().keys():
    augment_settings_dict[key] = pyso.augment.get_augmentation_settings_dict(key)
# if want to change augmentation settings:
# Note: these changes will over-ride the default values of the generator
augment_settings_dict['pitch_decrease']['num_semitones'] = 1 
augment_settings_dict['add_white_noise']['snr'] = [10,15,20]
augment_settings_dict['speed_decrease']['perc'] = 0.1

audiodata_path = '../mini-audio-datasets/speech_commands/'
#augmentation_harmondist.update(
    #dict(augment_settings_dict=augment_settings_dict))
augment_dict_list = [augmentation_vtlp,
                     augmentation_none, 
                     augmentation_pitchdown_novtlp,
                     augmentation_speedup]
labeled_data = True 
batch_size = 1
use_librosa = True 
frames_per_sample = None 
log_settings = True 
epochs =2
patience = 15



save_new_files_dir = pyso.check_dir('example_feats_models/envclassifer/', make=True)
#print(aug_settings_dict)
#aug_settings_dict_path = pyso.utils.save_dict(
    #filename = save_new_files_dir.joinpath('aug_settings.csv'),
    #dict2save = aug_settings_dict,
    #overwrite = True)



model_dir, history = pyso.envclassifier_extract_train(
    model_name = 'testing_augment_buildin',
    audiodata_path = audiodata_path,
    augment_dict_list = augment_dict_list,
    save_new_files_dir = save_new_files_dir,
    #augment_settings_dict = aug_settings_dict_path,
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
plt.savefig('model_{}.png'.format(pyso.utils.get_date()))
