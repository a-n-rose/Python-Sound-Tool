'''
Validation data needs to be extracted already

Add label for invalid audio
'''


import pysoundtool as pyso 
import pysoundtool.models as pysodl
from random import shuffle
import numpy as np
import time

#def train_model(load_dict = None, 
         #feature_type = 'fbank',
         #num_feats = None,
         #mono = True,
         #rate_of_change = False,
         #rate_of_acceleration = False,
         #subtract_mean = False,
         #use_scipy = False,
         #dur_sec = 1,
         #win_size_ms = 25,
         #percent_overlap = 0.5,
         #sr = 22050,
         #fft_bins = None,
         #frames_per_sample = None, 
         #labeled_data = True, 
         #batch_size = 10,
         #use_librosa = True, 
         #center = True, 
         #mode = 'reflect', 
         #log_settings = True, 
         #model_name = 'env_classifier',
         #epoch = 5,
         #patience = 15,
         #callbacks = None,
         #use_generator = True,
         #augmentation_dict = None,
         #):
         
dataset_dict = None #'.csv' # None
load_dict= dataset_dict
model_name = 'augdata'
callbacks = None
use_generator = True
feature_type = 'fbank'
num_feats = 40
mono = True
rate_of_change = False
rate_of_acceleration = False
subtract_mean = False
use_scipy = False
dur_sec = 1
win_size_ms = 25
percent_overlap = 0.5
sr = 22050
fft_bins = None
frames_per_sample = None 
labeled_data = True 
batch_size = 1
use_librosa = True 
center = False 
mode = 'reflect' 
log_settings = True 
num_epochs = 1
patience = 15

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

augmentation_dicts = [augmentation_none, augmentation_noise, augmentation_speedup,
                      augmentation_speeddown, augmentation_pitchup,
                      augmentation_pitchdown, augmentation_timeshift,
                      augmentation_shuffle, augmentation_harmondist,
                      augmentation_vtlp, augmentation_all_speedup_pitchup,
                      augmentation_all_speedup_pitchdown,
                      augmentation_all_speeddown_pitchup,
                      augmentation_all_speeddown_pitchdown
                      ]
#augmentation_dicts = [augmentation_noise]

if load_dict is None:
    # collect labels of audio in data dir:
    data_dir = pyso.check_dir('../mini-audio-datasets/speech_commands/',
                            make = False)
    # collect labels
    labels = []
    for label in data_dir.glob('*/'):
        if label.suffix:
            # avoid adding unwanted files in the directory
            # want only directory names
            continue
        labels.append(label.stem)
    labels = set(labels)
    
    # setup dir to save feature extraction files:
    dataset_path = pyso.check_dir(
        './audiodata/example_feats_models/classifier/', 
        make=True)
    dataset_path = dataset_path.joinpath(
        'features_'+feature_type + '_' + pyso.utils.get_date())
    dataset_path = pyso.check_dir(dataset_path,
                                        make=True)

    # create encoding and decoding dictionaries of labels:
    dict_encode, dict_decode = pyso.datasets.create_dicts_labelsencoded(
        labels,
        add_invalid_label=True)

    # save labels and their encodings
    dict_encode_path = dataset_path.joinpath('dict_encode.csv')
    dict_decode_path = dataset_path.joinpath('dict_decode.csv')
    pyso.utils.save_dict(dict2save = dict_encode,
                        filename = dict_encode_path,
                        overwrite=True)
    dict_decode_path = pyso.utils.save_dict(dict2save = dict_decode,
                                            filename = dict_decode_path,
                                            overwrite=True)

    # get audio pathways and assign them their encoded labels:
    paths_list = pyso.files.collect_audiofiles(data_dir, recursive=True)
    paths_list = sorted(paths_list)

    dict_encodedlabel2audio = pyso.datasets.create_encodedlabel2audio_dict(
        dict_encode,
        paths_list)
    # path for saving dict for which audio paths are assigned to which labels:
    dict_encdodedlabel2audio_path = dataset_path.joinpath(
        'dict_encdodedlabel2audio.csv')

    pyso.utils.save_dict(dict2save = dict_encodedlabel2audio,
                        filename = dict_encdodedlabel2audio_path,
                        overwrite=True)

    # assign audio files int train, validation, and test datasets
    train, val, test = pyso.datasets.audio2datasets(
        dict_encdodedlabel2audio_path,
        perc_train=0.8,
        limit=None,
        seed=40)
    
    shuffle(train)
    shuffle(val)
    shuffle(test)

    # save audiofiles for each dataset to dict and save
    # for logging purposes
    dataset_dict = dict([('train', train),
                            ('val', val),
                            ('test', test)])
    dataset_dict_path = dataset_path.joinpath('dataset_audiofiles.csv')
    dataset_dict_path = pyso.utils.save_dict(
        dict2save = dataset_dict,
        filename = dataset_dict_path,
        overwrite=True)

else:
    dataset_dict = pyso.utils.load_dict(load_dict)

# should now be able to feed dictionaries to generator



# figure out shape of data:
total_samples = pyso.dsp.calc_frame_length(dur_sec*1000, sr=sr)
if use_librosa:
    frame_length = pyso.dsp.calc_frame_length(win_size_ms, sr)
    hop_length = int(win_size_ms*percent_overlap*0.001*sr)
    if fft_bins is None:
        fft_bins = frame_length
    # librosa centers samples by default, sligthly adjusting total 
    # number of samples
    if center:
        y_zeros = np.zeros((total_samples,))
        y_centered = np.pad(y_zeros, int(fft_bins // 2), mode=mode)
        total_samples = len(y_centered)
    # each audio file 
    if 'signal' in feature_type:
        # don't apply fft to signal (not sectioned into overlapping windows)
        total_rows_per_wav = total_samples // frame_length
    else:
        # do apply fft to signal (via Librosa) - (will be sectioned into overlapping windows)
        total_rows_per_wav = int(1 + (total_samples - fft_bins)//hop_length)
    # set defaults to num_feats if set as None:
    if num_feats is None:
        if 'mfcc' in feature_type or 'fbank' in feature_type:
            num_feats = 40
        elif 'powspec' in feature_type or 'stft' in feature_type:
            num_feats = int(1+fft_bins/2)
        elif 'signal' in feature_type:
            num_feats = frame_length
        else:
            raise ValueError('Feature type "{}" '.format(feature_type)+\
                'not understood.\nMust include one of the following: \n'+\
                    ', '.join(list_available_features()))
    if frames_per_sample is not None:
        # want smaller windows, e.g. autoencoder denoiser or speech recognition
        batch_size = math.ceil(total_rows_per_wav/frames_per_sample)
        if labeled_data:
            input_shape = (batch_size, frames_per_sample, num_feats + 1)
            desired_shape = (input_shape[0] * input_shape[1], 
                                input_shape[2]-1)
        else:
            input_shape = (batch_size, frames_per_sample, num_feats)
            desired_shape = (input_shape[0]*input_shape[1],
                                input_shape[2])
    else:
        if labeled_data:
            input_shape = (int(total_rows_per_wav), num_feats + 1)
            desired_shape = (input_shape[0], input_shape[1]-1)
        else:
            input_shape = (int(total_rows_per_wav), num_feats)
            desired_shape = input_shape
    #input_shape = (,) + input_shape
    # set whether or not features will include complex values:
    if 'stft' in feature_type:
        complex_vals = True
    else:
        complex_vals = False
    # limit feat_type to the basic feature extracted
    # for example:
    # feature_type 'powspec' is actually 'stft' but with complex info removed.
    # the basic feat_type is still 'stft'
    if 'mfcc' in feature_type:
        feat_type = 'mfcc'
    elif 'fbank' in feature_type:
        feat_type = 'fbank'
    elif 'stft' in feature_type:
        feat_type = 'stft'
    elif 'powspec' in feature_type:
        feat_type = 'stft'
    elif 'signal' in feature_type:
        feat_type = 'signal'
    else:
        raise TypeError('Expected '+', '.join(list_available_features())+\
            ' to be in `feature_type`, not {}'.format(feature_type))


    
# start training
start = time.time()
get_feats_kwargs = dict(sr = sr,
                        feature_type = feature_type,
                        win_size_ms = win_size_ms,
                        percent_overlap = percent_overlap,
                        num_filters = num_feats,
                        num_mfcc = num_feats,
                        dur_sec = dur_sec,
                        mono = mono,
                        rate_of_change = rate_of_change,
                        rate_of_acceleration = rate_of_acceleration,
                        subtract_mean = subtract_mean,
                        use_scipy = use_scipy)

input_shape = input_shape + (1,)
add_tensor_last = False
add_tensor_first = True
num_labels = len(dict_encode) 

for i, augmentation_dict in enumerate(augmentation_dicts):
    # designate where to save model and related files
    model_name = 'audioaugment_'
    if feature_type:
        model_name += '_'+feature_type + '_' + str(i)
    else:
        model_name += '_' + str(i)
    model_dir = dataset_path.joinpath(model_name)
    model_dir = pyso.utils.check_dir(model_dir, make=True)
    model_name += '.h5'
    model_path = model_dir.joinpath(model_name)
    
    # setup model 
    envclassifier, settings_dict = pysodl.cnn_classifier(
        input_shape = input_shape,
        num_labels = num_labels)
    callbacks = pysodl.setup_callbacks(
        patience = patience,
        best_modelname = model_path, 
        log_filename = model_dir.joinpath('log.csv'))
    optimizer = 'adam'
    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
    envclassifier.compile(optimizer = optimizer,
                            loss = loss,
                            metrics = metrics)
    
    train_generator = pysodl.GeneratorFeatExtraction(
        datalist = dataset_dict['train'],
        model_name = model_name,
        normalize = True,
        apply_log = False,
        randomize = False, 
        random_seed = 40,
        context_window = None,
        input_shape = input_shape,
        batch_size = batch_size, 
        add_tensor_last = add_tensor_last, 
        add_tensor_first = add_tensor_first,
        gray2color = False,
        visualize = True,
        vis_every_n_items = 1,
        decode_dict = dict_decode,
        dataset = 'train',
        augment_dict = augmentation_dict,
        ignore_invalid = False,
        **get_feats_kwargs)

    train_generator.generator()
    
    print('\nTRAINING SESSION ',i+1, ' out of ', len(augmentation_dicts))
    if augmentation_dict:
        print('Augmentation(s) applied: ')
        print(', '.join(augmentation_dict.keys()))
        print()
    else:
        print('No augmentations applied.\n')
    
    history = envclassifier.fit_generator(
        train_generator.generator(),
        steps_per_epoch = len(dataset_dict['train']),
        callbacks = callbacks,
        epochs = num_epochs,
        )

    model_features_dict = dict(model_path = model_path,
                            dataset_dict = dataset_dict,
                            augmentation_dict = augmentation_dict)
    model_features_dict.update(settings_dict)
    end = time.time()
    total_duration_seconds = round(end-start,2)
    time_dict = dict(total_duration_seconds=total_duration_seconds)
    model_features_dict.update(time_dict)

    model_features_dict_path = model_dir.joinpath('info_{}.csv'.format(
        model_name))
    model_features_dict_path = pyso.utils.save_dict(
        filename = model_features_dict_path,
        dict2save = model_features_dict)
    print('\nFinished training the model. The model and associated files can be '+\
        'found here: \n{}'.format(model_dir))
finished_time = time.time()
total_total_duration = finished_time - start
time_new_units, units = pyso.utils.adjust_time_units(total_total_duration)
print('\nEntire program took {} {}.\n\n'.format(time_new_units, units))
print('-'*79)
    
        
