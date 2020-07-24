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
center = True 
mode = 'reflect' 
log_settings = True 
num_epochs = 2
patience = 15

augmentation_dict1 = dict([('add_white_noise',True)])
augmentation_dict = dict([('add_white_noise',True),
                           ('speed_increase', True)])
augmentation_dict = dict([('add_white_noise',True),
                           ('speed_decrease', True)])

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

# designate where to save model and related files
if feature_type:
    model_name += '_'+feature_type + '_' + pyso.utils.get_date() 
else:
    model_name += '_' + pyso.utils.get_date() 
model_dir = dataset_path.joinpath(model_name)
model_dir = pyso.utils.check_dir(model_dir, make=True)
model_name += '.h5'
model_path = model_dir.joinpath(model_name)
num_labels = len(dict_encode) 

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

input_shape = input_shape + (1,)

# setup model 
envclassifier, settings_dict = pysodl.cnn_classifier(
    input_shape = input_shape,
    num_labels = num_labels)

if callbacks is None:
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

add_tensor_last = False
add_tensor_first = True

if use_generator:
    train_generator = pysodl.GeneratorFeatExtraction(
        datalist = dataset_dict['train'],
        model_name = 'testing_featextraction_gen',
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
        vis_every_n_items = 50,
        decode_dict = dict_decode,
        dataset = 'train',
        augment_dict = augmentation_dict,
        ignore_invalid = True,
        **get_feats_kwargs)
    
    val_generator = pysodl.GeneratorFeatExtraction(           
        datalist = dataset_dict['val'],
        model_name = 'testing_featextraction_gen',
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
        vis_every_n_items = 50,
        decode_dict = dict_decode,
        dataset = 'val',
        augment_dict = augmentation_dict,
        **get_feats_kwargs)
    
    test_generator = pysodl.GeneratorFeatExtraction(
        datalist = dataset_dict['test'],
        model_name = 'testing_featextraction_gen',
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
        vis_every_n_items = 50,
        decode_dict = dict_decode,
        dataset = 'test',
        augment_dict = augmentation_dict,
        **get_feats_kwargs)

    train_generator.generator()
    val_generator.generator()
    test_generator.generator()
    history = envclassifier.fit_generator(
        train_generator.generator(),
        steps_per_epoch = len(dataset_dict['train']),
        callbacks = callbacks,
        epochs = num_epochs,
        #validation_data = val_generator.generator(),
        #validation_steps = 1,#len(dataset_dict['val']),
        )
    
    ## TODO test how well prediction works. use simple predict instead?
    #y_predicted = envclassifier.predict_generator(
        #test_generator.generator(),
        #steps = len(dataset_dict['test']))

else:
    # TODO make scaling data optional?
    # data is separated and shaped for this classifier in scale_X_y..
    X_train, y_train, scalars = pyso.feats.scale_X_y(data_train,
                                                        is_train=True)
    X_val, y_val, __ = pyso.feats.scale_X_y(data_val,
                                            is_train=False, 
                                            scalars=scalars)
    X_test, y_test, __ = pyso.feats.scale_X_y(data_test,
                                                is_train=False, 
                                                scalars=scalars)
    
    history = envclassifier.fit(X_train, y_train, 
                                callbacks = callbacks, 
                                validation_data = (X_val, y_val),
                                **kwargs)
    
    envclassifier.evaluate(X_test, y_test)
    y_predicted = envclassifier.predict(X_test)
    # which category did the model predict?
    

#y_pred = np.argmax(y_predicted, axis=1)
#if len(y_pred.shape) > len(y_test.shape):
    #y_test = np.expand_dims(y_test, axis=1)
#elif len(y_pred.shape) < len(y_test.shape):
    #y_pred = np.expand_dims(y_pred, axis=1)
#try:
    #assert y_pred.shape == y_test.shape
#except AssertionError:
    #raise ValueError('The shape of prediction data {}'.format(y_pred.shape) +\
        #' does not match the `y_test` dataset {}'.format(y_test.shape) +\
            #'\nThe shapes much match in order to measure accuracy.')
        
#match = sum(y_test == y_pred)
#if len(match.shape) == 1:
    #match = match[0]
#test_accuracy = round(match/len(y_test),4)
#print('\nModel reached accuracy of {}%'.format(test_accuracy*100))

model_features_dict = dict(model_path = model_path,
                        dataset_dict = dataset_dict,
                        use_generator = use_generator)
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
  
    



#model_dir, history = train_model(
            #load_dict= dataset_dict,
            #feature_type = feature_type,
            #num_feats = num_feats,
            #mono = mono,
            #rate_of_change = rate_of_change,
            #rate_of_acceleration = rate_of_acceleration,
            #subtract_mean = subtract_mean,
            #use_scipy = use_scipy,
            #dur_sec = dur_sec,
            #win_size_ms = win_size_ms,
            #percent_overlap = percent_overlap,
            #sr = sr,
            #fft_bins = fft_bins,
            #frames_per_sample = frames_per_sample, 
            #labeled_data = labeled_data, 
            #batch_size = batch_size,
            #use_librosa = use_librosa, 
            #center = center, 
            #mode = mode, 
            #log_settings = log_settings, 
            #epoch = epoch,
            #patience = patience,
            #augmentation_dict = augmentation_dict
            #)
    
    
