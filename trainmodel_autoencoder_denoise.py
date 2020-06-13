import os
import numpy as np
import pathlib
import pysoundtool as pyst
import pysoundtool.models as soundmodels
import keras
import time


###############################################################################

model_name = 'model_autoencoder_denoise'
num_training_files = 3

feature_extraction_folder = 'features_stft_6m13d11h15m42s506ms_THREE_SUBSECTIONS'
dataset_path = pyst.utils.check_dir('./audiodata/denoiser/{}/'.format(
    feature_extraction_folder), make = False) 
feature_type = 'stft'
model_name += '_'+feature_type + '_' + pyst.utils.get_date() 
model_dir = dataset_path.joinpath(model_name)
model_dir = pyst.utils.check_dir(model_dir, make=True)
model_name += '.h5'
model_path = model_dir.joinpath(model_name)

# make sure data files exists:
dataset_base_filename = '{}_data_{}_{}.npy'

###########################################
# collect all possible files for noisy data
train_paths_noisy = []
try:
    for i in range(num_training_files):
        num = i+1
        data_path = dataset_path.joinpath(dataset_base_filename.format(
            'train', 'noisy', feature_type+'__{}'.format(num)))
        if not os.path.exists(data_path):
            raise FileNotFoundError
        train_paths_noisy.append(data_path)
except FileNotFoundError:
    data_path = dataset_path.joinpath(dataset_base_filename.format(
    'train', 'noisy', feature_type))
    train_paths_noisy.append(data_path)
    
val_paths_noisy = []
try:
    for i in range(num_training_files):
        num = i+1
        data_path = dataset_path.joinpath(dataset_base_filename.format(
            'val', 'noisy', feature_type+'__{}'.format(num)))
        if not os.path.exists(data_path):
            raise FileNotFoundError
        val_paths_noisy.append(data_path)
except FileNotFoundError:
    data_path = dataset_path.joinpath(dataset_base_filename.format(
    'val', 'noisy', feature_type))
    val_paths_noisy.append(data_path)
    
test_paths_noisy = []
try:
    for i in range(num_training_files):
        num = i+1
        data_path = dataset_path.joinpath(dataset_base_filename.format(
            'val', 'noisy', feature_type+'__{}'.format(num)))
        if not os.path.exists(data_path):
            raise FileNotFoundError
        test_paths_noisy.append(data_path)
except FileNotFoundError:
    data_path = dataset_path.joinpath(dataset_base_filename.format(
    'val', 'noisy', feature_type))
    test_paths_noisy.append(data_path)
    
###########################################
# collect all possible files for clean data
train_paths_clean = []
try:
    for i in range(num_training_files):
        num = i+1
        data_path = dataset_path.joinpath(dataset_base_filename.format(
            'train', 'clean', feature_type+'__{}'.format(num)))
        if not os.path.exists(data_path):
            raise FileNotFoundError
        train_paths_clean.append(data_path)
except FileNotFoundError:
    data_path = dataset_path.joinpath(dataset_base_filename.format(
    'train', 'clean', feature_type))
    train_paths_clean.append(data_path)
    
val_paths_clean = []
try:
    for i in range(num_training_files):
        num = i+1
        data_path = dataset_path.joinpath(dataset_base_filename.format(
            'val', 'clean', feature_type+'__{}'.format(num)))
        if not os.path.exists(data_path):
            raise FileNotFoundError
        val_paths_clean.append(data_path)
except FileNotFoundError:
    data_path = dataset_path.joinpath(dataset_base_filename.format(
    'val', 'clean', feature_type))
    val_paths_clean.append(data_path)
    
test_paths_clean = []
try:
    for i in range(num_training_files):
        num = i+1
        data_path = dataset_path.joinpath(dataset_base_filename.format(
            'val', 'clean', feature_type+'__{}'.format(num)))
        if not os.path.exists(data_path):
            raise FileNotFoundError
        test_paths_clean.append(data_path)
except FileNotFoundError:
    data_path = dataset_path.joinpath(dataset_base_filename.format(
    'val', 'clean', feature_type))
    test_paths_clean.append(data_path)


# ensure all files exist:
for pathway in train_paths_noisy + val_paths_noisy + test_paths_noisy + \
    train_paths_clean + val_paths_clean + test_paths_clean:
    if not os.path.exists(pathway):
        raise FileNotFoundError('File '+str(pathway) +\
            ' was not found or does not exist.')
    

# figure out input size of data:
# load smaller dataset
data_val_noisy = np.load(val_paths_noisy[0])
# expect shape (num_audiofiles, batch_size, num_frames, num_features)
use_generator = True
num_epochs = 5
normalized = False
input_shape = data_val_noisy.shape[2:] + (1,)
del data_val_noisy

denoiser, settings_dict = soundmodels.autoencoder_denoise(
    input_shape = input_shape)

callbacks = soundmodels.setup_callbacks(patience=5,
                                        best_modelname = model_path, 
                                        log_filename = model_dir.joinpath('log.csv'))
adm = keras.optimizers.Adam(learning_rate=0.0001)
denoiser.compile(optimizer=adm, loss='binary_crossentropy')

# save variables that are not too large:
local_variables = locals()
global_variables = globals()
pyst.utils.save_dict(local_variables, 
                     model_dir.joinpath('local_variables_{}.csv'.format(
                         model_name)),
                     overwrite=True)
pyst.utils.save_dict(global_variables,
                     model_dir.joinpath('global_variables_{}.csv'.format(
                         model_name)),
                     overwrite = True)
    
    
start = time.time()


for i, train_path in enumerate(train_paths_noisy):
    #if i == 0:
    start_session = time.time()
    data_train_noisy_path = train_path
    data_train_clean_path = train_paths_clean[i]
    # just use first validation data file
    data_val_noisy_path = val_paths_noisy[0]
    data_val_clean_path = val_paths_clean[0]

    print('\nTRAINING SESSION ',i+1)
    print("Training on: ")
    print(data_train_noisy_path)
    print(data_train_clean_path)
    print()
    
    data_train_noisy = np.load(data_train_noisy_path)
    data_val_noisy = np.load(data_val_noisy_path)
    data_train_clean = np.load(data_train_clean_path)
    data_val_clean = np.load(data_val_clean_path)
        
    data_train_noisy_shape = data_train_noisy.shape
    data_val_noisy_shape = data_val_noisy.shape
    data_train_clean_shape = data_train_clean.shape
    data_val_clean_shape = data_val_clean.shape

    if use_generator:
        train_generator = soundmodels.Generator(data_matrix1 = data_train_noisy, 
                                                data_matrix2 = data_train_clean,
                                                normalized = normalized)
        val_generator = soundmodels.Generator(data_matrix1 = data_val_noisy,
                                            data_matrix2 = data_val_clean,
                                            normalized = normalized)

        train_generator.generator()
        val_generator.generator()
        history = denoiser.fit_generator(train_generator.generator(),
                                                steps_per_epoch = data_train_noisy.shape[0],
                                                epochs = num_epochs,
                                                callbacks = callbacks,
                                                validation_data = val_generator.generator(),
                                                validation_steps = data_val_noisy.shape[0])

    else:
        #reshape to mix samples and batchsizes:
        train_shape = (data_train_noisy.shape[0]*data_train_noisy.shape[1],)+ data_train_noisy.shape[2:] + (1,)
        val_shape = (data_val_noisy.shape[0]*data_val_noisy.shape[1],)+ data_val_noisy.shape[2:] + (1,)
        
        X_train = pyst.feats.normalize(data_train_noisy).reshape(train_shape)
        y_train = pyst.feats.normalize(data_train_clean).reshape(train_shape)
        X_val = pyst.feats.normalize(data_val_noisy).reshape(val_shape)
        y_val = pyst.feats.normalize(data_val_clean).reshape(val_shape)
        
        denoiser.fit(X_train, y_train, 
                                    epochs = num_epochs,
                                    batch_size = data_train_noisy.shape[1],
                                    callbacks = callbacks, 
                                    validation_data = (X_val, y_val))
    end_session = time.time()
    total_dur_sec_session = round(end_session-start_session,2)
    model_features_dict = dict(model_path = model_path,
                            data_train_noisy_path_1 = data_train_noisy_path,
                            data_val_noisy_path = data_val_noisy_path, 
                            data_train_clean_path_1 = data_train_clean_path, 
                            data_val_clean_path = data_val_clean_path,
                            data_train_noisy_shape = data_train_noisy_shape,
                            data_val_noisy_shape = data_val_noisy_shape,
                            data_train_clean_shape = data_train_clean_shape,
                            data_val_clean_shape = data_val_clean_shape,
                            total_dur_sec_session = total_dur_sec_session,
                            use_generator = use_generator)
    model_features_dict.update(settings_dict)
    if i == len(train_paths_noisy)-1:
        end = time.time()
        total_duration_seconds = round(end-start,2)
        time_dict = dict(total_duration_seconds=total_duration_seconds)
        model_features_dict.update(time_dict)

    model_features_dict_path = model_dir.joinpath('info_{}_{}.csv'.format(
        model_name, i))
    model_features_dict_path = pyst.utils.save_dict(model_features_dict,
                                                    model_features_dict_path)
