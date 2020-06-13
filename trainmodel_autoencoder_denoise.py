import os
import numpy as np
import pathlib
import pysoundtool as pyst
import pysoundtool.models as soundmodels
import keras
import time


###############################################################################

model_name = 'model_autoencoder_denoise'

feature_extraction_folder = 'features_stft_6m13d11h16m3s744ms_ONE_SECTION'
dataset_path = pyst.utils.check_dir('./audiodata/denoiser/{}/'.format(
    feature_extraction_folder), make = False) 
feature_type = 'stft'
model_name += '_'+feature_type + '_' + pyst.utils.get_date() 
model_dir = dataset_path.joinpath(model_name)
model_dir = pyst.utils.check_dir(model_dir)
model_name += '.h5'
model_path = model_dir.joinpath(model_name)

# TODO allow for multiple files to be loaded
# make sure data files exists:
dataset_base_filename = '{}_data_{}_{}.npy'
data_train_noisy_path = dataset_path.joinpath(dataset_base_filename.format(
    'train', 'noisy', feature_type))
data_val_noisy_path = dataset_path.joinpath(dataset_base_filename.format(
    'val', 'noisy', feature_type))
data_test_noisy_path = dataset_path.joinpath(dataset_base_filename.format(
    'test', 'noisy', feature_type))

data_train_clean_path = dataset_path.joinpath(dataset_base_filename.format(
    'train', 'clean', feature_type))
data_val_clean_path = dataset_path.joinpath(dataset_base_filename.format(
    'val', 'clean', feature_type))
data_test_clean_path = dataset_path.joinpath(dataset_base_filename.format(
    'test', 'clean', feature_type))

for pathway in [data_train_noisy_path, data_val_noisy_path, data_test_noisy_path, 
                data_train_clean_path, data_val_clean_path, data_test_clean_path]:
    if not os.path.exists(pathway):
        raise FileNotFoundError('File '+str(pathway) +\
            ' was not found or does not exist.')
    

# figure out input size of data:
# load smaller dataset
data_val_noisy = np.load(data_val_noisy_path)
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

data_train_noisy = np.load(data_train_noisy_path)
data_val_noisy = np.load(data_val_noisy_path)

data_train_clean = np.load(data_train_clean_path)
data_val_clean = np.load(data_val_clean_path)
    
    
data_train_noisy_shape = data_train_noisy.shape
data_val_noisy_shape = data_val_noisy.shape
data_train_clean_shape = data_train_clean.shape
data_val_clean_shape = data_val_clean.shape
    
    
start = time.time()

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
end = time.time()
total_dur_sec = round(end-start,2)
model_features_dict = dict(model_path = model_path,
                           data_train_noisy_path = data_train_noisy_path,
                           data_val_noisy_path = data_val_noisy_path, 
                           data_train_clean_path = data_train_clean_path, 
                           data_val_clean_path = data_val_clean_path,
                           data_train_noisy_shape = data_train_noisy_shape,
                           data_val_noisy_shape = data_val_noisy_shape,
                           data_train_clean_shape = data_train_clean_shape,
                           data_val_clean_shape = data_val_clean_shape,
                           total_dur_sec = total_dur_sec,
                           use_generator = use_generator)
model_features_dict.update(settings_dict)

model_features_dict_path = model_dir.joinpath('info_{}.csv'.format(
    model_name))
model_features_dict_path = pyst.utils.save_dict(model_features_dict,
                                                model_features_dict_path)
