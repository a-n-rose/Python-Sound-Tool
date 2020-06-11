import os
import numpy as np
import pathlib
import pysoundtool as pyst
import pysoundtool.models as soundmodels
import time

###############################################################################

model_name = 'model_cnn_classifier'

dataset_path = pyst.utils.check_dir('./audiodata/scene_classifier/', make = False) 
feature_type = 'stft'
model_name += '_'+feature_type + '_' + pyst.utils.get_date() 
model_dir = dataset_path.joinpath(model_name)
model_dir = pyst.utils.check_dir(model_dir)
model_name += '.h5'
model_path = model_dir.joinpath(model_name)

# make sure data files exists:
data_train_path = dataset_path.joinpath('train_data_{}.npy'.format(feature_type))
data_val_path = dataset_path.joinpath('val_data_{}.npy'.format(feature_type))
data_test_path = dataset_path.joinpath('test_data_{}.npy'.format(feature_type))
dict_decode_path = dataset_path.joinpath('dict_decode.csv')

for pathway in [data_train_path, data_val_path, data_test_path, dict_decode_path]:
    if not os.path.exists(pathway):
        raise FileNotFoundError('File '+str(pathway) +\
            ' was not found or does not exist.')
    

# figure out input size of data:
# load smaller dataset
data_val = np.load(data_val_path)
# expect shape (num_audiofiles, num_frames, num_features + label_column)
# subtract the label column and add dimension for 'color scale' 
input_shape = (data_val.shape[1], data_val.shape[2] - 1, 1) 
del data_val

# load dictionary with labels to find out the number of labels:
dict_decode = pyst.utils.load_dict(dict_decode_path)
num_labels = len(dict_decode)

scene_classifier, settings_dict = soundmodels.cnn_classifier(input_shape = input_shape,
                                              num_labels = num_labels)

callbacks = soundmodels.setup_callbacks(patience=15,
                                        best_modelname = model_path)
optimizer = 'adam'
loss = 'sparse_categorical_crossentropy'
metrics = ['accuracy']
scene_classifier.compile(optimizer = optimizer,
                         loss = loss,
                         metrics = metrics)

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

data_train = np.load(data_train_path)
data_val = np.load(data_val_path)
data_test = np.load(data_test_path)

data_train_shape = data_train.shape
data_val_shape = data_val.shape
data_test_shape = data_test.shape

use_generator = True
num_epochs = 10
if feature_type == 'mfcc':
    normalized = True
else:
    normalized = False

start = time.time()

if use_generator:
    train_generator = soundmodels.Generator(data_matrix1 = data_train, 
                                            normalized = normalized)
    val_generator = soundmodels.Generator(data_matrix1 = data_val,
                                          normalized = normalized)
    test_generator = soundmodels.Generator(data_matrix1 = data_test,
                                           normalized = normalized)

    train_generator.generator()
    val_generator.generator()
    test_generator.generator()
    history = scene_classifier.fit_generator(train_generator.generator(),
                                            steps_per_epoch = data_train.shape[0],
                                            epochs = num_epochs,
                                            callbacks = callbacks,
                                            validation_data = val_generator.generator(),
                                            validation_steps = data_val.shape[0])
    
    X_test, y_test = pyst.feats.separate_dependent_var(data_test)
    y_predicted = scene_classifier.predict_generator(test_generator.generator(),
                                       steps = data_test.shape[0])

else:
    X_train, y_train, scalars = pyst.feats.scale_X_y(data_train, is_train=True)
    X_val, y_val, __ = pyst.feats.scale_X_y(data_val, is_train=False, scalars=scalars)
    X_test, y_test, __ = pyst.feats.scale_X_y(data_test, is_train=False, scalars=scalars)
    
    history = scene_classifier.fit(X_train, y_train, 
                                   epochs = num_epochs, 
                                   callbacks = callbacks, 
                                   validation_data = (X_val, y_val))
    
    scene_classifier.evaluate(X_test, y_test)
    y_predicted = scene_classifier.predict(X_test)
    # which category did the model predict?
    
y_pred = np.argmax(y_predicted, axis=1)
if len(y_pred.shape) > len(y_test.shape):
    y_test = np.expand_dims(y_test, axis=1)
elif len(y_pred.shape) < len(y_test.shape):
    y_pred = np.expand_dims(y_pred, axis=1)
try:
    assert y_pred.shape == y_test.shape
except AssertionError:
    raise ValueError('The shape of prediction data {}'.format(y_pred.shape) +\
        ' does not match the `y_test` dataset {}'.format(y_test.shape) +\
            '\nThe shapes much match in order to measure accuracy.')
match = sum(y_test == y_pred)
if len(match.shape) == 1:
    match = match[0]
test_accuracy = round(match/len(y_test),4)
print('\nModel reached accuracy of {}%'.format(test_accuracy*100))

end = time.time()
total_dur_sec = round(end-start,2)
history_params = history.params
model_layers = scene_classifier.layers
model_features_dict = dict(model_path = model_path, 
                           data_train_path = data_train_path, 
                           data_val_path = data_val_path,
                           data_test_path = data_test_path,
                           data_train_shape = data_train_shape,
                           data_val_shape = data_val_shape,
                           data_test_shape = data_test_shape,
                           total_dur_sec = total_dur_sec, 
                           test_accuracy = test_accuracy,
                           optimizer = optimizer,
                           loss = loss,
                           metrics = metrics, 
                           history_params = history_params, 
                           model_layers = model_layers)
model_features_dict.update(settings_dict)

model_features_dict_path = model_dir.joinpath('info_{}.csv'.format(
    model_name))
model_features_dict_path = pyst.utils.save_dict(model_features_dict,
                                                model_features_dict_path)
