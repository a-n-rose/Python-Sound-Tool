import os
import numpy as np
import pathlib
import pysoundtool as pyst
import pysoundtool.models as soundmodels

###############################################################################

# make sure data exists:
dataset_path = pyst.utils.check_dir('./audiodata/test_classifier/', make = False) 
feature_type = 'stft'
data_train_path = dataset_path.joinpath('train_data_{}.npy'.format(feature_type))
data_val_path = dataset_path.joinpath('val_data_{}.npy'.format(feature_type))
data_test_path = dataset_path.joinpath('test_data_{}.npy'.format(feature_type))

# make sure dictionary with labels is there
dict_decode_path = dataset_path.joinpath('dict_decode.csv')

for pathway in [data_train_path, data_val_path, data_test_path, dict_decode_path]:
    if not os.path.exists(pathway):
        raise FileNotFoundError('File '+str(pathway) +\
            ' was not found or does not exist.')
    

# figure out input size of data:
# load smaller dataset
data_val = np.load(data_val_path)
# expects shape (num_audiofiles, num_frames, num_features + label_column)
# subtract the label column:
input_shape = (data_val.shape[1], data_val.shape[2] - 1, 1) 
print(input_shape)

# load dictionary with labels to find out the number of labels:
dict_decode = pyst.data.load_dict(dict_decode_path)
num_labels = len(dict_decode)

scene_classifier = soundmodels.cnn_classifier(input_shape = input_shape,
                                              num_labels = num_labels)

callbacks = soundmodels.setup_callbacks(patience=15)
scene_classifier.compile(optimizer = 'adam',
                         loss = 'sparse_categorical_crossentropy',
                         metrics = ['accuracy'])
data_train = np.load(data_train_path)
data_val = np.load(data_val_path)
data_test = np.load(data_test_path)

use_generator = False
num_epochs = 10

if use_generator:
    train_generator = soundmodels.Generator(data_matrix1 = data_train)
    val_generator = soundmodels.Generator(data_matrix1 = data_val)
    test_generator = soundmodels.Generator(data_matrix1 = data_test)

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
print('\nModel reached accuracy of {}%'.format(round(match/len(y_test)*100,2)))
