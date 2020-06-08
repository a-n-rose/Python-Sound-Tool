import os
import numpy as np
import pathlib
import pysoundtool as pyst
import pysoundtool.models as soundmodels

###############################################################################

# make sure data exists:
dataset_path = pyst.utils.check_dir('./audiodata/test_classifier/', make = False) 
data_train_path = dataset_path.joinpath('train_data_fbank.npy')
data_val_path = dataset_path.joinpath('val_data_fbank.npy')
data_test_path = dataset_path.joinpath('test_data_fbank.npy')

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

use_generator = True
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

else:
    X_train, y_train, scalars = pyst.feats.scale_X_y(data_train, train=True)
    X_val, y_val, __ = pyst.feats.scale_X_y(data_val, train=False, scalars=scalars)
    X_test, y_test, __ = pyst.feats.scale_X_y(data_test, train=False, scalars=scalars)
    
    history = scene_classifier.fit(X_train, y_train, 
                                   epochs = num_epochs, 
                                   callbacks = callbacks, 
                                   validation_data = (X_val, y_val))

