

###############################################################################

import os, sys
import inspect
import pathlib
import numpy as np
# for building and training models
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, normalize
 
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)

import pysoundtool as pyst


class SoundClassifier:
    '''Build a mobile compatible CNN to classify noise / acoustic scene
    
    References
    ----------
    A. Sehgal and N. Kehtarnavaz, "A Convolutional Neural Network 
    Smartphone App for Real-Time Voice Activity Detection," in IEEE Access, 
    vol. 6, pp. 9017-9026, 2018. 
    
    Attributes
    ----------
    model_path : str or pathlib.PosixPath
        The full path to the model to be trained or pretrained model. If `str`, 
        will be converted to pathlib.PosixPath object. The path will end with 
        extension '.h5'.
    models_dir : str or pathlib.PosixPath
        The directory of where the model is stored.
    features_dir : str or pathlib.PosixPath
        The directory where relevant data files are located. Data files are 
        expected to be in .npy format.
    encoded_labels_path : str
        A path to .csv file with dictionary containing the encoded labels for 
        training data.
    feature_session : None
        Currently doesn't seem to be used within this class. TODO remove. 
    data_settings : dict
        Loaded dictionary containing data extraction settings, pulled from
        `features_dir`.
    modelsettings_path : str or pathlib.PosixPath
        Path to where model training settings are to be saved.
    '''
    def __init__(self,
                 modelname, models_dir,
                 features_dir, encoded_labels_path, feature_session,
                 modelsettings_path, newmodel=True):
        self.model_path = self.create_model_path(
            models_dir, modelname, newmodel=newmodel)
        self.models_dir = self.model_path.parent
        self.features_dir = features_dir
        self.encoded_labels_path = encoded_labels_path
        self.feature_session = feature_session
        self.data_settings = pyst.paths.load_dict(
            pyst.paths.load_settings_file(self.features_dir))
        self.modelsettings_path = modelsettings_path
        '''Initializes class for training a sound classifier via CNN 
        '''

    def create_model_path(self, models_dir, modelname, newmodel=True):
        '''Creates directory and checks validity of model pathway.
        
        Parameters
        ----------
        models_dir : str or pathlib.PosixPath
            TODO make this optional. 
            
        modelname : str or pathlib.PosixPath
        
        newmodel : bool 
            If True and model path already exists, raises FileExistsError. 
            (default True)
            
        Returns
        -------
        model_path : pathlib.PosixPath
            The verified pathway to the model.
        '''
        if isinstance(modelname, str) and '.h5' not in modelname:
            modelname += '.h5'
        if not models_dir:
            if not isinstance(modelname, pathlib.PosixPath):
                modelname = pathlib.Path(modelname)
            model_path = modelname
            return model_path

        if not isinstance(models_dir, pathlib.PosixPath):
            models_dir = pathlib.Path(models_dir)
        model_path = models_dir.joinpath(modelname)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        if os.path.exists(model_path) and newmodel:
            raise FileExistsError('A model already exists at this path:\
                \n{}\nMove or rename.'.format(model_path))
        return model_path

    def load_train_val_data(self, test_model=False):
        '''Loads training data and trains model
        
        Expects at least train and validation data, test data optional. If 
        no test data, set `test_model` as False.
        
        Parameters
        ----------
        test_model : bool 
            If False, model only trains with train and validation data. If 
            True, model will train with train and validation data, and test 
            with test data. (default False)
            
        Returns
        -------
        None
        '''
        self.train_path = None
        self.val_path = None
        self.test_path = None
        data_files = list(self.features_dir.glob('**/*.npy'))
        for filename in data_files:
            if 'train' in filename.parts[-1]:
                self.train_path = filename
            elif 'val' in filename.parts[-1]:
                self.val_path = filename
            elif 'test' in filename.parts[-1]:
                self.test_path = filename
        if not self.train_path:
            raise FileNotFoundError(
                'No file for {} data found'.format('train'))
        if not self.val_path:
            raise FileNotFoundError(
                'No file for {} data found'.format('validation'))
        if test_model:
            if not self.test_path:
                raise FileNotFoundError(
                    'No file for {} data found'.format('test'))

    def load_labels(self):
        '''Loads the encoded labels for the training data.
        
        Parameters
        ----------
        self : class instance
            Contains the attribute `encoded_labels_path` which is used to 
            load the .csv file containing the encoded labels dictionary.
            
        Returns
        -------
        labels_encoded : dict { encoded_label : label }
            The dictionary containing the encoded labels and labels for training
            data.
        '''
        labels_encoded = pyst.paths.load_dict(self.encoded_labels_path)
        return labels_encoded

    # Continue documentation here
    def set_model_params(self,  color_scale=1, num_layers=None, feature_maps=None,
                         kernel_size=None, halve_feature_maps=True, strides=2,
                         dense_hidden_units=100, epochs=100, activation_layer='relu',
                         activation_output='softmax', dropout=0.25):
        '''Sets parameters of model to class attributes to save in model settings.
        
        Parameters
        ----------
        color_scale : int 
            Set to 1, gray scale; set to 3, RGB scale; set to 4, TODO: find out what happens. (default 1) 
        num_layers : int, optional
            Number of convolutional neural network layers. (default 3)
        feature_maps : TODO 
        '''
        self.num_features = pyst.tools.make_number(
            self.data_settings['num_columns'])
        self.feature_sets = pyst.tools.make_number(
            self.data_settings['feature_sets'])
        self.num_env_images = pyst.tools.make_number(
            self.data_settings['num_images_per_audiofile'])
        if num_layers is None:
            self.num_layers = 3
        else:
            self.num_layers = num_layers
        if feature_maps is None:
            feature_maps = []
            prev_feature_map = 0
            for layer in range(self.num_layers):
                if layer == 0:
                    feature_maps.append(self.num_features)
                    prev_feature_map += self.num_features
                else:
                    if halve_feature_maps:
                        curr_feature_map = int(prev_feature_map/2)
                    elif halve_feature_maps is None:
                        curr_feature_map = prev_feature_map
                    elif halve_feature_maps is False:
                        curr_feature_map = int(prev_feature_map*2)
                    feature_maps.append(curr_feature_map)
                    prev_feature_map = curr_feature_map
        if kernel_size is None:
            kernel_size = []
            for layer in range(self.num_layers):
                kernel_size.append((3, 3))
        self.kernel_size = kernel_size
        self.feature_maps = feature_maps
        self.strides = strides
        self.dense_hidden_units = dense_hidden_units
        self.input_shape = (self.feature_sets, self.num_features, color_scale)
        self.num_labels = len(self.load_labels())
        self.activation_layer = activation_layer
        self.dropout = dropout
        self.activation_output = activation_output
        self.epochs = epochs
        return None

    def build_cnn_model(self):
        '''set up model architecture using Keras Sequential() class
        '''
        model = Sequential()
        assert type(self.feature_maps) == type(self.kernel_size)
        if isinstance(self.feature_maps, list):
            assert len(self.feature_maps) == len(self.kernel_size)
        if isinstance(self.feature_maps, list):
            for i, featmap in enumerate(self.feature_maps):
                if i == 0:
                    model.add(Conv2D(featmap,
                                     self.kernel_size[i],
                                     strides=self.strides,
                                     activation=self.activation_layer,
                                     input_shape=self.input_shape))
                else:
                    model.add(Conv2D(featmap,
                                     self.kernel_size[i],
                                     strides=self.strides,
                                     activation=self.activation_layer))
        else:
            model.add(Conv2D(self.feature_maps,
                             self.kernel_size,
                             strides=self.strides,
                             activation=self.activation_layer,
                             input_shape=self.input_shape))
        if self.dense_hidden_units is not None:
            model.add(Dense(self.dense_hidden_units))
        if self.dropout is not None:
            model.add(Dropout(self.dropout))
        model.add(Flatten())
        model.add(Dense(self.num_labels, activation=self.activation_output))
        return model

    def compile_model(self, model, optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']):
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def set_up_callbacks(self, early_stop=True, patience=15, csv_log=True,
                         csv_filename=None, save_bestmodel=True, best_modelname=None,
                         monitor='val_loss', verbose=1, save_best_only=True, mode='min'):
        callbacks = []
        if early_stop:
            early_stopping_callback = EarlyStopping(
                monitor=monitor,
                patience=patience)
            callbacks.append(early_stopping_callback)
        if csv_log:
            if csv_filename is None:
                csv_filename = self.models_dir.joinpath('log.csv')
            csv_logging = CSVLogger(filename=csv_filename)
            callbacks.append(csv_logging)
        if save_bestmodel:
            if best_modelname is None:
                # turn Path object to string object
                modelname = self.model_path.parts[-1]
                best_modelname = str(self.models_dir.joinpath(
                    'bestmodel_{}'.format(modelname)))
            checkpoint_callback = ModelCheckpoint(best_modelname, monitor=monitor,
                                                  verbose=verbose,
                                                  save_best_only=save_best_only,
                                                  mode=mode)
            callbacks.append(checkpoint_callback)
        if callbacks:
            return callbacks
        return None

    def build_cnn_reduced(self):
        '''Reduces layers of CNN until the model can be built

        If the number of filters for 'mfcc' or 'fbank' is in the lower range
        (i.e. 13 or so), this causes issues with the default settings of
        the cnn architecture. The architecture was built with at least 40
        filters being applied during feature extraction. To deal with this
        problem, the number of CNN layers are reduced. 
        '''
        if self.num_layers > 1:
            print('\nReducing number of CNN layers from {} to {}\n'.format(
                self.num_layers, self.num_layers-1))
            self.set_model_params(num_layers=self.num_layers-1)
            try:
                model = self.build_cnn_model()
                print('\nReducing number of CNN layers worked!')
                print('Now training model..\n')
            except ValueError:
                model = self.reduce_build_cnn()
        else:
            raise ValueError(
                '\nModel cannot be built given data and parameter settings')
        return model

    def train_scene_classifier(self):
        '''loads all data and trains model

        advantage of loading all data is it is easier to scale and
        normalize. 
        '''
        train_data = np.load(self.train_path)
        val_data = np.load(self.val_path)
        X_train, y_train, scalars = scale_X_y(train_data)
        X_val, y_val, __ = scale_X_y(val_data,
                                       is_train=False,
                                       scalars=scalars)
        self.set_model_params(halve_feature_maps=True)
        callbacks = self.set_up_callbacks()
        try:
            model = self.build_cnn_model()
        except ValueError as e:
            print('Warning:'.upper())
            print('\nError building model.')
            print(e)
            print('\nProblem likely due to a small number of feature filters extracted')
            print('To avoid this problem, extract features with at least 20 filters')
            print('i.e. mfcc or fbank features with 20+ filters')
            model = self.build_cnn_reduced()
        model = self.compile_model(model)
        self.save_class_settings(overwrite=False)
        history = model.fit(
            X_train, y_train,
            epochs=self.epochs,
            callbacks=callbacks,
            validation_data=(X_val, y_val),
        )
        if self.test_path:
            test_data = np.load(self.test_path)
            X_test, y_test, __ = scale_X_y(
                test_data, is_train=False, scalars=scalars)
            score = model.evaluate(X_test, y_test)
            loss = round(score[0], 2)
            acc = round(score[1]*100, 3)
            self.test_loss = loss
            self.test_acc = acc
        # turn Path object to string object
        model.save(str(self.model_path))
        self.save_class_settings(overwrite=True)
        return None

    def save_class_settings(self, overwrite=False):
        '''saves class settings to dictionary
        '''
        class_settings = self.__dict__
        pyst.paths.save_dict(class_settings, self.modelsettings_path,
                          replace=overwrite)
        return None


class ClassifySound:
    '''Takes new audio and classifies it with classifier.
    '''
    def __init__(self, sounddata, filter_class, feature_class, model_class):
        self.sounddata = sounddata
        self.feature_settings = pyst.paths.load_dict(
            pyst.paths.load_settings_file(model_class.features_dir))
        self.modelsettings = pyst.paths.load_dict(model_class.modelsettings_path)
        self.model_path = model_class.model_path
        self.models_dir = model_class.models_dir
        self.decode_label = pyst.paths.load_dict(filter_class.labels_encoded_path)
        self.label_encoded = None
        self.label = None
        self.average_power_dir = filter_class.powspec_path

    def check_for_best_model(self):
        models_list = self.models_dir.glob('**/*.h5')
        best_model = None
        for model_name in models_list:
            if 'best' in model_name.stem:
                best_model = model_name
        return best_model

    def load_modelsettings(self):
        best_model = self.check_for_best_model()
        if best_model:
            self.model_path = best_model
        model = load_model(str(self.model_path))
        input_shape = self.modelsettings['input_shape']
        chars = '()'
        for char in chars:
            input_shape = input_shape.replace(char, '')
        input_shape = input_shape.split(',')
        for i, item in enumerate(input_shape):
            input_shape[i] = int(item)
            input_shape_model = (1,)+tuple(input_shape)
        return model, input_shape_model

    def extract_feats(self):
        prev_featextraction_settings_dict = self.feature_settings
        get_feats = pyst.feats.getfeatsettings(
            prev_featextraction_settings_dict)
        dur_sec = get_feats.training_segment_ms/1000
        feats = get_feats.extractfeats(self.sounddata, dur_sec=dur_sec,
                                       augment_data=False)
        return feats

    def get_label(self):
        feats = self.extract_feats()
        model, input_shape = self.load_modelsettings()
        feats = feats.reshape(feats.shape + (1,))
        try:
            assert input_shape == feats.shape
        except AssertionError:
            print('input shape: ', input_shape)
            print('feats shape: ', feats.shape)
        pred = model.predict(feats)
        self.label_encoded = np.argmax(pred[0])
        # load label
        self.label = self.decode_label[str(self.label_encoded)]
        return self.label, self.label_encoded

    def load_assigned_avepower(self, label_encoded, raw_samples=False):
        ave_pow_list = list(self.average_power_dir.glob('**/*.npy'))
        if ave_pow_list:
            for item in ave_pow_list:
                num_digits = len(str(label_encoded))
                file_stem = item.stem
                if file_stem[-num_digits:] == str(label_encoded):
                    return item
        print('no average values for this class found')
        return None

def buildclassifier(filter_class):
    scene = SoundClassifier(modelname=filter_class.modelname,
                            models_dir=filter_class.model_dir,
                            features_dir=filter_class.features_dir,
                            encoded_labels_path=filter_class.labels_encoded_path,
                            feature_session=filter_class.features_dir.parts[-1],
                            modelsettings_path=filter_class.model_settings_path,
                            newmodel=True
                            )
    scene.load_train_val_data(test_model=False)
    scene.train_scene_classifier()
    return scene

def loadclassifier(filter_class):
    scene = SoundClassifier(modelname=filter_class.model,
                            models_dir=None, #loaded from modelname
                            features_dir=filter_class.features_dir,
                            encoded_labels_path=filter_class.labels_encoded_path,
                            feature_session=filter_class.feature_dirname,
                            modelsettings_path=filter_class.model_settings_path,
                            newmodel=False
                            )
    return scene


def buildmodel(filter_class):
    scene = SoundClassifier(modelname=filter_class.modelname,
                            models_dir=filter_class.model_dir,
                            features_dir=filter_class.features_dir,
                            encoded_labels_path=filter_class.labels_encoded_path,
                            feature_session=filter_class.features_dir.parts[-1],
                            modelsettings_path=filter_class.model_settings_path,
                            newmodel=True,
                            model_type = filter_class.model_type
                            )
    scene.load_train_val_data(test_model=False)
    scene.train_scene_classifier()
    return scene

def loadmodel(filter_class):
    '''need to generalize loading of models.
    '''
    sm = SoundModel(modelname=filter_class.model,
                            models_dir=None, #loaded from modelname
                            features_dir=filter_class.features_dir,
                            encoded_labels_path=None,
                            feature_session=filter_class.feature_dirname,
                            modelsettings_path=filter_class.model_settings_path,
                            newmodel=False,
                            model_type = filter_class.model_type
                            )
    return sm

def scale_X_y(matrix, is_train=True, scalars=None):
    '''Separates and scales data into X and y arrays. Adds dimension for keras.
    
    Assumes the last column of the last dimension is the y or label data.
    
    Parameters
    ----------
    matrix : np.ndarray [size = (num_samples, num_frames, num_features)]
        Matrix with X and y data
    is_train : bool
        Relevant for the `scalars` parameter. If the data is training
        data (i.e. True), the `scalars` will be created. If the data
        is test data (i.e. False), the function expects `scalars` to 
        be provided. (default True)
    scalars : dict, optional
        Dictionary with scalars to be applied to non-training data.
        
    Returns
    -------
    X : np.ndarray [size = (num_sampls, num_frames, num_features-1, 1)]
        Scaled features with extra dimension
    y : np.ndarray [size = (num_samples, 1, 1)]
        Scaled independent variable with extra dimension
    scalars : dict
        The scalars either created or previously loaded.
    '''
    X, y = pyst.feats.separate_dependent_var(matrix)
    if is_train:
        scalars = {}
    elif scalars is None:
        raise TypeError('If non-train data, `scalars` cannot be of type None.')
    if len(X.shape) != 3:
        raise ValueError('Expected 3d input, not input of shape {}.'.format(
            matrix.shape))
    for j in range(X.shape[2]):
        if is_train:
            scalars[j] = StandardScaler()
            X[:, :, j] = scalars[j].fit_transform(X[:, :, j])
        else:
            X[:, :, j] = scalars[j].transform(X[:, :, j])
        X[:, :, j] = normalize(X[:, :, j])
    # Keras needs an extra dimension as a tensor / holder of data
    X = pyst.feats.add_tensor(X)
    y = pyst.feats.add_tensor(y)
    return X, y, scalars
