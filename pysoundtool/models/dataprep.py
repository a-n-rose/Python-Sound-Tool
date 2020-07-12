'''The models.dataprep module covers functionality for feeding features to models.
'''

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)

import numpy as np
import random
import pysoundtool as pyst


###############################################################################



#feed data to models
class Generator:
    def __init__(self, data_matrix1, data_matrix2=None, 
                 normalized=False, apply_log = False, adjust_shape = None,
                 labeled_data = False, add_tensor_last = None, gray2color = False):
        '''
        This generator pulls data out in sections (i.e. batch sizes). Prepared for 3 dimensional data.
        
        Note: Keras adds a dimension to input to represent the "Tensor" that 
        #handles the input. This means that sometimes you have to add a 
        shape of (1,) to the shape of the data. 
        
        Parameters
        ----------
        data_matrix1 : np.ndarray [size=(num_samples, batch_size, num_frames, num_features) or (num_samples, num_frames, num_features+label_column)]
            The training data. This can contain the feature and label data or 
            just the input feature data. 
        data_matrix2 : np.ndarray [size = (num_samples, ) `data_matrix1`.shape], optional
            Either label data for `data_matrix1` or, for example, the clean 
            version of `data_matrix1` if training an autoencoder. (default None)
        normalized : bool 
            If True, the data has already been normalized and won't be normalized
            by the generator. (default False)
        apply_log : bool 
            If True, log will be applied to the data.
        adjust_shape : int or tuple, optional
            The desired number of features or shape of data to feed a neural network.
            If type int, only the last column of features will be adjusted (zeropadded
            or limited). If tuple, the entire data shape will be adjusted (all columns). 
            If the int or shape is larger than that of the data provided, data will 
            be zeropadded. If the int or shape is smaller, the data will be restricted.
        batches : bool 
            If the data is expected to be separated in batches. If True, data should have 
            shape (num_samples, batch_size, num_frames, num_features); if False, data
            should have shape (num_samples, num_frames, num_features+label_column).
            (default False)
        '''
        self.batch_size = 1
        self.samples_per_epoch = data_matrix1.shape[0]
        self.number_of_batches = self.samples_per_epoch/self.batch_size
        self.counter=0
        self.datax = data_matrix1
        self.datay = data_matrix2
        self.normalized = normalized
        self.apply_log = apply_log
        self.add_tensor_last = add_tensor_last
        self.gray2color = gray2color # if need to change grayscale data to rgb
        if len(self.datax.shape) == 4:
            # expects shape (num_samples, batch_size, num_frames, num_feats)
            self.batches_per_sample = self.datax.shape[1]
            self.num_frames = self.datax.shape[2]
        elif len(self.datax.shape) == 3:
            # expects shape (num_samples, num_frames, num_feats)
            self.batches_per_sample = None
            self.num_frames = self.datax.shape[1]
        else:
            raise ValueError('Expected 4 or 3 dimensional data, not data '+\
                'with shape {}.'.format(self.datax.shape))
        if self.datay is None:
            # separate the label from the feature data
            self.datax, self.datay = pyst.feats.separate_dependent_var(self.datax)
            # assumes last column of features is the label column
            self.num_feats = self.datax.shape[-1] 
            self.labels = True
        else:
            self.labels = None
        if labeled_data:
            self.labels = True
        if adjust_shape is not None:
            if isinstance(adjust_shape, int):
                if self.batches_per_sample is not None:
                    self.desired_shape = (self.batches_per_sample, 
                                        self.num_frames,
                                        adjust_shape)
                else:
                    self.desired_shape = (self.num_frames,
                                        adjust_shape)
            elif isinstance(adjust_shape, tuple):
                self.desired_shape = adjust_shape
            else:
                self.desired_shape = None
        else:
            self.desired_shape = None
        # raise warnings if data will be significantly adjusted 
        # TODO test this or delete it
        if self.desired_shape is not None:
            if len(self.desired_shape)+1 != len(self.datax.shape):
                import warnings
                message = '\nWARNING: The number of dimensions will be adjusted in the '+\
                    'generator.\nOriginal data has ' + str(len(self.datax.shape))+\
                        ' dimensions\nAdjusted data has ' + str(len(self.desired_shape)+1) +\
                            'dimensions'
            if self.desired_shape < self.datax.shape[1:]:
                import warnings
                message = '\nWARNING: Desired shape '+ str(self.desired_shape) +\
                    ' is smaller than the original data shape ' + str(self.datax.shape[1:])+\
                        '. Some data will therefore be removed, NOT ZEROPADDED.'

    def generator(self):
        '''Shapes, norms, and feeds data depending on labeled or non-labeled data.
        '''
        while 1:

            # will be size (batch_size, num_frames, num_features)
            batch_x = self.datax[self.counter] 
            batch_y = self.datay[self.counter]
            # TODO: is there a difference between taking log of stft before 
            # or after normalization?
            if not self.normalized or self.datax.dtype == np.complex_:
                # if complex data, power spectrum will be extracted
                # power spectrum = np.abs(complex_data)**2
                batch_x = pyst.feats.normalize(batch_x)
                if self.labels is None:
                    batch_y = pyst.feats.normalize(batch_y)
            # apply log if specified
            if self.apply_log: 
                batch_x = np.log(np.abs(batch_x))
                # don't need to touch label data
                if self.labels is None:
                    batch_y = np.log(np.abs(batch_y))

            # TODO test
            # if need greater number of features --> zero padding
            # could this be applied to including both narrowband and wideband data?
            if self.desired_shape is not None:
                batch_x = pyst.feats.adjust_shape(batch_x, self.desired_shape, change_dims=True)
                
                if self.labels is None:
                    batch_y = pyst.feats.adjust_shape(batch_y, self.desired_shape, change_dims=True)
            
            if self.gray2color:
                # expects colorscale to be last column.
                # will copy first channel into the other (assumed empty) channels.
                # the empty channels were created in pyst.feats.adjust_shape
                batch_x = pyst.feats.grayscale2color(batch_x, 
                                                     colorscale = batch_x.shape[-1])
                if self.labels is None:
                    batch_y = pyst.feats.gray2color(batch_y, 
                                                    colorscale = batch_y.shape[-1])
            ## add tensor dimension
            if self.add_tensor_last is True:
                # e.g. for some conv model
                X_batch = batch_x.reshape(batch_x.shape+(1,))
                y_batch = batch_y.reshape(batch_y.shape+(1,))
            elif self.add_tensor_last is False:
                # e.g. for some lstm models
                X_batch = batch_x.reshape((1,)+batch_x.shape)
                y_batch = batch_y.reshape((1,)+batch_y.shape)
            else:
                X_batch = batch_x
                y_batch = batch_y
            
            if len(X_batch.shape) == 3:
                # needs to be 4 dimensions / have extra tensor
                X_batch = X_batch.reshape((1,)+X_batch.shape)
                y_batch = y_batch.reshape((1,)+y_batch.shape)
            
            #send the batched and reshaped data to model
            self.counter += 1
            yield X_batch, y_batch

            #restart counter to yeild data in the next epoch as well
            if self.counter >= self.number_of_batches:
                self.counter = 0



class GeneratorFeatExtraction:
    def __init__(self, datalist, datalist2 = None, model_name = None,  
                 normalize = True, apply_log = False, randomize = False, random_seed = 40,
                 context_window = None, input_shape = None, batch_size = 1,
                 add_tensor_last = True, gray2color = False, visualize = False,
                 vis_every_n_items = 50, save_visuals_path = None, **kwargs):
        '''
        Do not add extra tensor dimensions to expected input_shape.
        
        Parameters
        ----------
        datalist : list 
            List of audiofile pathways for feature extraction and model training. 
            If labeled data, expects pathway and encoded label (i.e. int) to be paired 
            together in a tuple (a list of tuples).
        
        datalist2 : list, optional
            List of audiofile pathways or labels for feature extraction and model training. 
            This list might contain clean versions of `datalist`. These will be assigned 
            as the 'label' or expected output of the input features.
        
        **kwargs : additional keyword arguments
            Keyword arguments for pysoundtool.feats.get_feats
        '''
        if sample_length is None and input_shape is None and 'dur_sec' not in kwargs.keys():
            raise ValueError('No information pertaining to amount of audio data '+\
                'to be extracted is supplied. Please specify `sample_length`, '+\
                    '`input_shape`, or `dur_sec`.')
        if randomize:
            if random_seed is not None:
                random.seed(random_seed)
            random.shuffle(datalist)
            if datalist2 is not None:
                if random_seed is not None:
                    random.seed(random_seed)
                else:
                    raise ValueError('If two audiolists lists are provided, ' +\
                        'a `random_seed` is necessary to ensure they still match '+\
                            'post randomization.')
                random.shuffle(datalist2)
                
        self.batch_size = batch_size
        self.samples_per_epoch = len(datalist)
        self.number_of_batches = self.samples_per_epoch//batch_size
        self.counter = 0
        self.audiolist = datalist
        self.audiolist2 = datalist2
        self.normalize = normalize
        self.input_shape = input_shape
        self.add_tensor_last = add_tensor_last
        self.gray2color = gray2color
        self.visualize = visualize
        self.vis_every_n_items = vis_every_n_items
        self.kwargs = kwargs
        
        # Ensure `feature_type` and `sr` are provided in **kwargs
        try:
            f = kwargs['feature_type']
        except KeyError:
            raise KeyError('Feature type not indicated. '+\
                'Please set `feature_type` to one of the following: '+\
                '\nfbank\nstft\npowspec\nsignal\nmfcc\n')
        try: 
            sr = kwargs['sr']
        except KeyError:
            raise KeyError('Sample rate is not indicated. '+\
                'Please set `sr` (e.g. sr = 16000)')
        
        # if these are present, great. If not, set to None
        # useful for plotting features with time in seconds
        try: 
            win_size_ms = kwargs['win_size_ms']
        except KeyError:
            kwargs['win_size_ms'] = None
        try: 
            percent_overlap = kwargs['percent_overlap']
        except KeyError:
            kwargs['percent_overlap'] = None
        
        def generator(self):
            '''Extracts features and feeds them to model according to `input_shape`.
            '''
            while 1:
                audioinfo = self.audiolist[self.counter]
                # does the list contain label audiofile pairs?
                if isinstance(audioinfo, tuple):
                    if len(tuple) != 2:
                        raise ValueError('Expected tuple containing audio file path and label. '+\
                            'Instead received tuple of length: \n{}'.format(len(tuple)))
                    # if label is a string digit, int, or float - turn to int
                    if isinstance(tuple[0], int) or isinstance(tuple[0], float) or \
                        isinstance(tuple[0], str) and tuple[0].isdigit():
                        label = int(tuple[0])
                        audiopath = tuple[1]
                    elif isinstance(tuple[1], int) or isinstance(tuple[1], float) or \
                        isinstance(tuple[1], str) and tuple[1].isdigit():
                        label = int(tuple[1])
                        audiopath = tuple[1]
                    else:
                        raise ValueError('Expected tuple to contain an integer label '+\
                            'and audio pathway. Received instead tuple with types '+\
                                '{} and {}.'.format(type(tuple[0]), type(tuple[1])))
                # otherwise list of audiofiles
                else:
                    audiopath = audioinfo
                    label = None
                if self.audiolist2 is not None:
                    # expects audiolist2 to be either integer labels or audiofile pathways
                    audioinfo2 = self.audiolist2[self.counter]
                    if isinstance(audioinfo2, int) or isinstance(audioinfo2, str) and \
                        audioinfo2.isdigit():
                            if label is None:
                                label = audioinfo2
                            else:
                                if label == int(audioinfo2):
                                    pass
                                else:
                                    raise ValueError('Provided conflicting labels for '+\
                                        'current audiofile: {}.'.format(audiopath) +\
                                            '\nReceived both label {} and {} .'.format(
                                                label, int(audioinfo2)))
                    else:
                        audiopath2 = audioinfo2
                if label is not None:
                    labeled_data = True
                else:
                    labeled_data = False
                
                # extract features
                # will be shape (num_frames, num_features)
                feats = pyst.feats.get_feats(audiopath, self.kwargs)
                if apply_log:
                    # TODO test
                    if feats[0].any() < 0:
                            feats = np.abs(feats)
                    feats = np.log(feats)
                if normalize:
                    feats = pyst.feats.normalize(feats)
                if not labeled_data and self.audiolist2 is not None:
                    feats2 = pyst.feats.get_feats(audiopath2, self.kwargs)
                    if apply_log:
                        # TODO test
                        if feats2[0].any() < 0:
                                feats2 = np.abs(feats2)
                        feats2 = np.log(feats2)
                    if normalize:
                        feats2 = pyst.feats.normalize(feats2)
                else:
                    feats2 = None
                    
                # Save visuals if desired
                if visualize:
                    if self.counter % self.vis_every_n_items == 0:
                        if save_visuals_path is None:
                            save_visuals_path = './images_label{}_training_{}_{}.png'.format(
                                label, model_name, pyst.utils.get_date())
                        feature_type = kwargs['feature_type']
                        sr = kwargs['sr']
                        win_size_ms = kwargs['win_size_ms']
                        percent_overlap = kwargs['percent_overlap']
                        if 'stft' in feature_type or 'powspec' in feature_type or 'fbank' \
                            in feature_type:
                                energy_scale = 'power_to_db'
                        else:
                            energy_scale = None
                        pyst.feats.plot(feats, feature_type, sr = sr, 
                                        win_size_ms = win_size_ms, percent_overlap = percent_overlap,
                                        energy_scale = energy_scale, save_pic = True, 
                                        name4pic = save_visuals_path,
                                        title = 'Label {} {} features \n'.format(label, feature_type)+\
                                            '(item {})'.format(self.counter))
                        if feats2 is not None:
                            # add '_2' to pathway
                            p = pyst.utils.string2pathlib(save_visuals_path)
                            p2 = p.name.stem
                            save_visuals_path2 = p.parent.joinpath(p2+'_2'+p.name.suffix)
                            pyst.feats.plot(feats2, feature_type, sr = sr, 
                                            win_size_ms = win_size_ms, percent_overlap = percent_overlap,
                                            energy_scale = energy_scale, save_pic = True, 
                                            name4pic = save_visuals_path2,
                                            title = 'Output {} features \n'.format(
                                                label, feature_type)+\
                                                '(item {})'.format(self.counter))
                                            
                # reshape to input shape. Will be zeropadded or limited to this shape.
                if self.input_shape is not None:
                    if len(input_shape) != len(feats.shape):
                        change_dims = True
                    else:
                        change_dims = False
                    feats = pyst.feats.adjust_shape(feats, input_shape, change_dims = change_dims)
                    if feats2 is not None:
                        feats2 = pyst.feats.adjust_shape(feats2, input_shape, change_dims = change_dims)
                        
                # grayscale 2 color 
                # assumes already zeropadded with new channels, channels last
                if self.gray2color:
                    feats = pyst.feats.grayscale2color(feats, colorscale = feats.shape[-1])
                    if feats2 is not None:
                        feats2 = pyst.feats.grayscale2color(feats2, colorscale = feats2.shape[-1])
                
                # prepare data to be fed to network:
                X_batch = feats
                if labeled_data:
                    y_batch = np.array(label)
                elif feats2 is not None:
                    y_batch = feats2
                else:
                    raise ValueError('No independent variable provided.')
                
                # add tensor dimension
                if self.add_tensor_last is True:
                    # e.g. for some conv model
                    X_batch = X_batch.reshape(X_batch.shape+(1,))
                    y_batch = y_batch.reshape(y_batch.shape+(1,))
                elif self.add_tensor_last is False:
                    # e.g. for some lstm models
                    X_batch = X_batch.reshape((1,)+X_batch.shape)
                    y_batch = y_batch.reshape((1,)+y_batch.shape)
                else:
                    X_batch = X_batch
                    y_batch = y_batch
                
                self.counter += 1
                yield X_batch, y_batch 
                
                #restart counter to yeild data in the next epoch as well
                if self.counter >= self.number_of_batches:
                    self.counter = 0
