import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)

import pysoundtool as pyst
import numpy as np

###############################################################################



#feed data to models
class Generator:
    def __init__(self, data_matrix1, data_matrix2=None, feature_type=None, 
                 normalized=False, apply_log = False, adjust_shape = None,
                 batches = False, labeled_data = False):
        '''
        This generator pulls data out in sections (i.e. batch sizes). Prepared for 3 dimensional data.
        
        Note: Keras adds a dimension to input to represent the "Tensor" that 
        handles the input. This means that sometimes you have to add a 
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
        self.feat_type = feature_type
        self.batch_size = 1
        self.samples_per_epoch = data_matrix1.shape[0]
        self.number_of_batches = self.samples_per_epoch/self.batch_size
        self.counter=0
        self.datax = data_matrix1
        self.datay = data_matrix2
        self.normalized = normalized
        self.apply_log = apply_log
        if batches:
            # expects shape (num_samples, batch_size, num_frames, num_feats)
            self.batches_per_sample = self.datax.shape[1]
            self.num_frames = self.datax.shape[2]
        else:
            # expects shape (num_samples, num_frames, num_feats)
            self.batches_per_sample = None
            self.num_frames = self.datax.shape[1]
        
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
        # raise warnings if data will be significatnly adjusted 
        if self.desired_shape is not None:
            if len(self.desired_shape)+1 != len(datax.shape):
                import warnings
                message = '\nWARNING: The number of dimensions will be adjusted in the '+\
                    'generator.\nOriginal data has ' + str(len(datax.shape))+\
                        ' dimensions\nAdjusted data has ' + str(len(self.desired_shape)+1) +\
                            'dimensions'
            if self.desired_shape < datax.shape[1:]:
                import warnings
                message = '\nWARNING: Desired shape '+ str(self.desired_shape) +\
                    ' is smaller than the original data shape ' + str(datax.shape[1:])+\
                        '. Some data will therefore be removed, NOT ZEROPADDED.'

    def generator(self):
        while 1:

            # will be size (batch_size, num_frames, num_features)
            batch_x = self.datax[self.counter] 
            batch_y = self.datay[self.counter]
            # TODO: is there a difference between taking log of stft before 
            # or after normalization?
            if not self.normalized:
                # need to take power - complex values in stft
                if self.feat_type is not None and self.feat_type == 'stft':
                    # take power of absoulte value of stft
                    batch_x = np.abs(batch_x)**2
                    if self.labels is None:
                        batch_y = np.abs(batch_y)**2
                batch_x = (batch_x - np.min(batch_x)) / \
                    (np.max(batch_x) - np.min(batch_x))
                # don't need to touch labels
                if self.labels is None: 
                    batch_y = (batch_y - np.min(batch_y)) / \
                        (np.max(batch_y) - np.min(batch_y))
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
                batch_x = pyst.data.adjust_data_shape(batch_x, self.desired_shape)
                
                if self.labels is None:
                    batch_y = pyst.data.adjust_data_shape(batch_y, self.desired_shape)
            
            ## TODO remove the first reshape step?
            ## why is this necessary?
            #X_batch = batch_x.reshape((batch_x.shape[0], batch_x.shape[1],
                                     #batch_x.shape[2])) 
            #if self.labels is None:
                #y_batch = batch_y.reshape((batch_y.shape[0], batch_y.shape[1],
                                        #batch_y.shape[2])) 
            # add tensor dimension
            X_batch = batch_x.reshape(batch_x.shape+(1,))
            y_batch = batch_y.reshape(batch_y.shape+(1,))
            
            if len(X_batch.shape) == 3:
                X_batch = X_batch.reshape((1,)+X_batch.shape)
                y_batch = y_batch.reshape((1,)+y_batch.shape)
            
            #send the batched and reshaped data to model
            self.counter += 1
            yield X_batch, y_batch

            #restart counter to yeild data in the next epoch as well
            if self.counter >= self.number_of_batches:
                self.counter = 0
