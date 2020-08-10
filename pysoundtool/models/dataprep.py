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
import pysoundtool as pyso
import librosa


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
        self.counter = 0
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
            self.datax, self.datay = pyso.feats.separate_dependent_var(self.datax)
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
                batch_x = pyso.feats.normalize(batch_x)
                if self.labels is None:
                    batch_y = pyso.feats.normalize(batch_y)
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
                batch_x = pyso.feats.adjust_shape(batch_x, self.desired_shape, change_dims=True)
                
                if self.labels is None:
                    batch_y = pyso.feats.adjust_shape(batch_y, self.desired_shape, change_dims=True)
            
            if self.gray2color:
                # expects colorscale to be last column.
                # will copy first channel into the other (assumed empty) channels.
                # the empty channels were created in pyso.feats.adjust_shape
                batch_x = pyso.feats.grayscale2color(batch_x, 
                                                     colorscale = batch_x.shape[-1])
                if self.labels is None:
                    batch_y = pyso.feats.gray2color(batch_y, 
                                                    colorscale = batch_y.shape[-1])
            ## add tensor dimension
            if self.add_tensor_last is True:
                # e.g. for some conv model
                X_batch = batch_x.reshape(batch_x.shape + (1,))
                y_batch = batch_y.reshape(batch_y.shape + (1,))
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
                 normalize = True, apply_log = False, randomize = False,
                 random_seed=None, input_shape = None, batch_size = 1,
                 add_tensor_last = True, add_tensor_first = False,
                 gray2color = False, visualize = False,
                 vis_every_n_items = 50, visuals_dir = None,
                 decode_dict = None, dataset='train', 
                 augment_dict = None, label_silence = False,
                 vad_start_end = False,**kwargs):
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
            
        vad_start_end : bool 
            If True, VAD will be applied only to the beginng and end of the signal, to clip off
            the silences. If False, VAD will be applied to the entire signal; however, this is
            potentially finicky.
        
        **kwargs : additional keyword arguments
            Keyword arguments for pysoundtool.feats.get_feats
        '''
        if input_shape is None and 'dur_sec' not in kwargs.keys():
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
        
        self.dataset = dataset
        self.label_silence = label_silence
        self.vad_start_end = vad_start_end
        self.model_name = model_name
        self.batch_size = batch_size
        self.samples_per_epoch = len(datalist)
        self.number_of_batches = self.samples_per_epoch//batch_size
        self.counter = 0
        self.audiolist = datalist
        self.audiolist2 = datalist2
        self.normalize = normalize
        self.apply_log = apply_log
        self.input_shape = input_shape
        self.add_tensor_last = add_tensor_last
        self.add_tensor_first = add_tensor_first
        self.gray2color = gray2color
        self.visualize = visualize
        self.vis_every_n_items = vis_every_n_items
        self.visuals_dir = visuals_dir
        if decode_dict is None:
            decode_dict = dict()
        self.decode_dict = decode_dict
        if label_silence:
            if 'silence' not in decode_dict.values():
                raise ValueError('Cannot apply `silence` label if not included in '+\
                    '`decode_dict`.')
        if augment_dict is None:
            augment_dict = dict()
        self.augment_dict = augment_dict
        # if vtlp should be used as stft matrix
        if 'vtlp' in augment_dict:
            self.vtlp = augment_dict['vtlp']
        else:
            self.vtlp = None
        # ensure 'sr' is in keyword arguments
        # if not, set it to 16000
        if 'sr' in kwargs:
            self.sr = kwargs['sr']
        else:
            self.sr = 44100
            kwargs['sr'] = self.sr
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
                if len(audioinfo) != 2:
                    raise ValueError('Expected tuple containing audio file path and label. '+\
                        'Instead received tuple of length: \n{}'.format(len(audioinfo)))
                # if label is a string digit, int, or float - turn to int
                if isinstance(audioinfo[0], int) or isinstance(audioinfo[0], float) or \
                    isinstance(audioinfo[0], str) and audioinfo[0].isdigit():
                    label = int(audioinfo[0])
                    audiopath = audioinfo[1]
                elif isinstance(audioinfo[1], int) or isinstance(audioinfo[1], float) or \
                    isinstance(audioinfo[1], str) and audioinfo[1].isdigit():
                    label = int(audioinfo[1])
                    audiopath = audioinfo[1]
                else:
                    raise ValueError('Expected tuple to contain an integer label '+\
                        'and audio pathway. Received instead tuple with types '+\
                            '{} and {}.'.format(type(audioinfo[0]), type(audioinfo[1])))
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
                if self.decode_dict is not None:
                    label_pic = self.decode_dict[label].upper()
                else:
                    label_pic = label
            else:
                labeled_data = False
                label_pic = None
        
            # ensure audio is valid:
            y, sr = pyso.loadsound(audiopath, self.sr)
            
            if self.label_silence:
                if self.vad_start_end:
                    y_stft, vad = pyso.dsp.get_stft_clipped(y, sr=sr, 
                                                     win_size_ms = 50, 
                                                     percent_overlap = 0.5)
                else:
                    y_stft, __ = pyso.feats.get_vad_stft(y, sr=sr,
                                                        win_size_ms = 50,
                                                        percent_overlap = 0.5,
                                                        use_beg_ms = 120,
                                                        energy_thresh = 40, 
                                                        freq_thresh = 185, 
                                                        sfm_thresh = 5)
                if not y_stft.any():
                    label = len(self.decode_dict)-1
                    print('\nNo voice activity detected in {}'.format(audiopath))
                    print('Label {} adjusted to {}.'.format(label_pic,self.decode_dict[label]))
                    label_pic = self.decode_dict[label]
            # augment_data
            if self.augment_dict is not None:
                try:
                    augmented_data, augmentation = augment_features(y, 
                                                                self.sr, 
                                                                **self.augment_dict)
   
                except librosa.util.exceptions.ParameterError:
                    # invalid audio for augmentation
                    print('\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('File {} contains invalid sample data,'.format(audiopath)+\
                        ' incompatible with augmentation techniques. Trying again.')
                    if self.decode_dict is not None:
                        # relabel data to non-label (e.g. silence)
                        if not self.ignore_invalid:
                            label_invalid = len(self.decode_dict)-1
                            label_pic = self.decode_dict[label_invalid]
                            if label_pic == 'invalid':
                                print('Encoded label {} adjusted to {}'.format(label,
                                                                               label_invalid))
                                label = label_invalid
                                print('Label adjusted to `invalid`')
                                print('NOTE: If you would like to ignore invalid data, '+\
                                    'set `ignore_invalid` to True.\n')
                            else:
                                label_pic = self.decode_dict[label]
                                import Warnings
                                msg = '\nWARNING: Label dict does not include `invalid` label. '+\
                                    'Invalid audiofile {} will be fed '.format(audiopath)+\
                                        'to the network under {} label.\n'.format(self.decode_dict[label])
                                Warnings.warn(msg)
                        else:
                            print('Invalid data ignored (no `invalid` label applied)')
                            print('NOTE: If you do not want to ignore invalid data, '+\
                                'set `ignore_invalid` to False.\n')
                    else:
                        import Warnings
                        msg = '\nWARNING: Invalid data in audiofile {}. \n'.format(audiopath)+\
                            'No label dictionary with `invalid` label supplied. Therefore '+\
                                'model will be fed possibly invalid data with label {}\n'.format(
                                    label)
                    y = np.nan_to_num(y)
                    try:
                        augmented_data, augmentation = augment_features(
                            y, 
                            self.sr, 
                            **self.augment_dict)
                    except librosa.util.exceptions.ParameterError:
                        print('Augmentation: ', augmentation)
                        print('Augmentation failed. No augmentation applied.')
                        if not self.ignore_invalid:
                            print('Setting samples to zero.')
                            augmented_data, augmentation = np.zeros(len(y)), ''
                        else:
                            print('Caution: invalid data is ignored. Label unchanged.'+\
                                ' To label such data as '+\
                                '`invalid`, set `ignore_invalid` to False.')
                            augmented_data, augmentation = y, ''
                
            else:
                augmented_data, augmentation = y, ''
            # extract features
            # will be shape (num_frames, num_features)
            if self.vtlp:
                try:
                    win_size_ms = pyso.utils.restore_dictvalue(self.kwargs['win_size_ms'])
                except KeyError:
                    raise ValueError('win_size_ms not set for feature extraction.')
                try:
                    percent_overlap = pyso.utils.restore_dictvalue(self.kwargs['percent_overlap'])
                except KeyError:
                    percent_overlap = 0.5
                try:
                    fft_bins =  pyso.utils.restore_dictvalue(self.kwargs['fft_bins'])
                except KeyError:
                    fft_bins = None
                try:
                    window = pyso.utils.restore_dictvalue(self.kwargs['window'])
                except KeyError:
                    window = 'hann'
                try:
                    real_signal = pyso.utils.restore_dictvalue(self.kwargs['real_signal'])
                except KeyError:
                    real_signal = False
                # get vtlp
                # for 'stft' or 'powspec', can use as feats but needs to be correct size:
                if 'stft' in self.kwargs['feature_type'] or 'powspec' in \
                    self.kwargs['feature_type']:
                    expected_shape = self.input_shape[:-1]
                # for fbank, mfcc, signal features, must be able to put vtlp stft matrix
                # back into samples, therefore keep as much info as possible
                else:
                    expected_shape = None
                augmented_data, alpha = pyso.augment.vtlp(augmented_data, self.sr, 
                                          win_size_ms = win_size_ms,
                                          percent_overlap = percent_overlap, 
                                          fft_bins = fft_bins,
                                          window = window,
                                          real_signal = real_signal,
                                          expected_shape = expected_shape)
                
                ## TODO improve efficiency / issues with vtlp stft matrix and librosa
                ## Had issues converting vtlp stft into fbank or mfcc. 
                ## First turn into audio samples, then into fbank or mfcc or leave as samples.
                ## terribly slow and inefficient.
                #if 'stft' not in self.kwargs['feature_type'] and 'powspec' not in \
                    #self.kwargs['feature_type']:
                    #augmented_data = pyso.feats.feats2audio(
                        #feats = augmented_data, 
                        #feature_type = 'stft',
                        #sr = self.sr,
                        #win_size_ms = win_size_ms,
                        #percent_overlap = percent_overlap)
                #augmentation += 'alpha{}'.format(alpha)
            
            if self.vtlp and 'stft' in self.kwargs['feature_type'] or \
                'powspec' in self.kwargs['feature_type']:
                feats = augmented_data
                if 'powspec' in self.kwargs['feature_type']:
                    feats = np.abs(feats)**2
                    
            # finding it difficult to work with librosa due to slight 
            # differences in padding, fft_bins etc.
            # perhaps more reliable to use non-librosa function for 'stft' extraction
            elif 'stft'in self.kwargs['feature_type'] or \
                'powspec' in self.kwargs['feature_type']:
                feats = pyso.feats.get_stft(
                    augmented_data, 
                    win_size_ms = self.kwargs['win_size_ms'],
                    percent_overlap = self.kwargs['percent_overlap'],
                    real_signal = self.kwargs['real_signal'],
                    fft_bins = self.kwargs['fft_bins'],
                    rate_of_change = self.kwargs['rate_of_change'],
                    rate_of_acceleration = self.kwargs['rate_of_acceleration'],
                    window = self.kwargs['window'],
                    zeropad = self.kwargs['zeropad']
                    )
                
                if 'powspec' in self.kwargs['feature_type']:
                    feats = np.abs(feats)**2
            else:
                try:
                    feats = pyso.feats.get_feats(augmented_data, **self.kwargs)
                except TypeError as e:
                    print(e)
                    # invalid audio for feature_extraction
                    print('Are any non-nan? ',np.isfinite(augmented_data).any())
                    print('Are all non-nan? ',np.isfinite(augmented_data).all())
                    print('\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('File {} contains invalid sample data,'.format(audiopath)+\
                        ' incompatible with augmentation techniques. Removing NAN values.')
                    feats = pyso.feats.get_feats(np.nan_to_num(augmented_data), **self.kwargs)
                    if self.decode_dict is not None:
                        if not self.ignore_invalid:
                            label_invalid = len(self.decode_dict)-1
                            label_pic = self.decode_dict[label_invalid]
                            if label_pic == 'invalid':
                                print('Encoded label {} adjusted to {}'.format(label,
                                                                                label_invalid))
                                label = label_invalid
                                print('Label adjusted to `invalid`')
                                print('NOTE: If you would like to ignore invalid data, '+\
                                    'set `ignore_invalid` to True.\n')
                            else:
                                label_pic = self.decode_dict[label]
                                import Warnings
                                msg = '\nWARNING: Label dict does not include `invalid` label. '+\
                                    'Invalid audiofile {} will be fed '.format(audiopath)+\
                                        'to the network under {} label.\n'.format(self.decode_dict[label])
                                Warnings.warn(msg)
                        else:
                            print('Invalid data ignored (no `invalid` label applied)')
                            print('NOTE: If you do not want to ignore invalid data, '+\
                                'set `ignore_invalid` to False.\n')
                    else:
                        import Warnings
                        msg = '\nWARNING: Invalid data in audiofile {}. \n'.format(audiopath)+\
                            'No label dictionary with `invalid` label supplied. Therefore '+\
                                'model will be fed possibly invalid data with label {}\n'.format(
                                    label)
                
            if self.apply_log:
                # TODO test
                if feats[0].any() < 0:
                        feats = np.abs(feats)
                feats = np.log(feats)
            #if self.normalize:
                #feats = pyso.feats.normalize(feats)
            if not labeled_data and self.audiolist2 is not None:
                feats2 = pyso.feats.get_feats(audiopath2, **self.kwargs)
                if self.apply_log:
                    # TODO test
                    if feats2[0].any() < 0:
                            feats2 = np.abs(feats2)
                    feats2 = np.log(feats2)
                if self.normalize:
                    feats2 = pyso.feats.normalize(feats2)
            else:
                feats2 = None
                
            # Save visuals if desired
            if self.visualize:
                if self.counter % self.vis_every_n_items == 0:
                    if self.visuals_dir is not None:
                        save_visuals_path = pyso.check_dir(self.visuals_dir, make=True)
                    else:
                        save_visuals_path = pyso.check_dir('./training_images/', make=True)
                    save_visuals_path = save_visuals_path.joinpath(
                        '{}_label{}_training_{}_{}_{}.png'.format(
                            self.dataset,
                            label_pic, 
                            self.model_name, 
                            augmentation, 
                            pyso.utils.get_date()))
                    feature_type = self.kwargs['feature_type']
                    sr = self.kwargs['sr']
                    win_size_ms = self.kwargs['win_size_ms']
                    percent_overlap = self.kwargs['percent_overlap']
                    if 'stft' in feature_type or 'powspec' in feature_type or 'fbank' \
                        in feature_type:
                            energy_scale = 'power_to_db'
                    else:
                        energy_scale = None
                    pyso.feats.saveplot(
                        feature_matrix = feats, 
                        feature_type = feature_type, 
                        sr = sr, 
                        win_size_ms = win_size_ms, percent_overlap = percent_overlap,
                        energy_scale = energy_scale, save_pic = True, 
                        name4pic = save_visuals_path,
                        title = 'Label {} {} features \n'.format(label_pic, feature_type)+\
                            '(item {})'.format(self.counter))
                    if feats2 is not None:
                        # add '_2' to pathway
                        p = pyso.utils.string2pathlib(save_visuals_path)
                        p2 = p.name.stem
                        save_visuals_path2 = p.parent.joinpath(p2+'_2'+p.name.suffix)
                        pyso.feats.saveplot(
                            feature_matrix = feats2, 
                            feature_type = feature_type, 
                            sr = sr, 
                            win_size_ms = win_size_ms, percent_overlap = percent_overlap,
                            energy_scale = energy_scale, save_pic = True, 
                            name4pic = save_visuals_path2,
                            title = 'Output {} features \n'.format(
                                label_pic, feature_type)+\
                                '(item {})'.format(self.counter))
                                        
            # reshape to input shape. Will be zeropadded or limited to this shape.
            if self.input_shape is not None:
                if len(self.input_shape) != len(feats.shape):
                    change_dims = True
                else:
                    change_dims = False
                feats = pyso.feats.adjust_shape(feats, self.input_shape, 
                                                change_dims = change_dims)
                if feats2 is not None:
                    feats2 = pyso.feats.adjust_shape(feats2, self.input_shape, 
                                                     change_dims = change_dims)
                    
            # grayscale 2 color 
            # assumes already zeropadded with new channels, channels last
            if self.gray2color:
                feats = pyso.feats.grayscale2color(feats, colorscale = feats.shape[-1])
                if feats2 is not None:
                    feats2 = pyso.feats.grayscale2color(feats2, colorscale = feats2.shape[-1])
            
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
            if self.add_tensor_first is True:
                # e.g. for some lstm models
                X_batch = X_batch.reshape((1,)+X_batch.shape)
                y_batch = y_batch.reshape((1,)+y_batch.shape)
            if not self.add_tensor_first and not self.add_tensor_last:
                X_batch = X_batch
                y_batch = y_batch
            
            self.counter += 1
            yield X_batch, y_batch 
            
            #restart counter to yield data in the next epoch as well
            if self.counter >= self.number_of_batches:
                self.counter = 0


def check4na(numpyarray):
    if not np.isfinite(numpyarray).all():
        print('NAN present.')
        return True
    else:
        return False

def augment_features(sound,
                     sr,
                     add_white_noise = False, 
                     snr = [5,10,20],
                     speed_increase = False,
                     speed_decrease = False,
                     speed_perc = 0.15,
                     time_shift = False,
                     shufflesound = False,
                     num_subsections = 3,
                     harmonic_distortion = False,
                     pitch_increase = False,
                     pitch_decrease = False,
                     num_semitones = 2,
                     vtlp = False,
                     bilinear_warp = True,
                     augment_settings_dict = None,
                     ):
    if augment_settings_dict is not None:
        aug_settings = dict(augment_settings_dict)
    else:
        aug_settings = augment_settings_dict
    if speed_increase and speed_decrease:
        raise ValueError('Cannot have both speed_increase and speed_decrease'+\
            ' as augmentation options. Set just one to True.')
    if pitch_increase and pitch_decrease:
        raise ValueError('Cannot have both pitch_increase and pitch_decrease'+\
            ' as augmentation options. Set just one to True.')
    if isinstance(sound, np.ndarray):
        data = sound
    else:
        data, sr2 = pyso.loadsound(sound, sr=sr)
        assert sr2 == sr
    samples = data.copy()
    samples_augmented = samples.copy()
    augmentation = ''
    if add_white_noise:
        # allow default settings to be used/overwritten
        if aug_settings is not None:
            kwargs_aug = aug_settings['add_white_noise']
            if isinstance(kwargs_aug['snr'], str):
                kwargs_aug['snr'] = pyso.utils.restore_dictvalue(kwargs_aug['snr'])
            # if a list of snr values: choose randomly
            if isinstance(kwargs_aug['snr'], list):
                snr = np.random.choice(kwargs_aug['snr'])
        else:
            snr = np.random.choice(snr)
        samples_augmented = pyso.augment.add_white_noise(samples_augmented, 
                                                         sr = sr,
                                                         snr = snr)
        augmentation += '_whitenoise{}SNR'.format(snr)
        
    if speed_increase:
        if aug_settings is not None:
            kwargs_aug = aug_settings['speed_increase']
        else:
            kwargs_aug = dict([('perc', speed_perc)])
        samples_augmented = pyso.augment.speed_increase(samples_augmented,
                                                        sr = sr,
                                                        **kwargs_aug)
        augmentation += '_speedincrease{}'.format(kwargs_aug['perc'])


    elif speed_decrease:
        if aug_settings is not None:
            kwargs_aug = aug_settings['speed_decrease']
        else:
            kwargs_aug = dict([('perc', speed_perc)])
        samples_augmented = pyso.augment.speed_decrease(samples_augmented,
                                                        sr = sr,
                                                        **kwargs_aug)
        augmentation += '_speeddecrease{}'.format(kwargs_aug['perc'])


    if time_shift:
        samples_augmented = pyso.augment.time_shift(samples_augmented, 
                                                    sr = sr)
        augmentation += '_randtimeshift'


    if shufflesound:
        if aug_settings is not None:
            kwargs_aug = aug_settings['shufflesound']
        else:
            kwargs_aug = dict([('num_subsections', num_subsections)])
        samples_augmented = pyso.augment.shufflesound(samples_augmented, 
                                                      sr = sr,
                                                    **kwargs_aug)
        augmentation += '_randshuffle{}sections'.format(kwargs_aug['num_subsections'])


    if harmonic_distortion: 
        samples_augmented = pyso.augment.harmonic_distortion(samples_augmented,
                                                             sr = sr)
        augmentation += '_harmonicdistortion'


    if pitch_increase:
        if aug_settings is not None:
            kwargs_aug = aug_settings['pitch_increase']
        else:
            kwargs_aug = dict([('num_semitones', num_semitones)])
        samples_augmented = pyso.augment.pitch_increase(samples_augmented,
                                                        sr = sr,
                                                        **kwargs_aug)
        augmentation += '_pitchincrease{}semitones'.format(kwargs_aug['num_semitones'])


    elif pitch_decrease:
        if aug_settings is not None:
            kwargs_aug = aug_settings['pitch_decrease']
        else:
            kwargs_aug = dict([('num_semitones', num_semitones)])
        samples_augmented = pyso.augment.pitch_decrease(samples_augmented,
                                                        sr = sr,
                                                        **kwargs_aug)
        augmentation += '_pitchdecrease{}semitones'.format(kwargs_aug['num_semitones'])


    # all augmentation techniques return sample data except for vtlp
    # therefore vtlp will be handled outside of this function (returns stft matrix)
    if vtlp:
        augmentation += '_vtlp'

    samples_augmented = pyso.dsp.set_signal_length(samples_augmented, len(samples))

    return samples_augmented, augmentation

# TODO: add default values?
# does real_signal influence shape??
def get_input_shape(kwargs_get_feats, labeled_data = False,
                    frames_per_sample = None, use_librosa = True, mode = 'reflect'):
    # set defaults if not provided
    try:
        feature_type = kwargs_get_feats['feature_type']
    except KeyError:
        raise ValueError('Missing `feature_type` key and value.')
    try:
        dur_sec = kwargs_get_feats['dur_sec']
    except KeyError:
        raise ValueError('Missing `dur_sec` key and value.')
    try:
        sr = kwargs_get_feats['sr']
    except KeyError:
        kwargs_get_feats['sr'] = 441000
        sr = kwargs_get_feats['sr']
    try:
        win_size_ms = kwargs_get_feats['win_size_ms']
    except KeyError:
        kwargs_get_feats['win_size_ms'] = 25
        win_size_ms = kwargs_get_feats['win_size_ms']
    try:
        percent_overlap = kwargs_get_feats['percent_overlap']
    except KeyError:
        kwargs_get_feats['percent_overlap'] = 0.5
        percent_overlap = kwargs_get_feats['percent_overlap']
    try:
        fft_bins = kwargs_get_feats['fft_bins']
    except KeyError:
        kwargs_get_feats['fft_bins'] = None
        fft_bins = kwargs_get_feats['fft_bins']
    try:
        center = kwargs_get_feats['center']
    except KeyError:
        kwargs_get_feats['center'] = True
        center = kwargs_get_feats['center']
    try:
        num_filters = kwargs_get_feats['num_filters']
    except KeyError:
        raise ValueError('Missing `num_filters` key and value.')
        num_filters = kwargs_get_feats['num_filters']
    try:
        num_mfcc = kwargs_get_feats['num_mfcc']
    except KeyError:
        kwargs_get_feats['num_mfcc'] = None
        num_mfcc = kwargs_get_feats['num_mfcc']
    try:
        real_signal = kwargs_get_feats['real_signal']
    except KeyError:
        kwargs_get_feats['real_signal'] = True
        real_signal = kwargs_get_feats['real_signal']
    # figure out shape of data:
    total_samples = pyso.dsp.calc_frame_length(dur_sec*1000, sr=sr)
    if use_librosa:
        frame_length = pyso.dsp.calc_frame_length(win_size_ms, sr)
        win_shift_ms = win_size_ms - (win_size_ms * percent_overlap)
        hop_length = int(win_shift_ms*0.001*sr)
        if fft_bins is None:
            fft_bins = int(win_size_ms * sr // 1000)
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
        if num_filters is None:
            if 'mfcc' in feature_type:
                if num_mfcc is None:
                    num_feats = 40
                else:
                    num_feats = num_mfcc
            elif 'fbank' in feature_type:
                num_feats = 40
            elif 'powspec' in feature_type or 'stft' in feature_type:
                num_feats = int(1+fft_bins/2)
            elif 'signal' in feature_type:
                num_feats = frame_length
            else:
                raise ValueError('Feature type "{}" '.format(feature_type)+\
                    'not understood.\nMust include one of the following: \n'+\
                        ', '.join(list_available_features()))
        else:
            if 'signal' in feature_type:
                num_feats = frame_length
            elif 'stft' in feature_type or 'powspec' in feature_type:
                num_feats = int(1+fft_bins/2)
            else:
                num_feats = num_filters
            
        if frames_per_sample is not None:
            # want smaller windows, e.g. autoencoder denoiser or speech recognition
            batch_size = math.ceil(total_rows_per_wav/frames_per_sample)
            if labeled_data:
                orig_shape = (batch_size, frames_per_sample, num_feats + 1)
                input_shape = (orig_shape[0] * orig_shape[1], 
                                    orig_shape[2]-1)
            else:
                orig_shape = (batch_size, frames_per_sample, num_feats)
                input_shape = (orig_shape[0]*orig_shape[1],
                                    orig_shape[2])
        else:
            if labeled_data:
                orig_shape = (int(total_rows_per_wav), num_feats + 1)
                input_shape = (orig_shape[0], orig_shape[1]-1)
            else:
                orig_shape = (int(total_rows_per_wav), num_feats)
                input_shape = orig_shape
    return input_shape
