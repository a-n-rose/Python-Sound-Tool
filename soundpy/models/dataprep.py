'''The models.dataprep module covers functionality for feeding features to models.
'''

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)
import numpy as np
import math
import random
import soundpy as sp
import librosa


###############################################################################

#feed data to models
class Generator:
    def __init__(self, data_matrix1, data_matrix2=None, timestep = None,
                 axis_timestep = 0, normalize=True, apply_log = False, 
                 context_window = None, axis_context_window = -2, labeled_data = False,
                 gray2color = False, zeropad = True,
                 desired_input_shape = None, combine_axes_0_1=False):
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
        
        normalize : bool 
            If False, the data has already been normalized and won't be normalized
            by the generator. (default True)
        
        apply_log : bool 
            If True, log will be applied to the data.
        
        timestep : int 
            The number of frames to constitute a timestep.
            
        axis_timestep : int 
            The axis to apply the `timestep` to. (default 0)
        
        context_window : int
            The size of `context_window` or number of samples padding a central frame.
            This may be useful for models training on small changes occuring in the signal, e.g. to break up the image of sound into smaller parts. 
            
        axis_context_window : int 
            The axis to `apply_context_window`, if `context_window` is not None. 
            Ideally should be in axis preceding feature column.
            (default -2)
        
        zeropad : bool 
            If features should be zeropadded in reshaping functions.
        
        desired_input_shape : int or tuple, optional
            The desired number of features or shape of data to feed a neural network.
            If type int, only the last column of features will be adjusted (zeropadded
            or limited). If tuple, the entire data shape will be adjusted (all columns). 
            If the int or shape is larger than that of the data provided, data will 
            be zeropadded. If the int or shape is smaller, the data will be restricted.
            (default None)
        '''
        self.batch_size = 1
        self.samples_per_epoch = data_matrix1.shape[0]
        self.number_of_batches = self.samples_per_epoch/self.batch_size
        self.counter = 0
        self.datax = data_matrix1
        self.datay = data_matrix2
        self.normalize = normalize
        self.apply_log = apply_log
        self.timestep = timestep
        self.axis_timestep = axis_timestep
        self.context_window = context_window
        self.axis_context = axis_context_window
        self.zeropad = zeropad
        self.gray2color = gray2color # if need to change grayscale data to rgb
        if self.datay is None:
            # separate the label from the feature data
            self.datax, self.datay = sp.feats.separate_dependent_var(self.datax)
            if self.datay.dtype == np.complex64 or self.datay.dtype == np.complex64:
                self.datay = self.datay.astype(float)
            # assumes last column of features is the label column
            self.num_feats = self.datax.shape[-1] 
            self.labels = True
        else:
            self.labels = None
        if labeled_data:
            self.labels = True
        self.desired_shape = desired_input_shape
        self.combine_axes_0_1 = combine_axes_0_1

    def generator(self):
        '''Shapes, norms, and feeds data depending on labeled or non-labeled data.
        '''
        while 1:

            # will be size (batch_size, num_frames, num_features)
            batch_x = self.datax[self.counter] 
            batch_y = self.datay[self.counter]
            
            # ensure label is shape (1,)
            if self.labels:
                if isinstance(batch_y, np.ndarray) and len(batch_y) > 1:
                    batch_y = batch_y[:0]
                if not isinstance(batch_y, np.ndarray):
                    batch_y = np.expand_dims(batch_y, axis=0)
                    
            # TODO: is there a difference between taking log of stft before 
            # or after normalization?
            if self.normalize or self.datax.dtype == np.complex_:
                # if complex data, power spectrum will be extracted
                # power spectrum = np.abs(complex_data)**2
                batch_x = sp.feats.normalize(batch_x)
                if self.labels is None:
                    batch_y = sp.feats.normalize(batch_y)
            # apply log if specified
            if self.apply_log: 
                batch_x = np.log(np.abs(batch_x))
                # don't need to touch label data
                if self.labels is None:
                    batch_y = np.log(np.abs(batch_y))

            # reshape features to allow for timestep / subsection features
            if self.timestep is not None:
                batch_x = sp.feats.apply_new_subframe(
                    batch_x, 
                    new_frame_size = self.timestep, 
                    zeropad = self.zeropad,
                    axis = self.axis_timestep)
                if self.labels is None:
                    batch_y = sp.feats.apply_new_subframe(
                        batch_y, 
                        new_frame_size = self.timestep, 
                        zeropad = self.zeropad,
                        axis = self.axis_timestep)

            # reshape features to allow for context window / subsection features
            if self.context_window is not None:
                batch_x = sp.feats.apply_new_subframe(
                    batch_x, 
                    new_frame_size = self.context_window * 2 + 1, 
                    zeropad = self.zeropad,
                    axis = self.axis_context)
                if self.labels is None:
                    batch_y = apply_new_subframe(
                        batch_y, 
                        new_frame_size = self.context_window * 2 + 1, 
                        zeropad = self.zeropad,
                        axis = self.axis_context)
            
            if self.gray2color:
                # expects colorscale to be rgb (i.e. 3)
                # will copy first channel into the other color channels
                batch_x = sp.feats.grayscale2color(batch_x, 
                                                     colorscale = 3)
                if self.labels is None:
                    batch_y = sp.feats.grayscale2color(batch_y, 
                                                    colorscale = 3)
            
            if self.labels:
                if batch_y.dtype == np.complex64 or batch_y.dtype == np.complex128:
                    batch_y = batch_y.astype(int)
            
            # TODO test
            # if need greater number of features --> zero padding
            # could this be applied to including both narrowband and wideband data?
            # check to ensure batches match desired input shape
            if self.combine_axes_0_1 is True:
                batch_x = batch_x.reshape((batch_x.shape[0]*batch_x.shape[1],)+ batch_x.shape[2:])
                if self.labels is None:
                    batch_y = batch_y.reshape((batch_y.shape[0]*batch_y.shape[1],)+ batch_y.shape[2:])
                
            if self.desired_shape is not None:
                # can add dimensions of length 1 to first and last axis:
                try:
                    batch_x = sp.feats.adjust_shape(batch_x, self.desired_shape)
                    if self.labels is None:
                        batch_y = sp.feats.adjust_shape(batch_y, self.desired_shape)
                except ValueError:
                    raise ValueError('Data batch with shape {}'.format(batch_x.shape))+\
                        ' cannot be reshaped to match `desired_input_shape` of '+\
                            '{}. Perhaps try setting '.format(self.desired_shape) +\
                                'parameter `combine_axes_0_1` to True or False. ' +\
                                    '(default is False)'

            #send the batched and reshaped data to model
            self.counter += 1
            yield batch_x, batch_y

            #restart counter to yeild data in the next epoch as well
            if self.counter >= self.number_of_batches:
                self.counter = 0


class GeneratorFeatExtraction(Generator):
    def __init__(self, datalist, datalist2 = None, model_name = None,  
                 normalize = True, apply_log = False, randomize = True,
                 random_seed=None, desired_input_shape = None, 
                 timestep = None, axis_timestep = 0, context_window = None,
                 axis_context_window = -2, batch_size = 1,
                 gray2color = False, visualize = False,
                 vis_every_n_items = 50, visuals_dir = None,
                 decode_dict = None, dataset='train', 
                 augment_dict = None, label_silence = False,
                 vad_start_end = False, **kwargs):
        '''
    
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
            Keyword arguments for soundpy.feats.get_feats
        '''
        if desired_input_shape is None and 'dur_sec' not in kwargs.keys():
            raise ValueError('No information pertaining to amount of audio data '+\
                'to be extracted is supplied. Please specify `sample_length`, '+\
                    '`desired_input_shape`, or `dur_sec`.')
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
        self.context_window = context_window
        self.axis_context = axis_context_window
        self.timestep = timestep
        self.axis_timestep = axis_timestep
        self.normalize = normalize
        self.apply_log = apply_log
        self.desired_input_shape = desired_input_shape
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
        self.kwargs = kwargs
        
        # Ensure `feature_type` and `sr` are provided in **kwargs
        try:
            feature_type = kwargs['feature_type']
        except KeyError:
            raise KeyError('Feature type not indicated. '+\
                'Please set `feature_type` to one of the following: '+\
                '\nfbank\nstft\npowspec\nsignal\nmfcc\n')
        try: 
            sr = kwargs['sr']
        except KeyError:
            raise KeyError('Sample rate is not indicated. '+\
                'Please set `sr` (e.g. sr = 22050)')
        
    def generator(self):
        '''Extracts features and feeds them to model according to `desired_input_shape`.
        '''
        while 1:
            augmentation = ''
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
            y, sr = sp.loadsound(audiopath, self.kwargs['sr'])
            
            if self.label_silence:
                if self.vad_start_end:
                    y_stft, vad = sp.dsp.get_stft_clipped(y, sr=sr, 
                                                     win_size_ms = 50, 
                                                     percent_overlap = 0.5)
                else:
                    y_stft, __ = sp.feats.get_vad_stft(y, sr=sr,
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
                aug_dict = randomize_augs(self.augment_dict)

                augmented_data, augmentation = augment_features(y, 
                                                            self.kwargs['sr'], 
                                                            **aug_dict)
            else:
                augmented_data, augmentation = y, ''
                aug_dict = dict()
            # extract features
            # will be shape (num_frames, num_features)
            if 'vtlp' in aug_dict and aug_dict['vtlp']:
                sr = self.kwargs['sr']
                win_size_ms = sp.utils.restore_dictvalue(self.kwargs['win_size_ms'])
                percent_overlap = sp.utils.restore_dictvalue(self.kwargs['percent_overlap'])
                fft_bins =  sp.utils.restore_dictvalue(self.kwargs['fft_bins'])
                window = sp.utils.restore_dictvalue(self.kwargs['window'])
                real_signal = sp.utils.restore_dictvalue(self.kwargs['real_signal'])
                feature_type_vtlp = 'stft' 
                dur_sec = sp.utils.restore_dictvalue(self.kwargs['dur_sec'])
                zeropad = sp.utils.restore_dictvalue(self.kwargs['zeropad'])
                
                # need to tell vtlp the size of fft we need, in order to 
                # be able to extract fbank and mfcc features as well
                expected_stft_shape, __ = sp.feats.get_feature_matrix_shape(
                    sr = sr,
                    dur_sec = dur_sec, 
                    feature_type = feature_type_vtlp,
                    win_size_ms = win_size_ms,
                    percent_overlap = percent_overlap,
                    fft_bins = fft_bins,
                    zeropad = zeropad,
                    real_signal = real_signal)
                
                # TODO bug fix: oversize_factor higher than 1:
                # how to reduce dimension back to `expected_stft_shape` without
                # shaving off data?
                oversize_factor = 16
                augmented_data, alpha = sp.augment.vtlp(augmented_data, sr, 
                                          win_size_ms = win_size_ms,
                                          percent_overlap = percent_overlap, 
                                          fft_bins = fft_bins,
                                          window = window,
                                          real_signal = real_signal,
                                          expected_shape = expected_stft_shape,
                                          oversize_factor = oversize_factor,
                                          visualize=False) 
                # vtlp was last augmentation to be added to `augmentation` string
                # add the value that was applied
                augmentation += '_vtlp'+str(alpha) 
            
            if 'vtlp' in aug_dict and aug_dict['vtlp']:
                if 'stft' in self.kwargs['feature_type'] or \
                    'powspec' in self.kwargs['feature_type']:
                    if 'stft' in self.kwargs['feature_type'] and oversize_factor > 1:
                        import warnings
                        msg = '\nWARNING: due to resizing of STFT matrix due to '+\
                            ' `oversize_factor` {}, converted to '.format(oversize_factor)+\
                            'power spectrum. Phase information has been removed.'
                        warnings.warn(msg)
                    feats = augmented_data
                    if 'powspec' in self.kwargs['feature_type'] and oversize_factor == 1:
                        # otherwise already a power spectrum
                        feats = sp.dsp.calc_power(feats)
                    
            elif 'stft'in self.kwargs['feature_type'] or \
                'powspec' in self.kwargs['feature_type']:
                feats = sp.feats.get_stft(
                    augmented_data, 
                    sr = self.kwargs['sr'],
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
                    feats = sp.dsp.calc_power(feats)
                    
            if 'fbank' in self.kwargs['feature_type']:
                feats = sp.feats.get_fbank(
                    augmented_data,
                    sr = self.kwargs['sr'],
                    num_filters = self.kwargs['num_filters'],
                    win_size_ms = self.kwargs['win_size_ms'],
                    percent_overlap = self.kwargs['percent_overlap'],
                    real_signal = self.kwargs['real_signal'],
                    fft_bins = self.kwargs['fft_bins'],
                    rate_of_change = self.kwargs['rate_of_change'],
                    rate_of_acceleration = self.kwargs['rate_of_acceleration'],
                    window = self.kwargs['window'],
                    zeropad = self.kwargs['zeropad']
                    )
            
            elif 'mfcc' in self.kwargs['feature_type']:
                feats = sp.feats.get_mfcc(
                    augmented_data,
                    sr = self.kwargs['sr'],
                    num_mfcc = self.kwargs['num_mfcc'],
                    num_filters = self.kwargs['num_filters'],
                    win_size_ms = self.kwargs['win_size_ms'],
                    percent_overlap = self.kwargs['percent_overlap'],
                    real_signal = self.kwargs['real_signal'],
                    fft_bins = self.kwargs['fft_bins'],
                    rate_of_change = self.kwargs['rate_of_change'],
                    rate_of_acceleration = self.kwargs['rate_of_acceleration'],
                    window = self.kwargs['window'],
                    zeropad = self.kwargs['zeropad']
                    )
                
            if self.apply_log:
                # TODO test
                if feats[0].any() < 0:
                        feats = np.abs(feats)
                feats = np.log(feats)
            if self.normalize:
                feats = sp.feats.normalize(feats)
            if not labeled_data and self.audiolist2 is not None:
                feats2 = sp.feats.get_feats(audiopath2, **self.kwargs)
                if self.apply_log:
                    # TODO test
                    if feats2[0].any() < 0:
                        feats2 = np.abs(feats2)
                    feats2 = np.log(feats2)
                if self.normalize:
                    feats2 = sp.feats.normalize(feats2)
            else:
                feats2 = None
                
            # Save visuals if desired
            if self.visualize:
                if self.counter % self.vis_every_n_items == 0:
                    # make augmentation string more legible.
                    augments_vis = augmentation[1:].split('_')
                    if len(augments_vis) > 1:
                        augs1 = augments_vis[:len(augments_vis)//2]
                        augs2 = augments_vis[len(augments_vis)//2:]
                        augs1 = ', '.join(augs1)
                        augs2 = ', '.join(augs2)
                    else:
                        augs1 = augments_vis[0]
                        augs2 = ''
                    if self.visuals_dir is not None:
                        save_visuals_path = sp.check_dir(self.visuals_dir, make=True)
                    else:
                        save_visuals_path = sp.check_dir('./training_images/', make=True)
                    save_visuals_path = save_visuals_path.joinpath(
                        '{}_label{}_training_{}_{}_{}.png'.format(
                            self.dataset,
                            label_pic, 
                            self.model_name, 
                            augmentation, 
                            sp.utils.get_date()))
                    feature_type = self.kwargs['feature_type']
                    sr = self.kwargs['sr']
                    win_size_ms = self.kwargs['win_size_ms']
                    percent_overlap = self.kwargs['percent_overlap']
                    if 'stft' in feature_type or 'powspec' in feature_type or 'fbank' \
                        in feature_type:
                            energy_scale = 'power_to_db'
                    else:
                        energy_scale = None
                    sp.feats.plot(
                        feature_matrix = feats, 
                        feature_type = feature_type, 
                        sr = sr, 
                        win_size_ms = win_size_ms, percent_overlap = percent_overlap,
                        energy_scale = energy_scale, save_pic = True, 
                        name4pic = save_visuals_path,
                        title = '"{}" {} Aug: {}-\n{}'.format(
                            label_pic, 
                            feature_type.upper(),
                            augs1,
                            augs2),
                            use_tkinter=False) #use Agg backend for plotting
                    if feats2 is not None:
                        # add '_2' to pathway
                        p = sp.utils.string2pathlib(save_visuals_path)
                        p2 = p.name.stem
                        save_visuals_path2 = p.parent.joinpath(p2+'_2'+p.name.suffix)
                        sp.feats.plot(
                            feature_matrix = feats2, 
                            feature_type = feature_type, 
                            sr = sr, 
                            win_size_ms = win_size_ms, percent_overlap = percent_overlap,
                            energy_scale = energy_scale, save_pic = True, 
                            name4pic = save_visuals_path2,
                            title = 'Output {} features {}'.format(
                                label_pic, feature_type),
                            use_tkinter=False)
            
            batch_x = feats
            batch_y = feats2

            # reshape features to allow for timestep / subsection features
            if self.timestep is not None:
                batch_x = sp.feats.apply_new_subframe(
                    batch_x, 
                    new_frame_size = self.timestep, 
                    zeropad = self.zeropad,
                    axis = self.axis_timestep)
                if batch_y is not None:
                    batch_y = sp.feats.apply_new_subframe(
                        batch_y, 
                        new_frame_size = self.timestep, 
                        zeropad = self.zeropad,
                        axis = self.axis_timestep)

            # reshape features to allow for context window / subsection features
            if self.context_window is not None:
                batch_x = sp.feats.apply_new_subframe(
                    batch_x, 
                    new_frame_size = self.context_window * 2 + 1, 
                    zeropad = self.zeropad,
                    axis = self.axis_context)
                if batch_y is not None:
                    batch_y = apply_new_subframe(
                        batch_y, 
                        new_frame_size = self.context_window * 2 + 1, 
                        zeropad = self.zeropad,
                        axis = self.axis_context)
                    
            # grayscale 2 color 
            if self.gray2color:
                batch_x = sp.feats.grayscale2color(batch_x,
                                                   colorscale = 3) # default colorscale is 3
                if batch_y is not None:
                    batch_y = sp.feats.grayscale2color(batch_y, 
                                                       colorscale = 3)

            # reshape to input shape. Will be zeropadded or limited to this shape.
            # tensor dimensions on either side can be added here as well.
            if self.desired_input_shape is not None:
                batch_x = sp.feats.adjust_shape(batch_x, self.desired_input_shape)
                if batch_y is not None:
                    batch_y = sp.feats.adjust_shape(batch_y, self.desired_input_shape)
            
            # prepare data to be fed to network:
            if labeled_data:
                # has to be at least (1,)
                batch_y = np.expand_dims(np.array(label), axis=0)
                
            elif batch_y is not None:
                pass
            else:
                raise ValueError('No independent variable provided.')


            self.counter += 1
            yield batch_x, batch_y 
            
            #restart counter to yield data in the next epoch as well
            if self.counter >= self.number_of_batches:
                self.counter = 0


def check4na(numpyarray):
    if not np.isfinite(numpyarray).all():
        print('NAN present.')
        return True
    else:
        return False
    
    
def randomize_augs(aug_dict, random_seed=None):
    '''Creates copy of dict and chooses which augs applied randomly.
    
    Can apply random seed for number of augmentations applied and shuffling
    order of possible augmentations.
    '''
    possible_augs = []
    num_possible_aug = 0
    if aug_dict is not None:
        for key, value in aug_dict.items():
            if value == True:
                num_possible_aug += 1
                possible_augs.append(key)
               
    if random_seed is not None:
        np.random.seed(random_seed)
    num_augs = np.random.choice(range(num_possible_aug+1))
    
    if num_augs == 0:
        # no augmentations applied:
        new_dict = dict(aug_dict)
        for key, value in new_dict.items():
            if value == True:
                new_dict[key] = False
        return new_dict
    
    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(possible_augs)
    augs = possible_augs[:num_augs]
    augs_leftover = augs[num_augs:]
    if 'speed_increase' in augs and 'speed_decrease' in augs:
        i1 = augs.index('speed_increase')
        i2 = augs.index('speed_decrease')
        x = [i1, i2]
        random.shuffle(x)
        speed2remove = augs.pop(x[0])
        if augs_leftover:
            aug_added = augs_leftover.pop(0)
            augs.append(aug_added)
    if 'pitch_increase' in augs and 'pitch_decrease' in augs:
        i1 = augs.index('pitch_increase')
        i2 = augs.index('pitch_decrease')
        x = [i1, i2]
        random.shuffle(x)
        pitch2remove = augs.pop(x[0])
        if augs_leftover:
            aug_added = augs_leftover.pop(0)
            augs.append(aug_added)
    new_dict = dict(aug_dict)
    for key, value in new_dict.items():
        if value == True:
            if key not in augs:
                new_dict[key] = False
    return new_dict

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
                     random_seed = None,
                     ):
    '''Randomly applies augmentations to audio
    '''
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
        data, sr2 = sp.loadsound(sound, sr=sr)
        assert sr2 == sr
    samples = data.copy()
    samples_augmented = samples.copy()
    augmentation = ''
    if add_white_noise:
        # allow default settings to be used/overwritten
        if aug_settings is not None:
            kwargs_aug = aug_settings['add_white_noise']
            if isinstance(kwargs_aug['snr'], str):
                kwargs_aug['snr'] = sp.utils.restore_dictvalue(kwargs_aug['snr'])
            # if a list of snr values: choose randomly
            if isinstance(kwargs_aug['snr'], list):
                snr = np.random.choice(kwargs_aug['snr'])
        else:
            snr = np.random.choice(snr)
        samples_augmented = sp.augment.add_white_noise(samples_augmented, 
                                                         sr = sr,
                                                         snr = snr)
        augmentation += '_whitenoise{}SNR'.format(snr)
        
    if speed_increase:
        if aug_settings is not None:
            kwargs_aug = aug_settings['speed_increase']
        else:
            kwargs_aug = dict([('perc', speed_perc)])
        samples_augmented = sp.augment.speed_increase(samples_augmented,
                                                        sr = sr,
                                                        **kwargs_aug)
        augmentation += '_speedincrease{}'.format(kwargs_aug['perc'])


    elif speed_decrease:
        if aug_settings is not None:
            kwargs_aug = aug_settings['speed_decrease']
        else:
            kwargs_aug = dict([('perc', speed_perc)])
        samples_augmented = sp.augment.speed_decrease(samples_augmented,
                                                        sr = sr,
                                                        **kwargs_aug)
        augmentation += '_speeddecrease{}'.format(kwargs_aug['perc'])


    if time_shift:
        samples_augmented = sp.augment.time_shift(samples_augmented, 
                                                    sr = sr)
        augmentation += '_randtimeshift'


    if shufflesound:
        if aug_settings is not None:
            kwargs_aug = aug_settings['shufflesound']
        else:
            kwargs_aug = dict([('num_subsections', num_subsections)])
        samples_augmented = sp.augment.shufflesound(samples_augmented, 
                                                      sr = sr,
                                                    **kwargs_aug)
        augmentation += '_randshuffle{}sections'.format(kwargs_aug['num_subsections'])


    if harmonic_distortion: 
        samples_augmented = sp.augment.harmonic_distortion(samples_augmented,
                                                             sr = sr)
        augmentation += '_harmonicdistortion'


    if pitch_increase:
        if aug_settings is not None:
            kwargs_aug = aug_settings['pitch_increase']
        else:
            kwargs_aug = dict([('num_semitones', num_semitones)])
        samples_augmented = sp.augment.pitch_increase(samples_augmented,
                                                        sr = sr,
                                                        **kwargs_aug)
        augmentation += '_pitchincrease{}semitones'.format(kwargs_aug['num_semitones'])


    elif pitch_decrease:
        if aug_settings is not None:
            kwargs_aug = aug_settings['pitch_decrease']
        else:
            kwargs_aug = dict([('num_semitones', num_semitones)])
        samples_augmented = sp.augment.pitch_decrease(samples_augmented,
                                                        sr = sr,
                                                        **kwargs_aug)
        augmentation += '_pitchdecrease{}semitones'.format(kwargs_aug['num_semitones'])


    # all augmentation techniques return sample data except for vtlp
    # therefore vtlp will be handled outside of this function (returns stft or powspec)
    if vtlp:
        pass

    samples_augmented = sp.dsp.set_signal_length(samples_augmented, len(samples))

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
    total_samples = sp.dsp.calc_frame_length(dur_sec*1000, sr=sr)
    if use_librosa:
        frame_length = sp.dsp.calc_frame_length(win_size_ms, sr)
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

def make_gen_callable(_gen):
    '''Prepares Python generator for `tf.data.Dataset.from_generator`
    
    Bug fix: Python generator fails to work in Tensorflow 2.2.0 + 
    
    Parameters
    ----------
    _gen : generator
        The generator function to feed to a deep neural network.
        
    Returns
    -------
    x : np.ndarray [shape=(batch_size, num_frames, num_features, 1)]
        The feature data
        
    y : np.ndarray [shape=(1,1)]
        The label for the feature data.
    References
    ----------
    Shu, Nicolas (2020) https://stackoverflow.com/a/62186572
    CC BY-SA 4.0
    '''
    def gen():
        for x,y in _gen:
                yield x,y
    return gen
