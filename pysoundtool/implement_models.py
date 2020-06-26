import numpy as np
import keras
from keras.models import load_model
import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
packagedir = os.path.dirname(currentdir)
sys.path.insert(0, packagedir)
import pysoundtool as pyst
import pysoundtool.models as pystmodels

    
###############################################################################


def denoiser_run(model, new_audiofile, feat_settings_dict):
    '''Implements a pre-trained denoiser
    
    Parameters
    ----------
    model : str or pathlib.PosixPath
        The path to the denoising model.
    
    new_audiofile : str or pathlib.PosixPath
        The path to the noisy audiofile.
        
    feat_settings_dict : dict 
        Dictionary containing necessary settings for how the features were
        extracted for training the model. Expected keys: 'feature_type', 
        'win_size_ms', 'percent_overlap', 'sr', 'window', 'frames_per_sample',
        'input_shape', 'desired_shape', 'dur_sec', 'num_feats'.
        
    Returns
    -------
    cleaned_audio : np.ndarray [shape = (num_samples, )]
        The cleaned audio samples ready for playing or saving as audio file.
    sr : int 
        The sample rate of `cleaned_audio`.
        
    See Also
    --------
    pysoundtool.feats.get_feats
        How features are extracted.
        
    pysoundtool.feats.feats2audio
        How features are transformed back tino audio samples.
    '''
    feature_type = feat_settings_dict['feature_type']
    win_size_ms = feat_settings_dict['win_size_ms']
    sr = feat_settings_dict['sr']
    percent_overlap = feat_settings_dict['percent_overlap']
    window = feat_settings_dict['window']
    frames_per_sample = feat_settings_dict['frames_per_sample']
    input_shape = feat_settings_dict['input_shape']
    dur_sec = feat_settings_dict['dur_sec']
    num_feats = feat_settings_dict['num_feats']
    desired_shape = feat_settings_dict['desired_shape']
    
    feats = pyst.feats.get_feats(new_audiofile, sr=sr, 
                                features = feature_type,
                                win_size_ms = win_size_ms,
                                percent_overlap = percent_overlap,
                                window = window, 
                                dur_sec = dur_sec,
                                num_filters = num_feats)
    # are phase data still present? (only in stft features)
    if feats.dtype == np.complex and np.min(feats) < 0:
        original_phase = pyst.dsp.calc_phase(feats,
                                               radians=False)
    elif 'stft' in feature_type or 'powspec' in feature_type:
        feats_stft = pyst.feats.get_feats(new_audiofile, 
                                          sr=sr, 
                                          features = 'stft',
                                          win_size_ms = win_size_ms,
                                          percent_overlap = percent_overlap,
                                          window = window, 
                                          dur_sec = dur_sec,
                                          num_filters = num_feats)
        original_phase = pyst.dsp.calc_phase(feats_stft,
                                               radians = False)
    else:
        original_phase = None
    
    feats = pystmodels.dataprep.prep_new_audiofeats(feats,
                                                    desired_shape,
                                                    input_shape)
    # ensure same shape as feats
    if original_phase is not None:
        original_phase = pystmodels.dataprep.prep_new_audiofeats(original_phase, 
                                                   desired_shape,
                                                   input_shape)
    
    
    feats_normed = pyst.feats.normalize(feats)
    denoiser = load_model(model)
    cleaned_feats = denoiser.predict(feats_normed, batch_size = frames_per_sample)
    
    # need to change shape back to 2D
    # current shape is (batch_size, num_frames, num_features, 1)
    # need (num_frames, num_features)
    
    if 'signal' in feature_type:
        feats_flattened = np.flatten(feats_normed)
        audio_shape = (feats_flattened.shape + (1,))
    
    else: 
        # remove last tensor dimension
        if feats_normed.shape[-1] == 1:
            feats_normed = feats_normed.reshape(feats_normed.shape[:-1])
        feats_flattened = feats_normed.reshape(-1, 
                                               feats_normed.shape[-1])
        audio_shape = (feats_flattened.shape)
    
    cleaned_feats = cleaned_feats.reshape(audio_shape)
    if original_phase is not None:
        original_phase = original_phase.reshape(audio_shape)
    
    # now combine them to create audio samples:
    cleaned_audio = pyst.feats.feats2audio(cleaned_feats, 
                                           feature_type = feature_type,
                                           sr = sr, 
                                           win_size_ms = win_size_ms,
                                           percent_overlap = percent_overlap,
                                           phase = original_phase)
    noisy_audio, noisy_sr = pyst.loadsound(new_audiofile, sr=sr)
    if len(cleaned_audio) > len(noisy_audio):
        cleaned_audio = cleaned_audio[:len(noisy_audio)]
    
    max_energy_original = np.max(noisy_audio)
    cleaned_audio = pyst.dsp.control_volume(cleaned_audio, max_limit = max_energy_original)
    return cleaned_audio, sr
