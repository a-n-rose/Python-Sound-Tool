# Updates of v0.1.0a3 release:

## Updates
- don't use librosa for feature extraction anymore. But compatible with previous versions.
- parameter: frames_per_sample and context_window, with depreciation warning
Just remove these parameters from feature extraction and limit to generators. Otherwise too messy and complex
- soundpy.models.builtin.implement_denoiser() raise warning if cleaned features cannot be 
converted to raw audio samples.
- BUG FIX: soundpy.feats.plot can now be used from within generator using backend Agg and 
then switch to Tkinker backend using use_tkinker parameter for normal use outside of training.
- require additional tensors to be added to the desired shape and then supplied to generator to make shape process more explicit in generator.

changed parameter (Generator) normalized to normalize (opposite bool); removed add_tensor_last parameter, adjusted grayscale2color sections: can be applied to 2D data; set sr default to 22050

- Got the augment cnn builtin functionality to run with pre-trained features.. needs cleaning
- got plotsound, plot vad, and plot dom freq, to work with stereo sound

Removing from envclassifier_extract_train:
    dataset_dict = None,
    num_labels = None,


## Updates of v0.1.0a2 release:

### Updated Dependencies
- Updated dependencies to newest versions still compatible with Tensorflow 2.1.0
- Note: bug in training with generators occurs with Tensorflow 2.2.0+. Models trained via generators fail to learn. Therefore, Tensorflow is limited to version 2.1.0 until that bug is fixed. 

### GPU option added
- provide instructions for running Docker image for GPU

### soundpy.dsp.vad
- add `use_beg_ms` parameter: improved VAD recognition of silences post speech.
- raise warning for sample rates lower than 44100 Hz. VAD seems to fail at lower sample rates.

### soundpy.feats.get_vad_samples and soundpy.feats.get_vad_stft
- moved from dsp module to the feats module
- add `extend_window_ms` paremeter: can extend VAD window if desired. Useful in higher SNR environments.
- raise warning for sample rates lower than 44100 Hz. VAD seems to fail at lower sample rates.

### added soundpy.feats.get_samples_clipped and soundpy.feats.get_stft_clipped
- another option for VAD 
- clips beginning and ending of audio data where high energy sound starts and ends.

### soundpy.models.dataprep.GeneratorFeatExtraction 
- can extract and augment features from audio files as each audio file fed to model. 
- example can be viewed: soundpy.models.builtin.envclassifier_extract_train
- note: still very experimental

### soundpy.dsp.add_backgroundsound
- improvements in the smoothness of the added signal.
- soundpy.dsp.clip_at_zero
- improved soundpy.dsp.vad and soundpy.feats.get_vad_stft

### soundpy.feats.normalize 
- can use it: soundpy.normalize (don't need to remember dsp or feats)

### soundpy.dsp.remove_dc_bias
- implemented in soundpy.files.loadsound() and soundpy.files.savesound()
- vastly improves the ability to work with and combine signals.

### soundpy.dsp.clip_at_zero
- clips beginning and ending audio at zero crossings (at negative to positive zero crossings)
- useful when concatenating signals
- useful for removing clicks at beginning or ending of audio signals

### soundpy.dsp.apply_sample_length
- can now mirror the sound as a form of sound extention with parameter `mirror_sound`.

### Removed soundpy_online (and therefore mybinder as well)
- for the time being, this is too much work to keep up. Eventually plan on bringing this back in a more maintainable manner.

### Added stereo sound functionality to the following functions:
- soundpy.dsp.add_backgroundsound
- soundpy.dsp.clip_at_zero
- soundpy.dsp.calc_fft
- soundpy.feats.get_stft
- soundpy.feats.get_vad_stft

### New functions related to stereo sound
- soundpy.dsp.ismono for checking if a signal is mono or stereo
- soundpy.dsp.average_channels for averaging amplitude in all channels (e.g. identifying when energetic sounds start / end: want to consider all channels)
- soundpy.dsp.add_channels for adding additional channels if needed (e.g. for applying a 'hann' or 'hamming' window to stereo sound)
