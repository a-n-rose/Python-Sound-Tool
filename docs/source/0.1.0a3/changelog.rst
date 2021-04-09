*********
Changelog
*********

v0.1.0a
=======


v0.1.0a3
--------
2021-04-09

Bug fixes
   -  no longer use Librosa for feature extraction: allow easier implementation of augmentations, especially during training. 
   -  `soundpy.feats.plot` now uses parameter `subprocess` to allow for different backends to be applied, depending on when funciton is called. For example, if plotting from within a Generator while training, `subprocess` should be set to True, and the 'Agg' backend will be applied. Otherwise, 'TkAgg' backend is used. Fixes issues with multi-threading.
   -  Fixed generator and Tensorflow issue: with Tensorflow 2.2.0+ the models in `soundpy.models.builtin` that were trained via generator failed. Use `tensorflow.data.Dataset.from_generator` to feed generator data to models.
   -  Improved `clip_at_zero`.

Features
   -  Python 3.8 can now be used.
   -  throw depreciation warning for parameters `context_window` or `frames_per_sample` as these "features" will be removed from feature extraction. Rather the features can be reshaped post feature extraction.
   -  added `timestep`, `axis_timestep`, `context_window`, `axis_context_window`  and `combine_axes_0_1` paremeters to  `soundpy.models.Generator`:  allow more control over shape of the features.
   -  can run `soundpy.models.builtin.envclassifier_extract_train` to run with pre-extracted val and test features. 
   -  `soundpy.feats.plotsound`, `soundpy.feats.plot_vad` and `soundpy.feats.plot_dom_freq` all can plot stereo sound: for each channel in a stereo signal, a plot is either generated or saved. If a filename already exists, a date stamp is added to filename to avoid overwriting images.
   - allow `grayscale2color` to be applied to 2D data.

Breaking changes
   -  `soundpy.models.Generator` uses parameter `normalize` instaed of `normalized`. Found this to be more intuitive. If `normalize` is set to True, data will be normalized. Before, if `normalized` was set to True, data would not be normalized.
   -  removed `add_tensor_last` and `add_tensor_first`: require adding of tensors (for keras) to be included in parameter `desired_input_shape`.
   
Other changes 
   -  CPU soundpy can use Tensorflow 2.1.0, 2.2.0 and 2.3.0. Dockerfile still uses Tensorflow 2.1.0 as it is still compatible with updated code.
   -  `soundpy.models.builtin.implement_denoiser` raises warning if cleaned features cannot be converted to raw audio samples.

   
v0.1.0a2
--------
2020-08-13


Bug fixes
   -  added `use_beg_ms` parameter in `soundpy.dsp.vad`: improved VAD recognition of silences post speech.

Features
   -  added GPU option: provide instructions and Docker image for running SoundPy with GPU
   -  added `extend_window_ms` paremeter to `soundpy.feats.get_vad_samples` and `soundpy.feats.get_vad_stft`:  can extend VAD window if desired. Useful in higher SNR environments.
   -  added `soundpy.feats.get_samples_clipped` and `soundpy.feats.get_stft_clipped` to clip off beginning and ending silences.
   -  added `beg_end_clipped` parameter to `soundpy.feats.plot_vad` to visualize VAD by clipping the beginning and ending silences (if True) or VAD instances throughout the signal (if False).
   -  added `soundpy.models.dataprep.GeneratorFeatExtraction` class for extracting and augmenting features during training (still experimental).
   -  added `soundpy.models.builtin.envclassifier_extract_train` as an example of extracting and augmenting features during training (still experimental).
   -  added `soundpy.dsp.clip_at_zero` to enable smoother concatenations of signals and enables removal of clicks at beginning and ending of signals.
   -  added `soundpy.dsp.remove_dc_bias` to enable smoother concatenations of signals
   -  added and set `remove_dc` parameter to True in `soundpy.files.loadsound` and `soundpy.files.savesound` to ensure signals all have mean zero.
   -  added `mirror_sound` option to `soundpy.dsp.apply_sample_length` as a way to extend sound.
   -  added `soundpy.dsp.ismono` to check if samples were mono or stereo.
   -  added `soundpy.dsp.average_channels` to average sample amplitudes across channels, e.g. to identify where high energy begins / ends in the signal without disregarding additional channels (if stereo sound).
   -  added `soundpy.dsp.add_channels` for adding additional channels if needed (e.g. for applying a 'hann' or 'hamming' window to stereo sound)
   -  added stereo sound functionality to `soundpy.dsp.add_backgroundsound`, `soundpy.dsp.clip_at_zero`, `soundpy.dsp.calc_fft`, `soundpy.feats.get_stft`, `soundpy.feats.get_vad_stft` 
   

Other changes
   -  name change: from pysoundtool to soundpy: simpler
   -  updated dependencies to newest versions still compatible with Tensorflow 2.1.0
   -  moved `soundpy.dsp.get_vad_samples` to `soundpy.feats.get_vad_samples`
   -  moved `soundpy.dsp.get_vad_stft` to `soundpy.feats.get_vad_stft`
   -  name change: allow `soundpy.feats.normalize` to be used as `soundpy.normalize`
   -  removed `pysoundtool_online` and mybinder button as maintaining the online version was not easily done. Aim to reimplement at some point.
   


v0.1.0a1
========

Initial public alpha release.
