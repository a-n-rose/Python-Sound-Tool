*********
Changelog
*********

v0.1.0a
=======

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
