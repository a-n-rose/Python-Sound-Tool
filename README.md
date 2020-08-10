# SoundPy

SoundPy is an experimental framework for exploring sound as well as machine learning in the context of sound. 

[![License](https://img.shields.io/badge/license-GNU%20AGPL-brightgreen)](https://github.com/a-n-rose/Python-Sound-Tool/blob/master/LICENSE.md)
[![PyPI pyversions](https://img.shields.io/badge/python-3.6-yellow)](https://www.python.org/downloads/release/python-360/)


## Documentation

For examples and to navigate the code, see the <a href="https://aislynrose.bitbucket.io/">documentation</a>. 

## Main Uses:

### Visualization
- pre and post filtering
- during feature extraction process 
- various feature types: raw signal vs stft vs fbank vs mfcc
- voice activity in signal
- dominant frequency in signal

### Audio Prep / Manipulation
- convert audiofiles 
- extract features: raw signal, stft, powspec, fbank, mfcc
- augment audio: speed, pitch, add noise, time shift, shuffle, vtlp, harmonic distortion
- filter noise (e.g. wiener filter)
- denoise signal (e.g. with pretrained denoiser model)
- remove non-speech from signal
- identify voice activity in signal
- measure dominant and basic frequencies in signal

### Train and Implement Deep Neural Networks
- cnn model (e.g. sound classifier)
- cnn+lstm model (e.g. speech recognition)
- autoencoder model (e.g. denoiser model)
- pretrained ResNet50 model (e.g. language classifier)

## Requirements

Python 3.6

### For Linux users:

In order to use all functionality (i.e. `soundpy` and not `soundpy_online` you must 
have `libsndfile1` installed in your system. <a href="https://pypi.org/project/SoundFile/">Soundfile</a> uses this library.

```
$ sudo apt-get install libsndfile1
```

(This should automatically get installed for other operating systems during `pip install soundfile`).

## Updates of upcoming release:

### soundpy.dsp.vad()
- add `use_beg_ms` parameter: improved VAD recognition of silences post speech.
- raise warning for sample rates lower than 44100 Hz. VAD seems to fail at lower sample rates.

### soundpy.feats.get_vad_samples() and soundpy.feats.get_vad_stft()
- moved from dsp module to the feats module
- add `extend_window_ms` paremeter: can extend VAD window if desired. Useful in higher SNR environments.
- raise warning for sample rates lower than 44100 Hz. VAD seems to fail at lower sample rates.

### added soundpy.feats.get_samples_clipped() and soundpy.feats.get_stft_clipped()
- another option for VAD 
- clips beginning and ending of audio data where high energy sound starts and ends.

### soundpy.models.dataprep.GeneratorFeatExtraction 
- can extract and augment features from audio files as each audio file fed to model. 
- example can be viewed: soundpy.builtin.envclassifier_extract_train

### soundpy.dsp.add_backgroundsound()
- improvements in the smoothness of the added signal.
- soundpy.dsp.clip_at_zero
- improved soundpy.dsp.vad and soundpy.feats.get_vad_stft

### soundpy.feats.normalize 
- can use it: soundpy.normalize (don't need to remember dsp or feats)

### soundpy.dsp.remove_dc_bias() 
- implemented in soundpy.files.loadsound() and soundpy.files.savesound()
- vastly improves the ability to work with and combine signals.

### soundpy.dsp.clip_at_zero()
- clips beginning and ending audio at zero crossings (at negative to positive zero crossings)
- useful when concatenating signals
- useful for removing clicks at beginning or ending of audio signals

### soundpy.dsp.apply_sample_length()
- can now mirror the sound as a form of sound extention with parameter `mirror_sound`.

### Removed soundpy_online (and therefore mybinder as well)
- for the time being, this is too much work to keep up. Eventually plan on bringing this back in a more maintainable manner.

## About

Note: as is, soundpy is not yet a stable framework, meaning changes might periodically be made without extreme focus on backwards compatibility. 

Those who might find this useful: 

* speech and sound enthusiasts
* digital signal processing / mathematics / physics / acoustics enthusiasts
* deep learning enthusiasts
* researchers
* linguists
* psycholinguists

The main goal of soundpy is to provide the code and functionality with more context via visualization, research, and mathematics. Most of the resources used to build the functionality stems from publicly available research and datasets. (For a list of open datasets, see my ever growing <a href='https://a-n-rose.github.io/2019/01/06/resources-publicly-available-speech-databases.html'>collection</a>.)

As it covers quite a large range, from audio file conversion to implementation of trained neural networks, the purpose of soundpy is not to be the perfect implementation of all functions (although that is also a goal :P ), but rather a peak into how they *can* be implemented, hopefully offering others a foundation for trying out different ways of implementation (feature extraction, building neural networks, etc.).

This project is still in the beginning stages and has a lot of room for growth, especially with contributers having a background / knowlege in data science, computer science, machine and deep learning, physics, acoustics, or dsp. Contributers from other backgrounds are also welcome! If you'd like soundpy to do something it doesn't, try making it or create an issue.

# Installation

You can install SoundPy by cloning this repository. 

You will be able to use the provided example audio and models in the folder `audiodata` and you can also use soundpy_online instead of soundpy, however, this has decreased functionality as it does not import librosa or soundfile, two awesome packages for working with audio.

Before either, I suggest a virtual environment:

```
$ python3 -m venv env
```
or, to better control your python version:
```
$ virtualenv -p python3.6 env
```
Then activate the environment
```
$ source env/bin/activate
(env)..$
```

## Install via cloning this repo

Clone this repository. Set the working directory where you clone this repository.

Then install the necessary dependencies via pip:
```
(env)..$ pip install -r requirements.txt --use-feature=2020-resolver
```
# Examples 

You can explore example code:

## Locally via Jupyter Notebook:

Install and run jupyter notebook:

```
(env)..$ pip install notebook
(env)..$ jupyter notebook
```
Venture into the folder `jupyter_notebooks` and have a go!

## Visually and Aurally in the Documentation:

<a href="https://aislynrose.bitbucket.io/example_cases.html">SoundPy Examples</a> 

# Example datasets

If you would like to play around with various types of sound, check out my <a href='https://github.com/a-n-rose/mini-audio-datasets'>repo</a> containing mini datasets of sound, ranging from speech to noise. They are very small so don't expect much as it comes to training neural networks. 

# Testing

If you want to run the tests for SoundPy, they currently use some <a href="https://github.com/a-n-rose/mini-audio-datasets/tree/master/test_audio">audiofiles available</a> in the example datasets repo, mentioned above. Also, see `tests_requirements.txt`. The packages located there will need to be installed via:

```
(env)..$ pip install -r tests_requirements.txt
```

# About the Author

I studied clinical linguistics for my master's which introduced me to the inspiring world of open source software, python programming, and the beauty of acoustics. My interests broadened from academic experimental design to the development of tools for identifiying speech irregularities and beyond. Through this I discovered a bit of a missing niche in the Python community: a high-level sound tool for filtering, analysis, **and** deep learning that also offers context to its functionality. You can read more about me and my projects on my <a href="https://a-n-rose.github.io/">blog</a>.

# The Beginnings of SoundPy

This project stemmed from the Prototype Fund project <a href="https://github.com/pgys/NoIze">NoIze</a> which was <a href="https://www.youtube.com/watch?v=BJ0f2x49Imc&feature=youtu.be">presented</a> at PyConDE / PyData Berlin in 2019. This fork broadens the application of the software from smart noise filtering to general sound analysis, filtering, visualization, preparation, etc. Therefore the name has been adapted to more general sound functionality.
