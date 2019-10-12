# PySoundTool

This project stemmed from the Prototype Fund project <a href="https://github.com/pgys/NoIze">NoIze</a>. This fork broadens the application of the software from smart noise filtering to general sound analysis, filtering, visualization, preparation, etc. Therefore the name has been adapted to more general sound functionality.

Note: for adjusting sound files, **apply only to copies of the originals**. Improvements need to be made to ensure files don't get overwritten except explicitly indicated. 

## Jupyter notebooks

You can use some of the tools availabe here in my Jupyter notebooks. You can check them out here on <a href="https://mybinder.org/v2/gh/a-n-rose/Python-Sound-Tool/master">Binder</a>. As I do include some audio data, it may take a couple of minutes to load..

# Installation

This repository serves as a place to explore sound. Therefore, small sound datasets are included in this repo. The size is appx. 30-40MB. If you clone this repo, this sound data will be cloned as well.

If you are fine with this, clone this repository. Set the working directory where you clone this repository.

Start a virtual environment:

```
$ python3 -m venv env
$ source env/bin/activate
(env)..$
```
Then install necessary installations via pip:
```
(env)..$ pip install -r requirements.txt
```

# ToDo

* **Ensure** files cannot be overwritten unless explicitly indicated
* Expand sound file compatibility: the software is JupyterLab/ notebook friendly but can only handle .wav files with 16 or 32 bitdepth
* Improve accessibility of Jupyter Notebooks (currently available on <a href="https://notebooks.ai/a-n-rose">notebooks.ai</a>)
* Error handling (especially of incompatible sound files)
* Adding more filters
* Adding more machine learning architectures
* Add more options for visualizations (e.g. <a href="https://en.wikipedia.org/wiki/Short-time_Fourier_transform">stft features</a>)
* Implement neural network with <a href="https://www.tensorflow.org/lite">TensorFlow Lite</a>
* Various platforms to store sample data (aside from Notebooks.ai)
* General speed and efficiency

# Examples 

You can run the examples below using ipython or other python console, or python script.

Install and run ipython:
```
(env)..$ pip install ipython
(env)..$ ipython
>>> # import what we need for the examples:
>>> import pysoundtool.explore_sound as exsound 
>>> import pysoundtool.prepsound as prepsound
>>> import pysoundtool as pyst 
>>> from pysoundtool.templates import soundclassifier
>>> from scipy.io.wavfile import write
```

## Visualization

### Time Domain

```
>>> exsound.visualize_signal('./audiodata/python.wav')
```
![Imgur](https://i.imgur.com/pz0MHui.png)

### Frequency Domain

#### Mel Frequency Cepstral Coefficients

```
>>> exsound.visualize_feats('./audiodata/python.wav', features='mfcc')
```
![Imgur](https://i.imgur.com/hx94jeQ.png)

#### Log-Mel Filterbank Energies
```
>>> exsound.visualize_feats('./audiodata/python.wav', features='fbank')
```
![Imgur](https://i.imgur.com/7CroE1i.png)

## Sound Creation

```
>>> data, sr = exsound.create_signal(freq=500, amplitude=0.5, samplerate=8000, dur_sec=1)
>>> exsound.visualize_signal(data, samplerate=sr)
```
![Imgur](https://i.imgur.com/qvWPrQu.png)

```
>>> exsound.visualize_feats(data, samplerate=sr, features='fbank')
```
![Imgur](https://i.imgur.com/HcjNHLH.png)

I am working on improving the x and y labels... I am used to using <a href="https://librosa.github.io/librosa/generated/librosa.display.specshow.html">Librosa.display.spechow</a> which did make this a bit easier... 

## Sound File Prep

### Convert to .wav file

```
>>> newfilename = soundprep.convert2wav('./audiodata/traffic.aiff')
>>> print(newfilename)
audiodata/traffic.wav
```

### Ensure sound data is mono channel

```
>>> from scipy.io.wavfile import read
>>> sr, data = read('./audiodata/python.wav')
>>> datamono = soundprep.stereo2mono(data) # if it is already, nothing will change
>>> len(data) == sum(data==datamono)
True
>>> sr, data_2channel = read('./audiodata/dogbark_2channels.wav')
>>> data_2channel.shape
(18672, 2)
>>> data_1channel = soundprep.stereo2mono(data_2channel)
>>> data_1channel.shape
(18672,)
>>> data_2channel[:5]
array([[208, 208],
       [229, 229],
       [315, 315],
       [345, 345],
       [347, 348]], dtype=int16)
>>> data_1channel[:5]
array([208, 229, 315, 345, 347], dtype=int16)
```

### Convert Soundfiles for use with scipy.io.wavfile

As of now, the software uses scipy.io.wavfile, a compatible module for Jupyter environments. If you have files that are not compatible, this should save the file / sound data as a compatible state.

```
>>> newfilename = soundprep.prep4scipywavfile('./audiodata/traffic.aiff')
Converting file to .wav
Saved file as audiodata/traffic.wav 
```

## Filtering 

NOTE: only .wav files of bit depth 16 or 32 can currently be used. See subsection <a href="https://github.com/a-n-rose/Python-Sound-Tool#convert-soundfiles-for-use-with-scipyiowavfile">'Convert Soundfiles for use with scipy.io.wavfile'</a>

### Noisy sound file

Add 'python' speech segment and traffic noise to create noisy speech. Save as .wav file.
```
>>> from scipy.io.wavfile import write
>>> speech = './audiodata/python.wav'
>>> noise = './audiodata/traffic.aiff'
>>> data_noisy, samplerate = soundprep.add_sound_to_signal(speech, noise, delay_target_sec=1, scale = 0.1)
>>> noisy_speech_filename = './audiodata/python_traffic.wav'
>>> write('./audiodata/python_traffic.wav', samplerate, data_noisy)
```
Then filter the traffic out:
```
>>> pyst.filtersignal(output_filename = 'python_traffic_filtered.wav',
                    wavfile = noisy_speech_filename,
                    scale = 1) # how strong the filter should be
```
If there is some distortion in the signal, try a post filter:
```
>>> pyst.filtersignal(output_filename = 'python_traffic_filtered_postfilter.wav',
                    wavfile = noisy_speech_filename,
                    scale = 1,
                    apply_postfilter = True) # how strong the filter should be
```

## Convolutional Neural Network: Simple sound classification

NOTE: only .wav files of bit depth 16 or 32 can currently be used. See subsection <a href="https://github.com/a-n-rose/Python-Sound-Tool#convert-soundfiles-for-use-with-scipyiowavfile">'Convert Soundfiles for use with scipy.io.wavfile'</a>

```
>>> from pysoundtool.templates import soundclassifier
>>> project_name = 'test_backgroundnoise_classifier'
>>> headpath = 'saved_features_and_models'
>>> audio_classes_dir = './audiodata/minidatasets/background_noise/'
>>> soundclassifier(project_name,
                headpath,
                audiodir = audio_classes_dir,
                feature_type = 'mfcc',
                target_wavfile = './audiodata/rain.wav')
```
Some model training stuff should print out... and at the end a label the sound was classified as:
```
Label classified:  cafe
```
