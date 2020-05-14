[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/a-n-rose/Python-Sound-Tool/master)

# PySoundTool

This project stemmed from the Prototype Fund project <a href="https://github.com/pgys/NoIze">NoIze</a>. This fork broadens the application of the software from smart noise filtering to general sound analysis, filtering, visualization, preparation, etc. Therefore the name has been adapted to more general sound functionality.

Note: for adjusting sound files, **apply only to copies of the originals**. Improvements need to be made to ensure files don't get overwritten except explicitly indicated.

## Functionality

* <a href= "https://github.com/a-n-rose/Python-Sound-Tool#visualization">Visualize signals</a>
* <a href="https://github.com/a-n-rose/Python-Sound-Tool#python-frequency-domain">Visualize feature extraction</a> (i.e. for machine and deep learning)
* <a href="https://github.com/a-n-rose/Python-Sound-Tool#sound-creation">Sound creation</a>
* <a href="https://github.com/a-n-rose/Python-Sound-Tool#adding-sounds">Sound manipulation</a> (e.g. adding sounds)
* <a href="https://github.com/a-n-rose/Python-Sound-Tool#filtering">Filtering</a>
* <a href="https://github.com/a-n-rose/Python-Sound-Tool#convolutional-neural-network-simple-sound-classification">Machine Learning</a> (i.e. Convolutional Neural Network)
* <a href="https://github.com/a-n-rose/Python-Sound-Tool#sound-file-prep">Sound data adjustment</a> (e.g. file type, bit depth, etc.). This is useful if you would like to use new sound files with these tools but that aren't compatible with scipy.io.wavfile (a Jupyter environment friendly module).

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
Feel free to use this tool in your own scripts (I show examples below). You can also explore *some* of its functionality via jupyter notebook:
```
(env)..$ jupyter notebook
```
Once this loads, click on the folder 'jupyternotebooks', and then on one of the .ipynb files.

# Examples 

You can run the examples below using ipython or other python console, or python script.

Install and run ipython:
```
(env)..$ pip install ipython
(env)..$ ipython
>>> # import what we need for the examples:
>>> import pysoundtool.explore_sound as exsound 
>>> import pysoundtool.soundprep as soundprep
>>> import pysoundtool as pyst 
>>> from pysoundtool.templates import soundclassifier
>>> from scipy.io.wavfile import write
```

## Visualization

### "Python": Time Domain

```
>>> exsound.visualize_audio('./audiodata/python.wav', feature_type='signal')
```
![Imgur](https://i.imgur.com/pz0MHui.png)

### "Python": Frequency Domain

The mel frequecy cepstral coefficients (MFCCs) and log-mel filterbank energies (FBANK) are two very common acoustic features used in machine and deep learning.

Let's take a look and see how the word "python" looks when these features are extracted and also how window settings influence the features.

#### MFCC: 

##### window 20 ms, window overlap 10 ms (default)

```
>>> exsound.visualize_audio('./audiodata/python.wav', feature_type='mfcc')
```
![Imgur](https://i.imgur.com/hx94jeQ.png)

##### window 100 ms, window overlap 50 ms

```
>>> exsound.visualize_audio('./audiodata/python.wav', feature_type='mfcc',
                            win_size_ms = 100, win_shift_ms = 50)
```
![Imgur](https://i.imgur.com/XY2rGQj.png)


#### FBANK: 

##### window 20 ms, window overlap 10 ms (default)
```
>>> exsound.visualize_audio('./audiodata/python.wav', feature_type='fbank')
```
![Imgur](https://i.imgur.com/7CroE1i.png)

##### window 100 ms, window overlap 50 ms 

```
>>> exsound.visualize_audio('./audiodata/python.wav', feature_type='fbank',
                            win_size_ms = 100, win_shift_ms = 50)
```
![Imgur](https://i.imgur.com/r3jVPdB.png)

## Sound Creation

```
>>> data, sr = exsound.create_signal(freq=500, amplitude=0.5, samplerate=8000, dur_sec=0.2)
>>> data2, sr = exsound.create_signal(freq=1200, amplitude=0.9, samplerate=8000, dur_sec=0.2)
>>> data3, sr = exsound.create_signal(freq=200, amplitude=0.3, samplerate=8000, dur_sec=0.2)
>>> data_mixed = data + data2 + data3
>>> exsound.visualize_audio(data_mixed, feature_type='signal', samplerate=sr)
```
![Imgur](https://i.imgur.com/UYU0Ft0.png)

Mixed with noise:
```
>>> noise = exsound.create_noise(len(data_mixed), amplitude=0.1)
>>> data_noisy = data_mixed + noise
>>> exsound.visualize_audio(data_noisy, feature_type='signal', samplerate = sr)
```
![Imgur](https://i.imgur.com/ZvyAdUZ.png)

In the time domain, it is difficult to see the three different signals at all...

```
>>> exsound.visualize_audio(data_noisy, samplerate=sr, feature_type='fbank')
```
![Imgur](https://i.imgur.com/RaDhEfq.png)

In the frequency domain, you can see that there are distinct frequencies in the signal, approximately 3.

Note: I am working on improving the x and y labels... I am used to using <a href="https://librosa.github.io/librosa/generated/librosa.display.specshow.html">Librosa.display.spechow</a> which did make this a bit easier... 

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

## Adding sounds

### Long background sound

Here we will add traffic background noise to the speech segment 'python'.

'python':
```
>>> python, sr = soundprep.loadsound('./audiodata/python.wav')
>>> exsound.visualize_audio(python, feature_type='signal', samplerate=sr)
```
![Imgur](https://i.imgur.com/pz0MHui.png)

traffic background noise:
```
>>> traffic, sr = soundprep.loadsound('./audiodata/traffic.aiff')
Step 1: ensure filetype is compatible with scipy library
Success!
>>> exsound.visualize_audio(traffic, feature_type='signal', samplerate=sr)
```
![Imgur](https://i.imgur.com/rFn3x2Y.png)

Combining them: 
* noise/sound to a scale of 0.3
* 1 second delay for the speech
* total length: 5 seconds
```
>>> python_traffic, sr = soundprep.add_sound_to_signal(
                                signal = './audiodata/python.wav',
                                sound = './audiodata/traffic.aiff',
                                scale = 0.3,
                                delay_target_sec = 1,
                                total_len_sec = 5)
>>> exsound.visualize_audio(python_traffic, feature_type='signal', samplerate=sr)
```
![Imgur](https://i.imgur.com/z4E2HSj.png)

### Short background sound

rain background noise:
```
>>> rain, sr = soundprep.loadsound('./audiodata/rain.wav')
>>> exsound.visualize_audio(rain, feature_type='signal', samplerate=sr)
```
![Imgur](https://i.imgur.com/kmmwZta.png)

This sound will be repeated to match the desired background noise length
Note: sometimes artifacts can occur and may need additional processing. Longer
background noises are more ideal.
```
>>> python_rain, sr = soundprep.add_sound_to_signal(
                                signal = './audiodata/python.wav',
                                sound = './audiodata/rain.wav',
                                scale = 0.3,
                                delay_target_sec = 1,
                                total_len_sec = 5)
>>> exsound.visualize_audio(python_rain, feature_type='signal', samplerate=sr)
```
![Imgur](https://i.imgur.com/BbdbDyu.png)

## Filtering 

NOTE: only .wav files of bit depth 16 or 32 can currently be used. See subsection <a href="https://github.com/a-n-rose/Python-Sound-Tool#convert-soundfiles-for-use-with-scipyiowavfile">'Convert Soundfiles for use with scipy.io.wavfile'</a>

For visuals, we will look at the sound as their FBANK features.

### Noisy sound file

Add 'python' speech segment and traffic noise to create noisy speech. Save as .wav file.
```
>>> from scipy.io.wavfile import write
>>> speech = './audiodata/python.wav'
>>> noise = './audiodata/traffic.aiff'
>>> data_noisy, samplerate = soundprep.add_sound_to_signal(speech, noise, delay_target_sec=1, scale = 0.3, total_len_sec=5)
>>> noisy_speech_filename = './audiodata/python_traffic.wav'
>>> write(noisy_speech_filename, samplerate, data_noisy)
>>> exsound.visualize_audio(noisy_speech_filename, feature_type='fbank')
```
![Imgur](https://i.imgur.com/X9whovI.png)

![Imgur](https://i.imgur.com/9G10mdb.png)

Then filter the traffic out:

![Imgur](https://i.imgur.com/FOcjwAl.png)

This is what the noise power spectrum of the full FFT looks like:

![Imgur](https://i.imgur.com/7CIiTfM.png)

If you set `real_signal` to true, this is what the noise power spectrum looks like:

![Imgur](https://i.imgur.com/6AWr5dV.png)

In numpy, you can use the full fft signal by using numpy.fft.fft or you can use the real fft, for audio signals for example, by using numpy.fft.rfft. The latter may be more efficent and there isn't a difference between the two. I've seen some Implement the full fft and others the rfft.

#### Wiener filter

```
>>> pyst.filtersignal(output_filename = 'python_traffic_wiener_filter.wav',
                    audiofile = noisy_speech_filename,
                    filter_type = 'wiener',
                    filter_scale = 1) # how strong the filter should be
```
What the filtered signal looks like in raw samples, power spectrum (basically stft), and fbank features: 

![Imgur](https://i.imgur.com/42liCr1.png)

![Imgur](https://i.imgur.com/tx87UEL.png)

![Imgur](https://i.imgur.com/TrwKJ4j.png)



#### Wiener filter with postfilter

If there is some distortion in the signal, try a post filter:
```
>>> pyst.filtersignal(output_filename = 'python_traffic_wiener_postfilter.wav',
                    audiofile = noisy_speech_filename,
                    filter_type = 'wiener_postfilter',
                    filter_scale = 1, # how strong the filter should be
                    apply_postfilter = True) 
```
What the filtered signal looks like in raw samples, power spectrum (basically stft), and fbank features: 

![Imgur](https://i.imgur.com/zTR4kX3.png)

![Imgur](https://i.imgur.com/lKe4dRQ.png)

![Imgur](https://i.imgur.com/AwontYt.png)

#### Band spectral subtraction filter

For comparison, try a band spectral subtraction filter:
```
>>> pyst.filtersignal(output_filename = 'python_traffic_bandspecsub.wav',
                    audiofile = noisy_speech_filename,
                    filter_type = 'band_spectral_subtracion',
                    filter_scale = 1, # how strong the filter should be
                    num_bands = 6) 
```
What the filtered signal looks like in raw samples, power spectrum (basically stft), and fbank features: 

![Imgur](https://i.imgur.com/Kg9cR2S.png)

![Imgur](https://i.imgur.com/jSX4ijV.png)

![Imgur](https://i.imgur.com/cFdaGLl.png)

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

# ToDo

* **Ensure** files cannot be overwritten unless explicitly indicated
* Expand sound file compatibility: the software is JupyterLab/ notebook friendly but can only handle .wav files with 16 or 32 bitdepth
* Improve accessibility of Jupyter Notebooks. Currently available on <a href="https://notebooks.ai/a-n-rose">notebooks.ai</a> (must have an account) and <a href="https://mybinder.org/v2/gh/a-n-rose/Python-Sound-Tool/master">Binder</a>(due to audiodata a bit slow)
* Error handling (especially of incompatible sound files)
* Adding more filters
* Adding more machine learning architectures
* Add more options for visualizations (e.g. <a href="https://en.wikipedia.org/wiki/Short-time_Fourier_transform">stft features</a>)
* Implement neural network with <a href="https://www.tensorflow.org/lite">TensorFlow Lite</a>
* Various platforms to store sample data (aside from Notebooks.ai and GitHub :P )
* Increase general speed and efficiency
