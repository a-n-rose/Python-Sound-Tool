# PySoundTool

This project stemmed from the Prototype Fund project <a href="https://github.com/pgys/NoIze">NoIze</a>. This fork broadens the application of the software from smart noise filtering to general sound analysis, filtering, visualization, preparation, etc. Therefore the name has been adapted to more general sound functionality.

Note: for adjusting sound files, **apply only to copies of the originals**. Improvements need to be made to ensure files don't get overwritten except explicitly indicated. 

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
* Expand sound file compatibility: the software is JupyterLab/ notebook friendly but can only handle mono channel .wav files with 16 or 32 bitdepth
* Improve accessibility of Jupyter Notebooks (currently available on <a href="https://notebooks.ai/a-n-rose">notebooks.ai</a>)
* Error handling (especially of incompatible sound files)
* Adding more filters
* Adding more machine learning architectures
* Add more options for visualizations (e.g. <a href="https://en.wikipedia.org/wiki/Short-time_Fourier_transform">stft features</a>)
* Implement neural network with <a href="https://www.tensorflow.org/lite">TensorFlow Lite</a>
* Various platforms to store sample data (aside from Notebooks.ai)
* General speed and efficiency

# Examples (in-the-works)

On my TODO list..

## Visualization

TBC

### Time Domain

TBC

### Frequency Domain

TBC

## Sound Creation

TBC

## Sound File Prep

TBC

## Wiener filer

TBC

## Convolutional Neural Network: Simple sound classification

TBC
