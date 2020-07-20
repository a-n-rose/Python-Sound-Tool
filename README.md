[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/a-n-rose/Python-Sound-Tool/master)
![License](https://img.shields.io/badge/license-GNU%20AGPL-brightgreen)(https://github.com/a-n-rose/Python-Sound-Tool/blob/master/LICENSE.md)


# PySoundTool

PySoundTool is an experimental framework (very much beta state) for exploring sound as well as machine learning in the context of sound. For examples and to navigate the code, see the <a href="https://aislynrose.bitbucket.io/">documentation</a>. Note: as is, PySoundTool is not yet a stable framework, meaning changes might periodically be made without extreme focus on backwards compatibility. 

Those who might find this useful: 

* speech and sound enthusiasts
* digital signal processing / mathematics / physics / acoustics enthusiasts
* deep learning enthusiasts
* researchers
* linguists
* psycholinguists

The main goal of PySoundTool is to provide the code and functionality with more context via visualization, research, and mathematics. Most of the resources used to build the functionality stems from publicly available research and datasets. (For a list of open datasets, see my ever growing <a href='https://a-n-rose.github.io/2019/01/06/resources-publicly-available-speech-databases.html'>collection</a>.)

As it covers quite a large range, from audio file conversion to implementation of trained neural networks, the purpose of PySoundTool is not to be the perfect implementation of all functions (although that is also a goal :P ), but rather a peak into how they *can* be implemented, hopefully offering others a foundation for trying out different ways of implementation (feature extraction, building neural networks, etc.).

This project is still in the beginning stages and has a lot of room for growth, especially with contributers having a background / knowlege in data science, computer science, machine and deep learning, physics, acoustics, or dsp. Contributers from other backgrounds are also welcome! If you'd like PySoundTool to do something it doesn't, try making it or create an issue.

# Installation

Clone this repository. Set the working directory where you clone this repository.

Start a virtual environment:

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
Then install necessary installations via pip:
```
(env)..$ pip install -r requirements.txt
```

# Examples 

You can explore example code via:

* Binder (albeit limited as some packages cannot be loaded into online environments)
* locally on your machine via Jupyter Notebook
* <a href="https://aislynrose.bitbucket.io/example_cases.html">examples</a> in the documentation. 

### Binder

Click on the Binder badge at the top of this README (it might take a while to load) and venture into the folder `binder_notebooks`.

### Locally via Jupyter Notebook:

Install and run jupyter notebook:

```
(env)..$ pip install notebook
(env)..$ jupyter notebook
```

Venture into the folder `jupyter_notebooks` and have a go!

## Example datasets

If you would like to play around with various types of sound, check out my <a href='https://github.com/a-n-rose/mini-audio-datasets'>repo</a> containing mini datasets of sound, ranging from speech to noise. They are very small so don't expect much as it comes to training neural networks. These example datasets are used in some of the documentation <a href="https://aislynrose.bitbucket.io/example_cases.html">examples</a>.

# About the Author

I studied clinical linguistics for my master's which introduced me to the inspiring world of open source software, python programming, and the beauty of acoustics. My interests broadened from academic experimental design to the development of tools for identifiying speech irregularities and beyond. Through this I discovered a bit of a missing niche in the Python community: a high-level sound tool for filtering, analysis, **and** deep learning that also offers context to its functionality. You can read more about me and my projects on my <a href="https://a-n-rose.github.io/">blog</a>.

# The Beginnings of PySoundTool

This project stemmed from the Prototype Fund project <a href="https://github.com/pgys/NoIze">NoIze</a> which was <a href="https://www.youtube.com/watch?v=BJ0f2x49Imc&feature=youtu.be">presented</a> at PyConDE / PyData Berlin in 2019. This fork broadens the application of the software from smart noise filtering to general sound analysis, filtering, visualization, preparation, etc. Therefore the name has been adapted to more general sound functionality.
