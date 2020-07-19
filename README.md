[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/a-n-rose/Python-Sound-Tool/master)

# PySoundTool

PySoundTool is an experimental framework for exploring sound as well as machine learning in the context of sound. For examples and to navigate the code, see the <a href="https://aislynrose.bitbucket.io/">documentation</a>. Note: as is, PySoundTool is not yet a stable framework, meaning changes might periodically be made without extreme focus on backwards compatibility.

Those who might find this useful: 

* speech and sound enthusiasts
* digital signal processing / mathematics / physics / acoustics enthusiasts
* researchers
* linguists
* psycholinguists

The main goal of PySoundTool is to provide the code and functionality with more context via visualization, research, and mathematics. Most of the resources used to build the functionality stems from publicly available research and datasets. (For a list of open datasets, see my ever growing <a href='https://a-n-rose.github.io/2019/01/06/resources-publicly-available-speech-databases.html'>collection</a>.)

As it covers quite a large range, from audio file conversion to implementation of trained neural networks, the purpose of PySoundTool is not to be the perfect implementation of all functions (although that is also a goal :P ), but rather a peak into how they *can* be implemented, hopefully offering people a foundation for trying out different ways of implementation (feature extraction, building neural networks, etc.).

This project is still in the beginning stages and has a lot of room for growth, especially from people with data science, computer science, physics, acoustics backgrounds / knowledge.

# Installation

Clone this repository. Set the working directory where you clone this repository.

Start a virtual environment:

```
$ python3 -m venv env
```
or, to better control your python version:
```
$ virtualenv -p python3.8 env
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

You can run the examples below using ipython or other python console, or python script.

Install and run ipython:
```
(env)..$ pip install ipython
(env)..$ ipython
>>> # import what we need for the examples:
>>> import pysoundtool as pyst
```
See the <a href="https://aislynrose.bitbucket.io/example_cases.html">examples</a> in the documentation for code to try out.

## Example datasets

If you would like to play around with various types of sound, check out my <a href='https://github.com/a-n-rose/mini-audio-datasets'>repo</a> containing minid datasets of sound, ranging from speech to noise. 

# About the Author

I studied clinical linguistics for my master's which introduced me to the inspiring world of open source software, python programming, and the beauty of acoustics. My interests broadened from academic experimental design to the development of tools for identifiying speech irregularities and beyond. Through this I discovered a bit of a missing niche in the Python community: a high-level sound tool for filtering, analysis, **and** deep learning that also offers context to its functionality. You can read more about me and my projects on my <a href="https://a-n-rose.github.io/">blog</a>.

# The Beginnings of PySoundTool

This project stemmed from the Prototype Fund project <a href="https://github.com/pgys/NoIze">NoIze</a> which was <a href="https://www.youtube.com/watch?v=BJ0f2x49Imc&feature=youtu.be">presented</a> at PyConDE / PyData Berlin in 2019. This fork broadens the application of the software from smart noise filtering to general sound analysis, filtering, visualization, preparation, etc. Therefore the name has been adapted to more general sound functionality.
