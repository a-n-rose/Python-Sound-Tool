[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/a-n-rose/Python-Sound-Tool/master)

# PySoundTool

PySoundTool is a framework for exploring sound as well as machine learning in the context of sound. For exmples and to navigate the code, see the <a href="https://aislynrose.bitbucket.io/">documentation</a>.

The goal is to make this exploration more accessible to people who didn't study acoustics, digital signal processing, or computer science but who are interested in these fields. 

A large portion of the functionality stems from previous research conducted in these fields, referenced within the functions themselves.

This is a project in development that can improve with contributions from people of all backgrounds, ranging from documentation readability to the efficiency, reliability, and expansion of functionality.

# Installation

This repository serves as a place to explore sound. Therefore, small sound datasets are included in this repo. The size is appx. 30MB. If you clone this repo, this sound data will be cloned as well.

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

# About the Author

I studied clinical linguistics for my master's which introduced me to the inspiring world of open source software, python programming, and the beauty of acoustics. My interests broadened from academic experimental design to the development of tools for identifiying speech irregularities and beyond. Through this I discovered a bit of a missing niche in the Python community: a high-level sound tool for filtering, analysis, and deep learning that also offers context to its functionality. You can read more about me and my projects on my <a href="https://a-n-rose.github.io/">blog</a>.

# The Beginnings of PySoundTool

This project stemmed from the Prototype Fund project <a href="https://github.com/pgys/NoIze">NoIze</a> which was <a href="https://www.youtube.com/watch?v=BJ0f2x49Imc&feature=youtu.be">presented</a> at PyConDE / PyData Berlin in 2019. This fork broadens the application of the software from smart noise filtering to general sound analysis, filtering, visualization, preparation, etc. Therefore the name has been adapted to more general sound functionality.

