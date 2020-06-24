# coding: utf-8
"""
========================================
Audio Dataset Exploration and Formatting
========================================

This notebook offers an example for how you can use PySoundTool to examine 
audio files within a dataset, and to reformat them if desired.
"""


###############################################################################################
#  
# Dataset Exploration
# ^^^^^^^^^^^^^^^^^^^


##########################################################
# Ignore this snippet of code: it is only for this example
import os
package_dir = '../../../'
os.chdir(package_dir)

#####################################################################
# Let's import pysoundtool, assuming it is in your working directory:
import pysoundtool as pyst
import matplotlib.pyplot as plt

#########################################################################
# Let's create a signal:
sig1, sr = pyst.dsp.generate_sound()
plt.plot(sig1)
plt.show()
