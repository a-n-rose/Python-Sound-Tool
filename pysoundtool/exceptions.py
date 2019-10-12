#!/bin/bash
# Copyright 2019 Peggy Sylopp und Aislyn Rose GbR
# All rights reserved
# This file is part of the  NoIze-framework
# The NoIze-framework is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by the  
# Free Software Foundation, either version 3 of the License, or (at your option) 
# any later version.
#
#@author Aislyn Rose
#@version 0.1
#@date 31.08.2019
#
# The  NoIze-framework  is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details. 
#
# You should have received a copy of the GNU AFFERO General Public License 
# along with the NoIze-framework. If not, see http://www.gnu.org/licenses/.

"""
The :mod:`pysoundtool.exceptions` module includes customized errors.
"""


def pathinvalid_error(path):
    raise FileNotFoundError("\nFileNotFoundError: \
        \nThe following path does not exist:\
        \n{}".format(path))


def noaudiofiles_error(path):
    raise FileNotFoundError("\nFileNotFoundError: \
        \nNo .wav files found at this destination:\
        \n{}".format(path))


def notsufficientdata_error(numtrain, numval, numtest, expected_numtrain):
    raise ValueError("\nValueError: Not enough training data\
        \nNumber train samples: {} (Minumum expected: {})\
        \nNumber val samples: {}\nNumber test samples: {}\
        \nPlease collect more audio data. There is not enough to\
        \nbuild a sufficient training, validation, and test dataset.".format(
        numtrain, expected_numtrain, numval, numtest))
