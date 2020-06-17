"""
The :mod:`pysoundtool.exceptions` module includes customized errors.
"""

def notsufficientdata_error(numtrain, numval, numtest, expected_numtrain):
    raise ValueError("\nValueError: Not enough training data\
        \nNumber train samples: {} (Minumum expected: {})\
        \nNumber val samples: {}\nNumber test samples: {}\
        \nPlease collect more audio data. There is not enough to\
        \nbuild a sufficient training, validation, and test dataset.".format(
        numtrain, expected_numtrain, numval, numtest))
