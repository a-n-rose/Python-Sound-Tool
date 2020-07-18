"""
The :mod:`pysoundtool.exceptions` module includes customized errors.
"""

def notsufficientdata_error(numtrain, numval, numtest, expected_numtrain):
    raise ValueError('Not enough training data:'+\
        '\nNumber train samples: {} '.format(numtrain)+\
            '(Minumum expected: {})'.format(expected_numtrain)+\
        '\nNumber val samples: {}'.format(numval)+\
            '\nNumber test samples: {}'.format(numtest) +\
                '\n\nPlease lower `perc_train` or collect more audio data.')
