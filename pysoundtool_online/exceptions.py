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


class VersionError(Exception):
    msg = 'This functionality is not available in PySoundTool-Online.'
    def __init__(self, message=msg):
        self.message = message
        super().__init__(self.message)
    pass

