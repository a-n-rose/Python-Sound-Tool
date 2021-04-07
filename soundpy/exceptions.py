"""
The :mod:`soundpy.exceptions` module includes customized errors.
"""

def notsufficientdata_error(numtrain, numval, numtest, expected_numtrain):
    raise ValueError('Not enough training data:'+\
        '\nNumber train samples: {} '.format(numtrain)+\
            '(Minumum expected: {})'.format(expected_numtrain)+\
        '\nNumber val samples: {}'.format(numval)+\
            '\nNumber test samples: {}'.format(numtest) +\
                '\n\nPlease lower `perc_train` or collect more audio data.')

def numfeatures_incompatible_templatemodel():
    raise ValueError('ERROR: Number of features is incompatible with the template model. '+\
        'Try a higher number or rely on the defaults. Apologies for this inconvenience.')
