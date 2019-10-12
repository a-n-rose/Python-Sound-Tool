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

'''The matrixfun module offers insight into the matrix manipulation
necessary in both digital signal processing and machine learning
'''
import numpy as np

# TODO: https://github.com/biopython/biopython/issues/1496
# Fix numpy array repr for Doctest. 
def add_tensor(matrix):
    '''Adds tensor / dimension to input ndarray.

    Keras requires an extra dimension at some layers, which represents 
    the 'tensor' encapsulating the data. 

    Further clarification taking the example below. The input matrix has 
    shape (2,3,4). Think of it as 2 different events, each having
    3 sets of measurements, with each of those having 4 features. So, 
    let's measure differences between 2 cities at 3 different times of
    day. Let's take measurements at 08:00, 14:00, and 19:00 in... 
    Magic City and Never-ever Town. We'll measure.. 1) tempurature, 
    2) wind speed 3) light level 4) noise level.

    How I best understand it, putting our measurements into a matrix
    with an added dimension/tensor, this highlights the separate 
    measurements, telling the algorithm: yes, these are 4 features
    from the same city, BUT they occur at different times. Or it's 
    just how Keras set up the code :P 

    Parameters
    ----------
    matrix : numpy.ndarray
        The `matrix` holds the numerical data to add a dimension to.

    Returns
    -------
    matrix : numpy.ndarray
        The `matrix` with an additional dimension.

    Examples
    --------
    >>> import numpy as np
    >>> matrix = np.arange(24).reshape((2,3,4))
    >>> matrix.shape
    (2, 3, 4)
    >>> matrix
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    <BLANKLINE>
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
    >>> matrix_2 = add_tensor(matrix)
    >>> matrix_2.shape
    (2, 3, 4, 1)
    >>> matrix_2
    array([[[[ 0],
             [ 1],
             [ 2],
             [ 3]],
    <BLANKLINE>
            [[ 4],
             [ 5],
             [ 6],
             [ 7]],
    <BLANKLINE>
            [[ 8],
             [ 9],
             [10],
             [11]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[12],
             [13],
             [14],
             [15]],
    <BLANKLINE>
            [[16],
             [17],
             [18],
             [19]],
    <BLANKLINE>
            [[20],
             [21],
             [22],
             [23]]]])
    '''
    if isinstance(matrix, np.ndarray) and len(matrix) > 0:
        matrix = matrix.reshape(matrix.shape + (1,))
        return matrix
    elif isinstance(matrix, np.ndarray):
        raise ValueError('Input matrix is empty.')
    else:
        raise TypeError('Expected type numpy.ndarray, recieved {}'.format(
            type(matrix)))

def create_empty_matrix(shape, complex_vals=False):
    '''Allows creation of a matrix filled with real or complex zeros.

    In digital signal processing, complex numbers are common; it is 
    important to note that if complex_vals=False and complex values are
    inserted into the matrix, the imaginary part will be removed.

    Parameters
    ----------
    shape : tuple or int
        tuple or int indicating the shape or length of desired matrix or
        vector, respectively
    complex_vals : bool
        indicator of whether or not the matrix will receive real or complex
        values (default False)

    Returns
    ----------
    matrix : ndarray
        a matrix filled with real or complex zeros

    Examples
    ----------
    >>> matrix = create_empty_matrix((3,4))
    >>> matrix
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])
    >>> matrix_complex = create_empty_matrix((3,4),complex_vals=True)
    >>> matrix_complex
    array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
    >>> vector = create_empty_matrix(5,)
    >>> vector
    array([0., 0., 0., 0., 0.])
    '''
    if complex_vals:
        matrix = np.zeros(shape, dtype=np.complex_)
    else:
        matrix = np.zeros(shape, dtype=float)
    return matrix

def separate_dependent_var(matrix):
    '''Separates matrix into features and labels.

    Assumes the last column of the last dimension of the matrix constitutes
    the dependent variable (labels), and all other columns the indpendent variables
    (features). Additionally, it is assumed that for each block of data, 
    only one label is needed; therefore, just the first label is taken for 
    each block.

    Parameters
    ----------
    matrix : numpy.ndarray
        The `matrix` holds the numerical data to separate

    Returns
    -------
    X : numpy.ndarray
        A matrix holding the (assumed) independent variables
    y : numpy.ndarray, numpy.int64, numpy.float64
        A vector holding the labels assigned to the independent variables.
        If only one value in array, just the value inside is returned

    Examples
    --------
    >>> import numpy as np
    >>> #vector
    >>> separate_dependent_var(np.array([1,2,3,4]))
    (array([1, 2, 3]), 4)
    >>> #simple matrix
    >>> matrix = np.arange(4).reshape(2,2)
    >>> matrix
    array([[0, 1],
           [2, 3]])
    >>> X, y = separate_dependent_var(matrix)
    >>> X
    array([[0],
           [2]])
    >>> y 
    1
    >>> #more complex matrix
    >>> matrix = np.arange(20).reshape((2,2,5))
    >>> matrix
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9]],
    <BLANKLINE>
           [[10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19]]])
    >>> X, y = separate_dependent_var(matrix)
    >>> X
    array([[[ 0,  1,  2,  3],
            [ 5,  6,  7,  8]],
    <BLANKLINE>
           [[10, 11, 12, 13],
            [15, 16, 17, 18]]])
    >>> y
    array([ 4, 14])
    '''
    # get last column
    y_step1 = np.take(matrix, -1, axis=-1)
    # because the label is the same for each block of data, just need the first
    # row,  not all the rows, as they are the same label.
    y = np.take(y_step1, 0, axis=-1)
    # get features:
    X = np.delete(matrix, -1, axis=-1)
    return X, y

if __name__ == "__main__":
    import doctest
    doctest.testmod()
