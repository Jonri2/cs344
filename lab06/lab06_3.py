"""
This program prints the number of testing and training examples in the boston_housing dataset and
prints the rank, shape, and data type of the examples

@author: jde27
@version 15mar2020
"""

from keras.datasets import boston_housing


def print_structures():
    """Function adapted from numpy.ipynb"""
    print(
        'training examples \
            \n\tcount: {} \
            \n\tdimensions: {} \
            \n\tshape: {} \
            \n\tdata type: {}\n\n'.format(
                len(train_examples),
                train_examples.ndim,
                train_examples.shape,
                train_examples.dtype
        ),
        'testing examples \
            \n\tcount: {} \
            \n\tdimensions: {} \
            \n\tshape: {} \
            \n\tdata type: {}\n'.format(
                len(test_labels),
                train_labels.ndim,
                test_labels.shape,
                test_labels.dtype
        )
    )


(train_examples, train_labels), (test_examples, test_labels) = boston_housing.load_data()
print_structures()
