from abc import ABC, abstractmethod

"""
    BluePrint module for defining common methods of the TibLib package classes,
    every class that represents a data pipeline step should inherith a blueprint from this module,
    the common pipeline steps should be:
    Source -> Pipe -> ... -> Pipe -> Faucet 
"""


class NoFitError(Exception):
    def __init__(self, message="Data was not fitted"):
        super().__init__(message)


class Source(ABC):
    """
    Source step, the initial step of the data pipeline, can take any object
    in input and outputs a data matrix
    """
    @abstractmethod
    def fit(self, x, y):
        """
        generic fit method for source step
        :param x: Object
        :param y: Object
        :return: None
        """

    def transform(self, x):
        """
        generic transform method for source step
        :param x: Object
        :return: numpy 2D array, columns are the features and rows the data samples
        """

    def fit_transform(self, x, y):
        """
         To be implemented as a fit and transform on the same input data, see fit and transform for details
        :param x:
        :param y:
        :return:
        """


class Pipe(ABC):
    """
    Pipeline Step, we refer as pipe, steps that recieve in input a data matrix and
    return in output a transformed data matrix
    See fit and transform methods for more details
    """

    @abstractmethod
    def fit(self, x, y):
        """
        Generic fit method used to train the Pipe
        :param x: numpy 2D array, columns should be the features and rows the data samples
        :param y: numpy 1D array of labels referring to rows of x
        :return: None
        """

    @abstractmethod
    def transform(self, x):
        """
        Generic transform method to transform a data matrix
        :param x: numpy 2D array, columns should be the features and rows the data samples
        :return: numpy 2D array
        """

    @abstractmethod
    def fit_transform(self, x, y):
        """
        To be implemented as a fit and transform on the same input data, see fit and transform for details
        :param x: numpy 2D array
        :param y: numpy 1D array
        :return: numpy 2D array
        """


class Faucet(ABC):
    """
    General Data ML algorithms, classificators, regressors or clusterers
    """

    @abstractmethod
    def fit(self, x, y):
        """
        Generic fit method used to train the Faucet
        :param x: numpy 2D array, columns should be the features and rows the data samples
        :param y: numpy 1D array of labels referring to rows of x
        :return: None
        """

    @abstractmethod
    def predict(self, x,  return_prob=False):
        """
        Generic Predict method used to label a dataset
        :param x: numpy 2D array, columns should be the features and rows the data samples
        :return: numpy 1D array of labels
        """

    @abstractmethod
    def fit_predict(self, x, y):
        """
        To be implemented as a fit and predict on the same input data, see fit and transform for details
        :param x: numpy 2D array
        :param y: numpy 1D array
        :return: numpy 1D array
        """