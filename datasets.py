# numpy
import numpy as np

# sklearn
from sklearn import datasets
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split


# datasets
class Datasets():

    """
    Description:
        Holds different classification datasets
    """

    # constructor
    def __init__(self, test_size, random_state):

        """
        Description:
            Constructor for our Datasets class
        
        Parameters:
            test_size: percentage of data to be allocated for testing
            random_state: random state chosen for reproducible output
        
        Returns:
            None
        """

        self.test_size = test_size
        self.random_state = random_state

    # breast cancer
    def load_breast_cancer(self):

        """
        Description:
            Loads sklearn's Breast Cancer Dataset

        Parameters:
            None
        
        Returns:
            X, y, feature_names, class_names, X_train, X_test, y_train, y_test
        """
        
        # load dataset
        data = datasets.load_breast_cancer()

        # load features, labels, and class names
        X, y, feature_names, class_names = data.data, data.target, data.feature_names, data.target_names
        # normalize X features to range = [0, 1] to avoid explosive computations
        X = minmax_scale(X, feature_range = (0, 1), axis = 0, copy = True)
        # perform train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state = self.random_state)

        # return
        return X, y, feature_names, class_names, X_train, X_test, y_train, y_test

    # simple xy
    def simple_Xy(self, n_samples, n_features, n_classes):
        
        """
        Description:
            Simple X vs. y dataset created by sklearn
        
        Parameters:
            n_samples: number of samples to genarate
            n_features: number of features (corresponds to number of dimension)
            n_classes: number of classes for our dummy classification dataset
        
        Returns:
            X, y, X_train, X_test, y_train, y_test
        """

        # create dataset
        X, y = datasets.make_classification(n_samples = n_samples, n_features = n_features, n_classes = n_classes,
                                                                                                    random_state = self.random_state)

        # normalize both X & y to range = [0, 1] to avoid explosive computations
        X = minmax_scale(X, feature_range = (0, 1), axis = 0, copy = True)
        # y = minmax_scale(y, feature_range = (0, 1), axis = 0, copy = True)

        # perform train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state = self.random_state)

        # return
        return X, y, X_train, X_test, y_train, y_test

    # diabetes
    def load_diabetes(self):

        """
        Description:
            Loads sklearn's Diabetes Dataset

        Parameters:
            None
        
        Returns:
            X, y_classification, feature_names, class_names, X_train, X_test, y_train, y_test
        """

        # This is a regression dataset but we will convert it to classification

        # load the dataset
        data = datasets.load_diabetes()

        # load features, "labels", and class names
        X, y, feature_names = data.data, data.target, data.feature_names

        # normalize X features to range = [0, 1] to avoid explosive computations
        X = minmax_scale(X, feature_range = (0, 1), axis = 0, copy = True)

        # convert y to labels we want using median, if valua > median assign True, else False
        # we will use 1 (True) for has diabetes and 0 (False) for no diabetes
        y_classification = np.array([y > np.median(y)]).reshape(-1)   # (y > y.median()).astype(int)

        class_names = ['non-diabetic', 'diabetic']

        # perform train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y_classification, test_size = self.test_size, random_state = self.random_state)

        # return
        return X, y_classification, feature_names, class_names, X_train, X_test, y_train, y_test

