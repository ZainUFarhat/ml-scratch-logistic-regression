# numpy
import numpy as np

# logistic regression
class LogisticRegression():

    """
    Description:
        My from scratch implementation of the logistic Regression Algorithm
    """

    # constructor
    def __init__(self, epochs, lr):

        """
        Description:
            Contructor for our LogisticRegression class
        
        Parameters:
            epochs: number of training iterations
        
        Returns:
            None
        """

        # epochs and learning rate
        self.epochs = epochs
        self.lr = lr
        # weights and biases
        self.w = []
        self.b = 0

    # sigmoid
    def sigmoid(self, x):
    
        """
        Description:
            Sigmoid Activation Function
        
        Parameters:
            x: input to sigmoid activation function

        Returns:
            sigmoid
        """

        # implement sigmoid function
        sigmoid = 1 / (1 + np.exp(-1 * x))

        # return
        return sigmoid

    # cross entropy loss
    def cross_entropy_loss(self, y, y_pred):
    
        """
        Description:
            Calculate the cross entropy loss between true and predicted labels

        Parameters:
            y: true labels
            y_pred: predicted labels

        Returns:
            cross_entropy_loss  
        """

        # we will set an epsilon value for clipping our arrays
        eps = 1e-15

        # we will clip the values predicted to avoid log(0) computational error
        # we'll choose epsilon as a min value and 1 - epsilon as a max value
        # since we have probabilites for y_pred we can set min and max in range (0, 1)
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # calculate loss 
        cross_entropy_loss = -1 * np.sum(y * np.log(y_pred))

        # return
        return cross_entropy_loss
    
    # fit
    # train + simple gradient descent for optimization
    def fit(self, X, y):

        """
        Description:
            Fits the training data of our Logistic Regression model

        Parameters:
            X: features
            y: labels
        
        Returns:
            cross_entropy_losses
        """

        # extract number of training samples N and number of features
        N, num_features = X.shape

        # intialize weights to zero
        self.w = np.zeros(num_features)

        # cross entropy losses
        cross_entropy_losses = []
        
        for _ in range(self.epochs):

            # first step is a linear prediction
            linear_prediction = np.dot(X, self.w) + self.b

            # find predictions
            y_preds = self.sigmoid(linear_prediction)

            # calculate cross entropy loss and append it to list
            cross_entropy = self.cross_entropy_loss(y, y_preds)
            cross_entropy_losses.append(cross_entropy)

            # compute the first derivatives of cross entropy loss with respect to weights and bias respectively
            dw = (1/N) * 2 * np.dot(X.T, (y_preds - y))
            db = (1/N) * 2 * np.sum(y_preds - y)

            # update weights and biases
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
        
        # return
        return cross_entropy_losses
    
    # predict
    def predict(self, X):

        """
        Description:
            Predict based on the fitted Logistic Regression model

        Parameters:
            X: unseen data we want to predict on
        
        Returns:
            predicted_classes
        """

        # first step is a linear prediction
        linear_prediction = np.dot(X, self.w) + self.b

        # find predictions
        y_pred = self.sigmoid(linear_prediction)

        # assign classes to our predictions, remember the output of sigmoid activation is a probability
        # if prediction is less than 0.5 it takes class 0 else class 1
        predicted_classes = [0 if pred <= 0.5 else 1 for pred in y_pred]

        # return
        return predicted_classes