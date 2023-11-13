# numpy
import numpy as np

# Logistic Regression
from LogisticRegression import *

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Calculate accuracy - out of 100 examples, what percentage does our model get right?
def accuracy_fn(y_true, y_pred):
  """
  calculates the accuracies of a given prediction

  Parameters:

    y_true: the true labels
    y_pred: our predicted labels
  
  Returns:

    accuracy
  """
  
  # find the number of correct predictions  
  correct = np.equal(y_true, y_pred).sum()
  # calculate the accuracy
  acc = (correct/len(y_pred))*100
  # return the accuracy
  return round(acc, 2) 

# cross entropy loss
def cross_entropy_loss(y, y_pred):
  
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

# scatter plot of given data
def scatter_plot(X, y, features, title, x_label, y_label, class_names, savepath):

    """
    Description:
        Plots a scatterplot based on X & y data provided

    Parameters:
        X: x-axis datapoints
        y: y-axis datapoints
        features: which 2 features to plot against
        title: tite of plot
        x_label: label for x axis
        y_label: label for y axis
        class_names: names of our target classes
        savepath: path to save our scatterplot to

    Returns:
        None
    """

    # intialize figure
    plt.figure(figsize = (7, 7))

    # set background color to lavender
    ax = plt.axes()
    ax.set_facecolor("lavender")

    # find features correspaonding to class labels
    class_0, class_1 = X[y == 0], X[y == 1]

    # scatter plots of class features against themselves
    plt.scatter(class_0[:, features[0]], class_0[:, features[1]], label = class_names[0], c = 'r')
    plt.scatter(class_1[:, features[0]], class_1[:, features[1]], label = class_names[1], c = 'b')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.legend()
    plt.savefig(savepath)

    # return
    return None

# visualize decision boundary
def visualize_decision_boundary(X, y, features, epochs, lr, title, x_label, y_label, class_names, resolution, savepath):
   
   """
   Description:
        Plot the decision doundary of a logistic regression model

   Parameters:
        X: features
        y: targets
        features: which 2 features to plot against
        epochs: number iterations to train our model on
        lr: model learning rate
        epochs: number of epochs logistic regression model was trained on
        x_label: x label of our plot
        y_label: y label of our plot
        resolution: resolution of grid for plotting the decision boundary
        savepath: path to save our decision plot boundary to

   Returns:
        None
   """

   # print(X.shape, y.shape)

   # initialize figure
   plt.figure(figsize = (7, 7))

   # scatter plot based on class labels
   # find features correspaonding to class labels
   class_0, class_1 = X[y == 0], X[y == 1]
   
   # scatter plots of class features against themselves
   plt.scatter(class_0[:, features[0]], class_0[:, features[1]], label = class_names[0], c = 'r')
   plt.scatter(class_1[:, features[0]], class_1[:, features[1]], label = class_names[1], c = 'b')
   plt.title(title)
   plt.xlabel(x_label)
   plt.ylabel(y_label)
   plt.grid()
   plt.legend()
   
   # get the min and max limits for our x-axis and y-axis
   # here the two features chosen for X_train_reduced will compose the x-axis and y-axis respectively
   x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
   y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   
   # we will create a meshgrid based on the x-axis and y-axis range
   # we will take resolution steps between min and max and we are aiming to classify all of these into our given labels
   xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))

   # fit a logistic regression model on reduced data
   model = LogisticRegression(epochs = epochs, lr = lr)
   model.fit(X, y)
   
   # predict the labels for all points on the created meshgrid
   predictions = model.predict(np.c_[xx.ravel(), yy.ravel()])
   Z = np.array(predictions)
   Z = Z.reshape(xx.shape)

   # Original blue and red colors
   original_blue = (0.0, 0.0, 1.0)  # RGB values for blue
   original_red = (1.0, 0.0, 0.0)   # RGB values for red

   # Adjust the brightness (increase the value for a brighter color)
   brighter_factor = 1.5
   brighter_blue = tuple(min(1.0, c * brighter_factor) for c in original_blue)
   brighter_red = tuple(min(1.0, c * brighter_factor) for c in original_red)

   # Create a custom colormap with the brighter colors
   cmap_brighter = ListedColormap([brighter_red, brighter_blue])

   # Plot the decision boundary
   plt.pcolormesh(xx, yy, Z, cmap = cmap_brighter, alpha = 0.3)

   plt.xlim(-0.1, 1.1)
   plt.ylim(-0.1, 1.1)

   # save fig
   plt.savefig(savepath)

   # return
   return None

# plot training curve
def plot_training_curve(epochs, losses, title, x_label, y_label, savepath):

    """
    Description:
        Plots the loss per epoch during fitting our Logistic Regressor
    
    Parameters:
        epochs: list holding range of number of epochs trained on
        losses: our training losses
        title: tite of plot
        x_label: label for x axis
        y_label: label for y axis
        savepath: path to save our training curve to

    Returns:
        None
    """

    plt.figure(figsize = (7, 7))

    ax = plt.axes()
    ax.set_facecolor("lavender")

    plt.plot(epochs, losses, c = 'b')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(savepath)

    # return
    return None

def iris_visualize(X, y, feature_names, class_names):
    """
    Decsription:
        Visualize Iris datset from its metadata
    
    Parameters:
        X: features
        y: labels
        feature_names: name of our feaures
        class_names: labels
    
    Returns:
        None
    """

    # feature data
    sepal_lengths, sepal_widths = X[:, 0], X[:, 1]
    petal_lengths, petal_widths = X[:, 2], X[:, 3]
    # targets
    targets = class_names
    # corresponding color (which is also their name)
    colors = y
    # the string title of sepal length and width features
    sepal_length_name, sepal_width_name = feature_names[0], feature_names[1]
    petal_length_name, petal_width_name = feature_names[2], feature_names[3]

    # set figure size
    plt.figure(figsize=(7, 7))

    # scatterplot
    sc = plt.scatter(sepal_lengths, sepal_widths, c = colors)
    plt.title('Iris Dataset Scatterplot - Sepal')
    plt.xlabel(sepal_length_name)
    plt.ylabel(sepal_width_name)
    plt.legend(sc.legend_elements()[0], targets, loc = 'lower right', title = 'Classes')
    plt.grid()
    plt.savefig(f'plots/iris/iris_sepal.png')

    # set figure size
    plt.figure(figsize=(7, 7))

    # scatterplot
    sc = plt.scatter(petal_lengths, petal_widths, c = colors)
    plt.title('Iris Dataset Scatterplot - Petal')
    plt.xlabel(petal_length_name)
    plt.ylabel(petal_width_name)
    plt.legend(sc.legend_elements()[0], targets, loc = 'lower right', title = 'Classes')
    plt.grid()
    plt.savefig(f'plots/iris/iris_petal.png')

    # nothing to return, we just want to save plots
    return None


def visualize_decision_boundaries_iris(X, y, epochs, lr, class_names, resolution, sepal_or_petal):

    """
    Description:
        Plots the decision bonudaries across the entire grid.
        This is to visualize our results for all possible datapoints on a given grid
    
    Parameters:
        X: features
        y: labels
        epochs: number of iterations to train our model on
        lr: learning rate of our model
        class_names: names of our classes
        resolution: resolution of grid for plotting the decision boundary
        sepal_or_petal: are these the sepal features or the petals?
    
    Returns:
        None
    """

    # get the min and max limits for our x-axis and y-axis
    # here the two features chosen for X_train_reduced will compose the x-axis and y-axis respectively
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # we will create a meshgrid based on the x-axis and y-axis range
    # we will take resolution steps between min and max and we are aiming to classify all of these into our given labels
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))

    model = LogisticRegression(epochs = epochs, lr = lr)
    model.fit(X, y)

    # predict the labels for all points on the created meshgrid
    predictions = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(predictions)

    # reshape so we can graph
    Z = Z.reshape(xx.shape)

    # create color maps for our plot, they correspond to the number of labels in iris
    # that is, first element is for label 0, second element is for label 1, and third element is for label 2
    cmap_background = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # plot decision boundaries
    plt.figure(figsize = (7, 7))
    # use color mesh and specify the background
    plt.pcolormesh(xx, yy, Z, cmap = cmap_background)

    # scatter the two chosen features across each other and use y labels as colors based on the color map above
    sc = plt.scatter(X[:, 0], X[:, 1], c = y, cmap = cmap_bold)
    plt.legend(sc.legend_elements()[0], class_names, title = 'Classes', loc = 'lower right')
    plt.title(f'Iris Decision Boundary Prediction - {sepal_or_petal}')
    plt.xlabel(f'{sepal_or_petal.lower()} length (cm)')
    plt.xlim(xx.min(), xx.max())
    plt.ylabel(f'{sepal_or_petal.lower()} width (cm)')
    plt.ylim(yy.min(), yy.max())

    save_path = f'plots/iris/iris_decision_boundaries_{sepal_or_petal.lower()}.png'

    # save plot
    plt.grid()
    plt.savefig(save_path)

    # return
    return None