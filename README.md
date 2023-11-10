# ml-scratch-logistic-regression
Logistic Regression Algorithm

## **Description**
The following is my from scratch implementation of the Logistic Regression algorithm.

### **Dataset**

For datasets I used three datasets: \
\
    &emsp;1. Breast Cancer Dataset \
    &emsp;2. Diabetes Dataset \
\
For each dataset I load it and scale the features variables to the range [0, 1]. This is to avoid the magnitude differences that can arise during the fitting process.

### **Walkthrough**

**1.** Need the following packages installed: sklearn, numpy, and matplotlib.

**2.** Once you made sure all these libraries are installed, evrything is simple, just head to main.py and execute it.

**3.** Since code is modular, main.py can easily: \
\
    &emsp;**i.** Load the two datasets \
    &emsp;**ii.** Split data into train and test sets \
    &emsp;**iii.** Build a logistic regressor \
    &emsp;**iv.** Fit the logistic regressor \
    &emsp;**v.** Predict on the test set \
    &emsp;**vi.** Plot the scatter plot, loss curve, and decision boundary.

**4.** In main.py I specify a set of hyperparameters, these can be picked by the user. The main ones worth noting are the number of epochs and learning rate. These hyperparameters were chosen through trail & error experimentation on each dataset.

### **Results**

For each dataset I will list the number of epochs, learning rate, test Cross Entropy loss score, and Accuracy score.
In addition I offer three visualization plots for a better assessment.
Because of the high dimensionality of each dataset I chose two features that I assumed to have the most correspondance and used them for visualizing my scatter plots and decision boundaries.

**1.** Breast Cancer Dataset:

- Hyperparameters:
     - Number of epochs = 1000
     - Learning rate = 0.02
 
- Numerical Result:
     - Cross Entropy Test Loss = 7.094325127354754e-14
     - Accuracy: 94.74%

- See visualizations below:

For this I chose the mean radius and mean texture features.

![alt text](https://github.com/ZainUFarhat/ml-scratch-logistic-regression/blob/main/plots/bc/bc_scatter.png?raw=true)

![alt text](https://github.com/ZainUFarhat/ml-scratch-logistic-regression/blob/main/plots/bc/bc_decision_boundary.png?raw=true)

![alt text](https://github.com/ZainUFarhat/ml-scratch-logistic-regression/blob/main/plots/bc/bc_loss.png?raw=true)

**2.** Diabetes Dataset:

- Hyperparameters:
     - Number of epochs = 100
     - Learning rate = 0.1
 
- Numerical Result:
     - Cross Entropy Test Loss = 241.7714
     - Accuracy: 77.53%

- See visualizations below:

For this I chose the ldl and hdl features.

![alt text](https://github.com/ZainUFarhat/ml-scratch-logistic-regression/blob/main/plots/db/db_scatter.png?raw=true)

![alt text](https://github.com/ZainUFarhat/ml-scratch-logistic-regression/blob/main/plots/db/db_decision_boundary.png?raw=true)

![alt text](https://github.com/ZainUFarhat/ml-scratch-logistic-regression/blob/main/plots/db/db_loss.png?raw=true)