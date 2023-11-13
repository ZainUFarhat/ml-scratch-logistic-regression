# datasets
from datasets import *

# utils
from utils import *

# set numpy random seed for reproducibility
np.random.seed(42)

def main():

    """
    Description:
        Trains, tests, and provides plots for our Logistic Regression model

    Parameters:
        None
    
    Returns:
        None    
    """

    # dataset hyperparameters
    test_size = 0.2
    random_state = 42
    dataset_name = 'Simple Xy'
    n_samples = 1000
    n_features = 6
    n_classes = 2

    # create an instance of Datasets class
    datasets = Datasets(test_size = test_size, random_state = random_state)

    # load Simple Xy dataset
    X, y, X_train, X_test, y_train, y_test = datasets.simple_Xy(n_samples = n_samples, n_features = n_features, n_classes = n_classes)
    
    print(f'Loading {dataset_name} Dataset...')
    print(f'\n{dataset_name} contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nLogistic Regression\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    # Logistic Regression Hyperparameters
    epochs = 1000
    lr = 0.1

    # logistic regression
    lreg = LogisticRegression(epochs = epochs, lr = lr)

    # fit and fetch our training losses
    cross_entropy_losses = lreg.fit(X_train, y_train)

    print('Done Training!') 
    print('---------------------------------------------------Testing----------------------------------------------------')
    # predictions
    y_pred = lreg.predict(X_test)

    # get cross entropy test loss
    cross_entropy_test_final = cross_entropy_loss(y = y_test, y_pred = y_pred)
    print(f'{dataset_name} Cross Entropy Test Loss =', round(cross_entropy_test_final, 4))

    # get accuracy
    acc = accuracy_fn(y_true = y_test, y_pred = y_pred)
    print('{0} Test Accuracy = {1}%'.format(dataset_name, acc))
    print('---------------------------------------------------Plotting---------------------------------------------------')
    print('PLotting Simple Xy...')
    features = [0, 1]
    title_scatter = f'{dataset_name} Scatterplot - feature 0 vs. feature 1'
    save_path_scatter = 'plots/xy/xy_scatter.png'
    scatter_plot(X = X, y = y, features = features, title = title_scatter, x_label = 'feature 0', y_label = 'feature 1', 
                                class_names = ['class 0', 'class 1'], savepath = save_path_scatter)

    # training curve
    title_loss = f'{dataset_name} - Cross Entropy Training Loss'
    save_path_loss = 'plots/xy/xy_loss.png'
    plot_training_curve(epochs = list(range(1, epochs+1)), losses = cross_entropy_losses, title = title_loss,
                                                        x_label = 'epochs', y_label = 'cross entropy', savepath = save_path_loss)  

    # decision boundary
    resolution = 0.001
    title_boundary = f'{dataset_name} Decision Boundary - feature 0 vs. feature 1'  
    save_path_boundary = 'plots/xy/xy_decision_boundary.png'
    visualize_decision_boundary(X = X[:, features], y = y, epochs = epochs, lr = lr, features = features, title = title_boundary, 
                                x_label = 'feature 0', y_label = 'feature 1', class_names = ['class 0', 'class 1'], 
                                resolution = resolution, savepath = save_path_boundary)
    print('Please refer to plots/xy directory to view all plots.')
    print('--------------------------------------------------------------------------------------------------------------')



    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    dataset_name = 'Breast Cancer'

    # load the breast cancer dataset
    X, y, feature_names, class_names, X_train, X_test, y_train, y_test = datasets.load_breast_cancer()

    print(f'Loading {dataset_name} Dataset...')
    print(f'\nThe Features of {dataset_name} Dataset are:', ', '.join(feature_names))
    print(f'The Labels of the {dataset_name} Dataset are:', ', '.join(class_names))
    print(f'\n{dataset_name} contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nLogistic Regression\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    # Logistic Regression Hyperparameters
    epochs = 1000
    lr = 0.02

    # logistic regression
    lreg = LogisticRegression(epochs = epochs, lr = lr)

    # fit and fetch our training losses
    cross_entropy_losses = lreg.fit(X_train, y_train)

    print('Done Training!') 
    print('---------------------------------------------------Testing----------------------------------------------------')
    # predictions
    y_pred = lreg.predict(X_test)

    # get cross entropy test loss
    cross_entropy_test_final = cross_entropy_loss(y = y_test, y_pred = y_pred)
    print(f'{dataset_name} Cross Entropy Test Loss =', cross_entropy_test_final)

    # get accuracy
    acc = accuracy_fn(y_true = y_test, y_pred = y_pred)
    print('{0} Test Accuracy = {1}%'.format(dataset_name, acc))
    print('---------------------------------------------------Plotting---------------------------------------------------')
    # scatter plot of original data
    title_scatter = f'{dataset_name} - {feature_names[0]} vs. {feature_names[1]}'
    save_path_scatter = 'plots/bc/bc_scatter.png'
    scatter_plot(X = X, y = y, features = [0,1], title = title_scatter, x_label = feature_names[0], y_label = feature_names[1], 
                                class_names = class_names, savepath = save_path_scatter)

    # training curve
    title_loss = f'{dataset_name} - Cross Entropy Training Loss'
    save_path_loss = 'plots/bc/bc_loss.png'
    plot_training_curve(epochs = list(range(1, epochs+1)), losses = cross_entropy_losses, title = title_loss,
                                                        x_label = 'epochs', y_label = 'cross entropy', savepath = save_path_loss)  

    # decision boundary
    resolution = 0.001
    title_boundary = f'{dataset_name} Decision Boundary - {feature_names[0]} vs. {feature_names[1]}'  
    save_path_boundary = 'plots/bc/bc_decision_boundary.png'
    visualize_decision_boundary(X = X[:, [0,1]], y = y, epochs = 3000, lr = 0.03, features = [0,1], title = title_boundary, 
                                x_label = feature_names[0], y_label = feature_names[1], class_names = class_names, 
                                resolution = resolution, savepath = save_path_boundary)
    print('Please refer to plots/bc directory to view all plots.')
    print('--------------------------------------------------------------------------------------------------------------')  

######################################################################################################################################
    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    dataset_name = 'Diabetes'

    # load the breast cancer dataset
    X, y, feature_names, class_names, X_train, X_test, y_train, y_test = datasets.load_diabetes()

    print(f'Loading {dataset_name} Dataset...')
    print(f'\nThe Features of {dataset_name} Dataset are:', ', '.join(feature_names))
    print(f'The Labels of the {dataset_name} Dataset are:', ', '.join(class_names))
    print(f'\n{dataset_name} contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nLogistic Regression\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    # Logistic Regression Hyperparameters
    epochs = 100
    lr = 0.1

    # logistic regression
    lreg = LogisticRegression(epochs = epochs, lr = lr)

    # fit and fetch our training losses
    cross_entropy_losses = lreg.fit(X_train, y_train)

    print('Done Training!') 
    print('---------------------------------------------------Testing----------------------------------------------------')
    # predictions
    y_pred = lreg.predict(X_test)

    # get cross entropy test loss
    cross_entropy_test_final = cross_entropy_loss(y = y_test, y_pred = y_pred)
    print(f'{dataset_name} Cross Entropy Test Loss =', round(cross_entropy_test_final, 4))

    # get accuracy
    acc = accuracy_fn(y_true = y_test, y_pred = y_pred)
    print('{0} Test Accuracy = {1}%'.format(dataset_name, acc))
    print('---------------------------------------------------Plotting---------------------------------------------------')
    # scatter plot of original data
    feature_1, feature_2 = 'ldl', 'hdl'
    title_scatter = f'{dataset_name} - {feature_1} vs. {feature_2}'
    save_path_scatter = 'plots/db/db_scatter.png'
    scatter_plot(X = X, y = y, features = [5,6], title = title_scatter, x_label = feature_1, y_label = feature_2, 
                                class_names = class_names, savepath = save_path_scatter)

    # training curve
    title_loss = f'{dataset_name} - Cross Entropy Training Loss'
    save_path_loss = 'plots/db/db_loss.png'
    plot_training_curve(epochs = list(range(1, epochs+1)), losses = cross_entropy_losses, title = title_loss,
                                                        x_label = 'epochs', y_label = 'cross entropy', savepath = save_path_loss)  

    # decision boundary
    resolution = 0.001
    title_boundary = f'{dataset_name} Decision Boundary - {feature_1} vs. {feature_2}'  
    save_path_boundary = 'plots/db/db_decision_boundary.png'
    visualize_decision_boundary(X = X[:, [5,6]], y = y, epochs = 2000, lr = 0.1, features = [0,1], title = title_boundary, 
                                x_label = feature_names[0], y_label = feature_names[1], class_names = class_names, 
                                resolution = resolution, savepath = save_path_boundary)
    print('Please refer to plots/db directory to view all plots.')
    print('--------------------------------------------------------------------------------------------------------------')


    # return
    return None

if __name__ == '__main__':

    # run everything
    main()