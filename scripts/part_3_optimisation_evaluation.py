# Description: Optimisation and evaluation of different classifiers
# Uses 10 K fold splits to find the optimal parameters for the classifiers
# Then uses the parameters found from this to make a classifier using 85% of the data set
# Created on: 04/11/2023
# Version number: 06/11/2023, commented the code
    
#Necessary Imports
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics

#Load data into X and y
X, y = load_digits(return_X_y=True)

#Random seed set to last 2 digits of my student number
seed = 74

#Setting how many K fold splits to be done
kfold_splits = 10
kf = KFold(n_splits=kfold_splits, shuffle=False)

################################################################
# This section is for a random forest classfier
# It loops through different hyperparameters to find the most optimal solution
################################################################

#Define hyperparameters to search through
#More values can be chosen to test if necessary
#Having more values greatly increases computing power and compile time
n_estimators_values = [1, 25, 100, 200]
max_depth_values = [1, 5, 10, 20, 50, 100, None]

#Initialise some variables for storage of data
rf_best_accuracy = 0.0
rf_best_params = {}
rf_testing_accuracy_scores = {}

#Looping through all the different hyperparameter variations
#Makes sure all possibilities are checked
for n_estimators in n_estimators_values:
    for max_depth in max_depth_values:
                mean_accuracy = 0.0

                #Loops through all the k Fold splits (10 in this case)
                for train_index, test_index in kf.split(X):
                    X_train_fold, X_val_fold  = X[train_index], X[test_index]
                    y_train_fold, y_val_fold  = y[train_index], y[test_index]
                    
                    # Instantiate RandomForestClassifier with current parameter values
                    rf = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=seed
                    )

                    # Train the classifier on the training fold
                    rf.fit(X_train_fold, y_train_fold)

                    # Predict on the validation fold
                    y_pred = rf.predict(X_val_fold)

                    # Calculate accuracy for this fold
                    fold_accuracy = accuracy_score(y_val_fold, y_pred)
                    mean_accuracy += fold_accuracy

                # Calculate mean accuracy across all folds for these parameters
                mean_accuracy /= kfold_splits
                
                #Stores the mean accuracy with the parameters as a key
                rf_testing_accuracy_scores[(n_estimators, max_depth)] = mean_accuracy
                
                #Finds the best accuracy and saves the best parameter combinations
                # Check if current parameter combination gives better accuracy
                if mean_accuracy > rf_best_accuracy:
                    rf_best_accuracy = mean_accuracy
                    rf_best_params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                    }

print("Best Parameters:", rf_best_params)
print("Best Accuracy:", rf_best_accuracy)

# Convert accuracy scores to a pandas DataFrame for table
rf_accuracy_df = pd.DataFrame(list(rf_testing_accuracy_scores.items()), columns=['Parameters', 'Testing Accuracy'])
rf_accuracy_df[['n_estimators', 'max_depth']] = pd.DataFrame(rf_accuracy_df['Parameters'].tolist(), index=rf_accuracy_df.index)
rf_accuracy_df.drop('Parameters', axis=1, inplace=True)

# Print table of accuracy scores
print(rf_accuracy_df)

# Create a bar plot for visual comparison
plt.figure(figsize=(10, 6))
rf_accuracy_df['ParameterCombination'] = rf_accuracy_df[['n_estimators', 'max_depth']].astype(str).agg('-'.join, axis=1)
rf_accuracy_df.plot(x='ParameterCombination', y='Testing Accuracy', kind='bar', legend=False)
plt.xlabel('Parameter Combinations')
plt.ylabel('Testing Accuracy')
plt.title('Testing Accuracy of Parameter Combinations')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

################################################################
# This section is for a K Nearest Neighbors classifier
# It loops through different hyperparameters to find the most optimal solution
################################################################

#Define hyperparameters to search through
n_neighbors_values = [1, 3, 5, 10, 20, 50, 100]
weights_list = ['uniform', 'distance']

#Initialise some variables for storage of data
kn_best_accuracy = 0.0
kn_best_params = {}
kn_testing_accuracy_scores = {}

#Looping through all the different hyperparameter variations
#Makes sure all possibilities are checked
for n_neighbors in n_neighbors_values:
    for weight_option in weights_list:
        mean_accuracy = 0.0

        #Loops through all the k Fold splits (10 in this case)
        for train_index, test_index in kf.split(X):
            X_train_fold, X_val_fold = X[train_index], X[test_index]
            y_train_fold, y_val_fold = y[train_index], y[test_index]

            # Instantiate K Neighbors Classifier with current parameter values
            kn = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weight_option
            )

            # Train the classifier on the training fold
            kn.fit(X_train_fold, y_train_fold)

            # Predict on the validation fold
            y_pred = kn.predict(X_val_fold)

            # Calculate accuracy for this fold
            fold_accuracy = accuracy_score(y_val_fold, y_pred)
            mean_accuracy += fold_accuracy

        # Calculate mean accuracy across all folds for these parameters
        mean_accuracy /= kfold_splits

        #Stores the mean accuracy with the parameters as a key
        kn_testing_accuracy_scores[(n_neighbors, weight_option)] = mean_accuracy

        #Finds the best accuracy and saves the best parameter combinations
        # Check if current parameter combination gives better accuracy
        if mean_accuracy > kn_best_accuracy:
            kn_best_accuracy = mean_accuracy
            kn_best_params = {
                'n_neighbors': n_neighbors,
                'weights': weight_option,
            }

print("Best Parameters:", kn_best_params)
print("Best Accuracy:", kn_best_accuracy)

# Convert accuracy scores to a pandas DataFrame for table
kn_accuracy_df = pd.DataFrame(list(kn_testing_accuracy_scores.items()), columns=['Parameters', 'Testing Accuracy'])
kn_accuracy_df[['n_neighbors', 'weights']] = pd.DataFrame(kn_accuracy_df['Parameters'].tolist(), index=kn_accuracy_df.index)
kn_accuracy_df.drop('Parameters', axis=1, inplace=True)

# Print table of accuracy scores
print(kn_accuracy_df)

# Create a bar plot for visual comparison
plt.figure(figsize=(10, 6))
kn_accuracy_df['ParameterCombination'] = kn_accuracy_df[['n_neighbors', 'weights']].astype(str).agg('-'.join, axis=1)
kn_accuracy_df.plot(x='ParameterCombination', y='Testing Accuracy', kind='bar', legend=False)
plt.xlabel('Parameter Combinations (n_neighbors-weights)')
plt.ylabel('Testing Accuracy')
plt.title('Testing Accuracy of Parameter Combinations (K Neighbors Classifier)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


################################################################
# This section is for a bagging classifier
# It loops through different hyperparameters to find the most optimal solution
################################################################

n_estimators_values = [1, 3, 5, 10, 20, 50, 100, 200]

bc_best_accuracy = 0.0
bc_best_params = {}
bc_testing_accuracy_scores = {}

for n_estimators in n_estimators_values:
    mean_accuracy = 0.0

    for train_index, test_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[test_index]
        y_train_fold, y_val_fold = y[train_index], y[test_index]

        # Instantiate K Neighbors Classifier with current parameter values
        bc = BaggingClassifier(n_estimators=n_estimators, random_state=seed)

        # Train the classifier on the training fold
        bc.fit(X_train_fold, y_train_fold)

        # Predict on the validation fold
        y_pred = bc.predict(X_val_fold)

        # Calculate accuracy for this fold
        fold_accuracy = accuracy_score(y_val_fold, y_pred)
        mean_accuracy += fold_accuracy

    # Calculate mean accuracy across all folds for these parameters
    mean_accuracy /= kfold_splits

    bc_testing_accuracy_scores[(n_estimators)] = mean_accuracy

    # Check if current parameter combination gives better accuracy
    if mean_accuracy > bc_best_accuracy:
        bc_best_accuracy = mean_accuracy
        bc_best_params = {
            'n_estimators': n_estimators,
        }
    
print("Best Parameters:", bc_best_params)
print("Best Accuracy:", bc_best_accuracy)

# Convert accuracy scores to a pandas DataFrame for table
bc_accuracy_df = pd.DataFrame(list(bc_testing_accuracy_scores.items()), columns=['Parameters', 'Testing Accuracy'])
bc_accuracy_df['n_estimators'] = bc_accuracy_df['Parameters']
bc_accuracy_df.drop('Parameters', axis=1, inplace=True)

# Print table of accuracy scores
print(bc_accuracy_df)

# Create a bar plot for visual comparison
plt.figure(figsize=(10, 6))
bc_accuracy_df.plot(x='n_estimators', y='Testing Accuracy', kind='bar', legend=False)
plt.xlabel('Number of Estimators (n_estimators)')
plt.ylabel('Testing Accuracy')
plt.title('Testing Accuracy for Different Number of Estimators (Bagging Classifier)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

################################################################
# This section is creating each classifier using the best parameters found.
# It then works out different scores to judge the accuracy
################################################################

# split data as 85% training, 15% testing
test_size = 0.15
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

################################################################
# Random Forest #
################################################################

# Creating a random forest classifier with the best parameters found previously
rf_classifier = RandomForestClassifier(n_estimators=rf_best_params['n_estimators'] ,max_depth=rf_best_params['max_depth'] ,random_state=seed)

#Training the classifier with 85% of the data
rf_classifier.fit(X_train, y_train)

#Predict the output on 15% of the data
rf_y_prediction = rf_classifier.predict(X_test)

#Calcualate the accruacy of the classifier
rf_accuracy = accuracy_score(y_test, rf_y_prediction)

print("RF Accuracy Score: ", rf_accuracy)

# Creating a confusion matrix using sklearn metrics
rf_conf_matrix = metrics.confusion_matrix(y_test, rf_y_prediction)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = rf_conf_matrix)

cm_display.plot()
plt.show()

################################################################
# K Neighbors #
################################################################

#Creating a KNeighbors Classifier with the best parameters
kn_classifier = KNeighborsClassifier(n_neighbors=kn_best_params['n_neighbors'], weights=kn_best_params['weights'])

#Training it using 85% of the data
kn_classifier.fit(X_train, y_train)

#Predicting using the test set (15% of data)
kn_y_prediction = kn_classifier.predict(X_test)

#Calculating the accuracy
kn_accuracy = accuracy_score(y_test, kn_y_prediction)

print("KN Accuracy Score: ", kn_accuracy)

# Creating a confusion matrix using sklearn metrics
kn_conf_matrix = metrics.confusion_matrix(y_test, kn_y_prediction)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = kn_conf_matrix)

cm_display.plot()
plt.show()

################################################################
# Bagging Classifier #
################################################################

#Creating a Bagging Classifier with the best parameters
bc_classifier = BaggingClassifier(n_estimators=bc_best_params['n_estimators'], random_state=seed)

#Training it using 85% of the data
bc_classifier.fit(X_train, y_train)

#Predicting using the test set
bc_y_prediction = bc_classifier.predict(X_test)

#Calculating the accuracy
bc_accuracy = accuracy_score(y_test, bc_y_prediction)

print("Bagging Classifier Accuracy Score: ", bc_accuracy)

# Creating a confusion matrix using sklearn metrics
bc_conf_matrix = metrics.confusion_matrix(y_test, bc_y_prediction)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = bc_conf_matrix)

cm_display.plot()
plt.show()

################################################################
# Graph for comparison #
################################################################

classifier_names = ['Random Forest', 'K Neighbors', 'Bagging Classifier']
accuracy_scores = [rf_accuracy, kn_accuracy, bc_accuracy]

# Plotting the bar graph
plt.figure(figsize=(8, 6))
plt.bar(classifier_names, accuracy_scores, color=['blue', 'green', 'orange'])
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Classifiers')
plt.ylim(0.8, 1.0)  # Set y-axis limit between 0.8 and 1.0 for accuracy values
plt.show()
        
