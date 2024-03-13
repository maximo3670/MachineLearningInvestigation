# Description: This file tests an ensemble of tree classifers and calculated the accuracy
# Created on: 27/11/2023
# Version number: 27/11/2023
    
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Reading in data with the column headers
column_names = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
initial_data = pd.read_csv('../data/breast-cancer.data', names=column_names)
    
# Replace the "?" with NaN values so they can be removed
initial_data.replace("?", pd.NA, inplace=True)
data = initial_data.dropna()

#Create a copyof the data to encode
data_encoded = data.copy()

# Select categorical columns for one-hot encoding
categorical_cols = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'breast', 'breast-quad']

# Perform one-hot encoding using pandas get_dummies
data_frame = pd.get_dummies(data_encoded, columns=categorical_cols)

# Apply label encoding to 'node-caps' and 'irradiat' and 'Class' columns
# Done it this way because they are only two values (1 or 0) when encoded
le = LabelEncoder()
data_frame['node-caps'] = le.fit_transform(data_frame['node-caps'])
data_frame['irradiat'] = le.fit_transform(data_frame['irradiat'])
data_frame['Class'] = le.fit_transform(data_frame['Class'])

# X contains the data except the 'Class' column
X = data_frame.drop('Class', axis=1)
y = data_frame['Class']


# seed for all random functions
seed = 74
# split data as 80% training, 20% testing
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

#Training tree classifiers
DT1 = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 1).fit(X_train, y_train)
DT2 = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 3).fit(X_train, y_train)
DT3 = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 10).fit(X_train, y_train)    

# Getting the predictions of the trees
DT1_predict = DT1.predict(X_test)
DT2_predict = DT2.predict(X_test)
DT3_predict = DT3.predict(X_test)

# Function for majority voting
# It loops through all the predictions 
# If there is a 1 it will add to the votes
# If there isnt a 1 it will do nothing.
# It tracks the overrall votes in the majority[] array
def majority_vote(predictions):
    majority = []
    for i in range(len(predictions[0])):
        votes = {}
        for pred in predictions:
            vote = pred[i]
            if vote in votes:
                votes[vote] += 1
            else:
                votes[vote] = 1
        majority.append(max(votes, key=votes.get))
    return majority

# Getting the majority vote
ensemble_predictions = majority_vote([DT1_predict, DT2_predict, DT3_predict])

# Collecting information for each test case
ensemble_results = []
for i in range(len(X_test)):
    # Assigning a case ID
    case_id = i + 1
    
    #Get the class label
    actual_class_label = y_test.iloc[i]
    
    #Prediction of each model for the current test case
    DT1_pred = DT1_predict[i]
    DT2_pred = DT2_predict[i]
    DT3_pred = DT3_predict[i]
    
    #Predicted label
    ensemble_pred = ensemble_predictions[i]

    # Append information for the current test case to ensemble_results list
    ensemble_results.append([case_id, actual_class_label, DT1_pred, DT2_pred, DT3_pred, ensemble_pred])


# Writing results to CSV
output_df = pd.DataFrame(ensemble_results, columns=['case_id', 'actual_class_label', 'DT1_predict', 'DT2_predict', 'DT3_predict', 'ensemble_predict'])
print(ensemble_results)
output_df.to_csv('..\output\part_2c_ensemble_out.csv', index=False)

# Computing accuracy of the ensemble
ensemble_accuracy = np.sum(ensemble_predictions == y_test.to_numpy()) / len(y_test)
print("Ensemble Accuracy:", ensemble_accuracy)
# Append accuracy to the CSV file
with open('..\output\part_2c_ensemble_out.csv', 'a') as f:
    f.write(f"Ensemble Accuracy: {ensemble_accuracy}\n")



