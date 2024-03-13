# Description: Creating tree classifiers based on UCI breast cancer data set
# Created on: 28/11/2023
# Version number: 29/11/2023, Finished off encoding the data and made tree classifiers
    
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

#Reading in the data
#Setting the column names
column_names = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
initial_data = pd.read_csv('../data/breast-cancer.data', names=column_names)
    
# Number of cases before the removal
total_cases = len(initial_data)

# Replace the "?" with NaN values so they can be removed
# Using pandas dropna() for more effient removal.
initial_data.replace("?", pd.NA, inplace=True)
data = initial_data.dropna()

# Number of cases after removal
total_cases_after = len(data)
    
# Number of attributes
num_attributes = len(data.columns)
    
# Calculate the number of removed cases
cases_removed = total_cases - total_cases_after
    
#printing values to console
print("Dataset:  UCI Breast Cancer dataset")
print("Number of cases removed: ", cases_removed)
print("Cases: ", total_cases_after)
print("Attributes: ", num_attributes)
    
#saving values to .txt file
output_file = "..\output\part_2b_decision_trees_data_preprocessing.txt"
    
with open(output_file, "w") as file:
     file.write("Dataset: UCI Breast Cancer dataset\n")
     file.write("Number of cases removed: " + str(cases_removed) + "\n")
     file.write("Cases: " + str(total_cases_after) + "\n")
     file.write("Attributes: " + str(num_attributes) + "\n")
     
#Create a copy of the data to code
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

with open(output_file, "a") as file:
    file.write("\noriginal column name: number of the binary columns generated \nClass: 1 \nage: 9 \nmenopause: 3 \ntumor-size: 12 \ninv-nodes: 13 \nnode-caps: 1 \ndeg-malig: 1 \nbreast: 2 \nbreast-quad: 5 \nirradiat: 1")

# X contains the data except the 'Class' column
X = data_frame.drop('Class', axis=1)
y = data_frame['Class']


# seed for all random functions
seed = 74
# split data as 80% training, 20% testing
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    
criterion = "entropy"
max_depth = 2

#Training tree classifier    
tr = tree.DecisionTreeClassifier(criterion=criterion, max_depth = max_depth).fit(X_train, y_train)    
    
# Accuracy test on tree classifier
y_test_tr_Prediction = tr.predict(X_test)
tr_acc_test = accuracy_score(y_test, y_test_tr_Prediction)
tr_bal_acc_test = balanced_accuracy_score(y_test, y_test_tr_Prediction)

#Printing out accuracy
print("Accuracy score: ", tr_acc_test)
print("Balanced accuracy score: ", tr_bal_acc_test)

# Define the output file for classifier configuration and performance scores
output_file_classifier = "../output/part_2b_decision_trees_out.txt"

# writing the classifier configuration and performance scores to the output file
with open(output_file_classifier, "w") as file:
    file.write("Decision Tree classifier\n")
    file.write("criterion: " + criterion + "\n")  
    file.write(f"max depth: {max_depth}\n")  
    file.write(f"accuracy: {tr_acc_test:.4f}\n")
    file.write(f"balanced accuracy: {tr_bal_acc_test:.4f}\n")

#getting feature names from X
feature_names_list = list(X.columns)

# Output image of the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(tr, filled=True, feature_names=feature_names_list, class_names=["No Recurrence", "Recurrence"])
plt.savefig("../output/part_2b_decision_trees_tree_out.png")

# Define the output file for classifier configurations and performance scores
output_file_depths = "../output/part_2b_decision_trees_max_depths_out.txt"

# Track the maximum accuracy and balanced accuracy along with their respective model configurations
max_accuracy = 0
max_accuracy_config = None
max_balanced_accuracy = 0
max_balanced_accuracy_config = None

# Lists to store accuracy and balanced accuracy scores for plotting
accuracy_scores = []
balanced_accuracy_scores = []

# Write to the output file
with open(output_file_depths, "w") as file:
    file.write("Model id, max depth, accuracy, balanced accuracy\n")

    # Loop through different max_depth values from 1 to 10
    for model_id, depth in enumerate(range(1, 11), start=1):
        # Create and train Decision Tree classifier with varying max_depth
        tr = tree.DecisionTreeClassifier(criterion=criterion, max_depth=depth).fit(X_train, y_train)

        # Predict using the trained classifier
        y_test_tr_Prediction = tr.predict(X_test)

        # Calculate accuracy scores
        tr_acc_test = accuracy_score(y_test, y_test_tr_Prediction)
        tr_bal_acc_test = balanced_accuracy_score(y_test, y_test_tr_Prediction)

        # Format and write model info to the file
        model_info = f"Model id {model_id}, max depth: {depth}, accuracy: {tr_acc_test:.4f}, balanced accuracy: {tr_bal_acc_test:.4f}\n"
        file.write(model_info)

        # Append scores to lists for plotting
        accuracy_scores.append(tr_acc_test)
        balanced_accuracy_scores.append(tr_bal_acc_test)

        # Check for the highest accuracy and balanced accuracy configurations
        if tr_acc_test > max_accuracy:
            max_accuracy = tr_acc_test
            max_accuracy_config = (model_id, depth)

        if tr_bal_acc_test > max_balanced_accuracy:
            max_balanced_accuracy = tr_bal_acc_test
            max_balanced_accuracy_config = (model_id, depth)

# Plotting the accuracy and balanced accuracy scores
plt.figure(figsize=(10, 6))
depths = range(1, 11)
plt.plot(depths, accuracy_scores, marker='o', label='Accuracy')
plt.plot(depths, balanced_accuracy_scores, marker='o', label='Balanced Accuracy')

# Mark the configurations with the highest accuracy and balanced accuracy
plt.scatter(max_accuracy_config[1], max_accuracy, color='red', s=100, label=f'Highest Accuracy: {max_accuracy:.4f}', zorder=5)
plt.scatter(max_balanced_accuracy_config[1], max_balanced_accuracy, color='green', s=100, label=f'Highest Balanced Accuracy: {max_balanced_accuracy:.4f}', zorder=5)

plt.xlabel('Max Depth')
plt.ylabel('Score')
plt.title('Accuracy and Balanced Accuracy Scores for Different Max Depths')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("../output/part_2b_decision_trees_max_depths_plot.png")
plt.show()
