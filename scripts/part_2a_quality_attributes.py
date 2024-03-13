# Description: Script to calculate the information gain, gini impurity and chi2
#           values given a contigency table.
# Created on: 21/11/2023
# Version number: 21/11/2023

import numpy as np

# calculate_entropy() function
# Extra function to calcualate entropy
# Uses the entropy equation to calculate a value and returns this value
# Saves writing same equation multiple times.
def calculate_entropy(value):
    if value != 0:
        #Equation to calculate entropy of a single value
        return -value * np.log2(value)
    else:
        #If the value is 0 the entropy is always 0
        return 0

# get_information_gain() function
# Function which returns the information gain of a contingency table
def get_information_gain(table):
    #Calcualates total of the table
    total = np.sum(table)
    
    #Calculating the probability of it being positive or negative
    #First calcualates the total pos and neg then divides by the total
    positive_probability = np.sum(table[:, 0]) / total
    negative_probability = np.sum(table[:, 1]) / total
                      
    #Hx is the total entropy.
    Hx = (calculate_entropy(positive_probability) + calculate_entropy(negative_probability))
    
    #probability of true and false for attribute 1 (Yes)
    #sums the total true and false and divides by the total number.
    probTrueAttribute1 = (table[0][0]) / (np.sum(table[0]))
    probFalseAttribute1 = (table[0][1]) / (np.sum(table[0]))
    
    #Ha1 is the entropy value for attribute 1
    Ha1 = (calculate_entropy(probTrueAttribute1) + calculate_entropy(probFalseAttribute1))
    
    #probability of true and false for attribute 2 (No)
    #sums the total true and false and divides by the total number.
    probTrueAttribute2 = (table[1][0]) / (np.sum(table[1]))
    probFalseAttribute2 = (table[1][1]) / (np.sum(table[1]))
    
    #Ha2 is the entropy for attribute 2
    Ha2 = (calculate_entropy(probTrueAttribute2) + calculate_entropy(probFalseAttribute2))
    
    #Final calculation for information gain
    informationGain = Hx - ( ((np.sum(table[0]) / total) * Ha1) + (np.sum(table[1]) / total) * Ha2)
    
    return informationGain

# calculate_gini() function
#Function to calculate the gini of each attribute
# Uses the gini equation
def calculate_gini(probability_A, probability_B):
    #Gini equation
    return 1 - ( ((probability_A)**2) + ((probability_B)**2))

#Function which returns the gini impurity of a contingency table
def get_gini_impurity(table):
    #Calcualates total of the table
    total = np.sum(table)
    
    #Calculating the probability of it being positive or negative
    #First calcualates the total pos and neg then divides by the total
    positive_probability = np.sum(table[:, 0]) / total
    negative_probability = np.sum(table[:, 1]) / total
    
    #Ix is the gini impurity of the root node
    Ix = calculate_gini(positive_probability, negative_probability)
    
    #probability of true and false for attribute 1 (Yes)
    #sums the total true and false and divides by the total number.
    probTrueAttribute1 = (table[0][0]) / (np.sum(table[0]))
    probFalseAttribute1 = (table[0][1]) / (np.sum(table[0]))
    
    #Ia1 is the gini impurity of attribute 1
    Ia1 = calculate_gini(probTrueAttribute1, probFalseAttribute1)
    
    #probability of true and false for attribute 2 (No)
    #sums the total true and false and divides by the total number.
    probTrueAttribute2 = (table[1][0]) / (np.sum(table[1]))
    probFalseAttribute2 = (table[1][1]) / (np.sum(table[1]))
    
    #Ia2 is the gini impurity of attribute 2
    Ia2 = calculate_gini(probTrueAttribute2, probFalseAttribute2)
    
    gini = Ix - (((np.sum(table[0]) / total) * Ia1) + (np.sum(table[1]) / total) * Ia2)
    
    
    return gini

#Function which returns the chi squared of a contingency table
def get_chi2(table):
    #Calcualates total of the table
    total = np.sum(table)
    
    #Calculating the probability of it being positive or negative
    #First calcualates the total pos and neg then divides by the total
    positive_probability = np.sum(table[:, 0]) / total
    negative_probability = np.sum(table[:, 1]) / total
    
    # Calculating the expected values
    posAttr1 = np.sum(table[0]) * positive_probability
    negAttr1 = np.sum(table[0]) * negative_probability
    posAttr2 = np.sum(table[1]) * positive_probability
    negAttr2 = np.sum(table[1]) * negative_probability
    
    #Calcualting the chi2 value
    posAttr1_calc = (((table[0][0] - posAttr1)**2) / posAttr1)
    negAttr1_calc = (((table[0][1] - negAttr1)**2) / negAttr1)
    posAttr2_calc = (((table[1][0] - posAttr2)**2) / posAttr2)
    negAttr2_calc = (((table[1][1] - negAttr2)**2) / negAttr2)
    
    chi2 = posAttr1_calc + negAttr1_calc + posAttr2_calc + negAttr2_calc
    
    return chi2



# Define the contingency tables 
# where the rows are "yes" & "no"
# Columns are "positive" & "negative"

headache = np.array([
            [3, 1],
            [2, 4]])

spots = np.array([
            [4, 2],
            [1, 3]])

stiffNeck = np.array([
            [4, 1],
            [1, 4]])


#Printing attribute values to console

print("Attribute: Headache\n", 
      "Contingency Table: \n", headache,
      "\nInformation Gain: ", get_information_gain(headache),
      "\nGini Impurity: ", get_gini_impurity(headache),
      "\nChi-Squared: ", get_chi2(headache))
print("\n")
print("Attribute: Spots\n", 
      "Contingency Table: \n", spots,
      "\nInformation Gain: ", get_information_gain(spots),
      "\nGini Impurity: ", get_gini_impurity(spots),
      "\nChi-Squared: ", get_chi2(spots))
print("\n")
print("Attribute: Stiff-Neck\n", 
      "Contingency Table: \n", stiffNeck,
      "\nInformation Gain: ", get_information_gain(stiffNeck),
      "\nGini Impurity: ", get_gini_impurity(stiffNeck),
      "\nChi-Squared: ", get_chi2(stiffNeck))

#saving attributes to .txt file
output_file = "..\output\part_2a_quality_attributes_out.txt"

with open(output_file, "w") as file:
    file.write("Attribute: Headache\n")
    file.write("Contingency Table:\n")
    file.write(str(headache) + "\n")
    file.write("Information Gain: " + str(get_information_gain(headache)) + "\n")
    file.write("Gini Impurity: " + str(get_gini_impurity(headache)) + "\n")
    file.write("Chi-Squared: " + str(get_chi2(headache)) + "\n\n")
    
    file.write("Attribute: Spots\n")
    file.write("Contingency Table:\n")
    file.write(str(spots) + "\n")
    file.write("Information Gain: " + str(get_information_gain(spots)) + "\n")
    file.write("Gini Impurity: " + str(get_gini_impurity(spots)) + "\n")
    file.write("Chi-Squared: " + str(get_chi2(spots)) + "\n\n")
    
    file.write("Attribute: Stiff-Neck\n")
    file.write("Contingency Table:\n")
    file.write(str(stiffNeck) + "\n")
    file.write("Information Gain: " + str(get_information_gain(stiffNeck)) + "\n")
    file.write("Gini Impurity: " + str(get_gini_impurity(stiffNeck)) + "\n")
    file.write("Chi-Squared: " + str(get_chi2(stiffNeck)) + "\n\n")


