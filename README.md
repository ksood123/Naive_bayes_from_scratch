# Naive_bayes_from_scratch
The code has been implemented in Python language.
1) The dataset has 4 categorical columns: 'Race', 'Gender' , 'Age' and 'Diabetic'. The fist three columns are predictor variables and the last column 'Diabetic' is the target
variable with '0' and '1' indicating negative and positive results for person having diabetes. The code for generating this data has been provided in folder named 'Generating Data'.
2) There are two data sets-'naive_dataset_train' and 'naive_dataset_test' and both have been generated using te same source code.
3) We load both datasets and output the training and testing accuracy. The specificity and sensitivity has also been calculated using
'classification_report' from 'sklearn.metrics'.
