import pandas as pd
import random
import numpy as np
from sklearn.metrics import classification_report

# This function fits the model by learning about the probabilities of various attributes and their corresponding values
def model_fit(df):
    # We create a dictionary to store the probability of '0' or '1' for each attribute
    probability_dict={}
    col_list=df.columns ; target=col_list[-1] ; col_list=col_list[:-1]
    # Calculating probability value across various columns
    for i in col_list:
        freq_counts=df[i].value_counts()
        freq_dict=freq_counts.to_dict()
        # Getting column names
        l=list(freq_dict.keys())
        # Calculating probability values for various attribute values in a column
        for j in l:
            # Calculating probability of an attribute for a 'No'
            n_c=len(df[(df[i]==j) & (df[target]=='Negative')])
            n=freq_dict[j]
            label=j+'_No'
            probability_dict[label]=(n_c+1)/(n+2)
            # Calculating probability of an attribute for a 'Yes'
            n_c=len(df[(df[i]==j) & (df[target]=='Positive')])
            label=j+'_Yes'
            probability_dict[label]=(n_c+1)/(n+2)

    return probability_dict

# This function calculates the probability of belonging to a particular class
def model_predict(test_instance,p_0,p_1,probability_dict):
    likelihood_prob_yes=likelihood_prob_no=1
    # Calculating likelihood tern for 'Yes' as well as 'No' in Bayesian formula
    # In here, we calculate the likelihood probability of belonging to class '0' or '1'
    for i in range(len(test_instance)):
        label=str(test_instance[i])+'_No'
        likelihood_prob_no*=probability_dict[label]
        label=str(test_instance[i])+'_Yes'
        likelihood_prob_yes*=probability_dict[label]

    # We calculate probabilities of '1' , that is, 'Yes' and that of '0' , that is of 'No'
    yes_prob=likelihood_prob_yes*p_1
    no_prob=likelihood_prob_no*p_0
    # We return the class for which a particular instance has maximum probability
    return 1 if (yes_prob > no_prob) else 0


# We load in the train dataset and calculate the corresponding accuracy of the model
df_train=pd.read_csv('naive_dataset.csv')
# Calculating probability that patients have diabetes
prob_0=len(df_train[df_train['Diabetic']==0])/df_train.shape[0]
# Calculating probability that patients don't have diabetes
prob_1=len(df_train[df_train['Diabetic']==1])/df_train.shape[0]
actual_values=df_train.iloc[:,-1]
# Replacing '0' with 'Negative' and '1' with 'Positive'
df_train['Diabetic'].replace([0,1],['Negative','Positive'],inplace=True)
prob_dict=model_fit(df_train)
y_pred=[]
for i in range(df_train.shape[0]):
    y_pred.append(model_predict(df_train.iloc[i,:-1].to_list(),prob_0,prob_1,prob_dict))

print('The sensitivity and specificity of the model fitting on training datastet is: ') ; print(classification_report(actual_values,y_pred))
print('The training accuracy of the classifier is: '+str(np.sum(y_pred==actual_values)/float(df_train.shape[0])))
# We load in the test dataset and calculate the corresponding accuracy of the model
df_test=pd.read_csv('naive_dataset_test.csv')
# Calculating probability that patients have diabetes
prob_0=len(df_test[df_test['Diabetic']==0])/df_test.shape[0]
# Calculating probability that patients don't have diabetes
prob_1=len(df_test[df_test['Diabetic']==1])/df_test.shape[0]
actual_values=df_test.iloc[:,-1]
# Replacing '0' with 'Negative' and '1' with 'Positive'
df_test['Diabetic'].replace([0,1],['Negative','Positive'],inplace=True)
prob_dict=model_fit(df_test)
y_pred=[]
for i in range(df_test.shape[0]):
    y_pred.append(model_predict(df_test.iloc[i,:-1].to_list(),prob_0,prob_1,prob_dict))
# Adding the predictions column to the test dataset
df_test['Predicted_value']=y_pred
# Extracting the predictions.csv file
predicitons=df_test.to_csv(r'D:\PyCharm Community Edition 2019.2.2\BDA-2\predictions.csv',index=None)
print('The sensitivity and specificity of the model fitting on testing datastet is: ') ; print(classification_report(actual_values,y_pred))
print('The testing accuracy of the classifier is: '+str(np.sum(y_pred==actual_values)/float(df_test.shape[0])))

