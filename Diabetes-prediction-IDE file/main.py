import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

#Convert csv into pandas dataframe
diabetes_data=pd.read_csv('Diabetes-prediction-IDE file\Diabetes.csv')
print(diabetes_data.head())

#Stastical parameters of dataset
print(diabetes_data.describe())

#Rows x columns
print(diabetes_data.shape)

#Number of instances for outcome column
print(diabetes_data['Outcome'].value_counts())

print(diabetes_data.groupby('Outcome').mean())

#0->Non Diabetic 
# 1-> Diabetic 
# Creating Data and Labels of dataset
X=diabetes_data.drop(columns='Outcome', axis=1)
Y=diabetes_data['Outcome']
print(X)
print('\n')
print(Y)
print('\n')

#Data Standardisation
scaler= StandardScaler()
scaler.fit(X)
standard_data=scaler.transform(X)
print(standard_data)
print('\n')

X= standard_data
Y=diabetes_data['Outcome']
print(X)
print('\n')
print(Y)
print('\n')

#Train test split
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
print('\n')

#Model Training
classifier = svm.SVC(kernel='linear')
#training the svm classifier
classifier.fit(X_train, Y_train)

#Model Evaluation and accuracy score
X_train_prediction= classifier.predict(X_train)
trained_data_accuracy= accuracy_score(X_train_prediction, Y_train)
print('Accuracy of trained data: ',trained_data_accuracy)
print('\n')

X_test_prediction= classifier.predict(X_test)
test_data_accuracy= accuracy_score(X_test_prediction, Y_test)
print('Accuracy of test data: ',test_data_accuracy)
print('\n')

#Making a Predictive system
input=(5,137,108,0,0,48.8,0.227,37)   #This is the required input field where user puts the required medical information
#conver list data to numpy array
input_as_numpy_array=np.asarray(input)
#reshaping the data
input_data_reshaped=input_as_numpy_array.reshape(1,-1)
#standardise the input data
std_data=scaler.transform(input_data_reshaped)
print(std_data)
print('\n')

#Result
prediction=classifier.predict(std_data)
print(prediction)
if(prediction[0]==0):
  print('The person is not diabetic')
  print('\n')
else:
  print('The person is diabetic')