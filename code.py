#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing dataset
dataset = pd.read_csv('data.csv')
df = dataset.copy()

# Inspecting data
df.head()
df.info()
df.isnull().sum() # There are no null values
df.describe()

df['Geography'].value_counts()   # There are only 3 countries
df['Gender'].value_counts()
df['Tenure'].value_counts()
df['NumOfProducts'].value_counts()  # range of values - 1 to 4

# We do not need surname as surname is not a factor to determine whether the customer exited the bank
# Hence dropping surname attribute
del df['Surname']

# Row number is used only for identifying the number of row, which again is not useful
# Hence dropping rownumber attribute as well
del df['RowNumber']

# We have 2 categorical deatures so we will do one hot encoding
# Encoding Gender attribute
df = pd.get_dummies(df, columns = ['Gender'], prefix = ['Gender'])
# Encoding Geaography attribute
df = pd.get_dummies(df, columns = ['Geography'], prefix = ['Geography'])

# Correlation matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize = (12, 9))
sns.heatmap(corrmat, vmax = 1, square = True)

# Now our dataset is ready as we have removed unwanted attributes and turned each column to numeric values

#Splitting dataset
X = df.drop(['Exited', 'Geography_Spain', 'Gender_Male'], axis = 1)
X = X.iloc[:, :].values

y = df.iloc[:, 9].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# BUILDING ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

#Creating the input layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 12))

# Creating the hidden layers
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Creating the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling(applying stochastic gradient descent) the ANN 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the data
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
# I got an accuracy of 83.70%


# Getting the prediction for train set
y_pred_train = classifier.predict(X_train)

# Getting the prediction for test set
y_pred_test = classifier.predict(X_test)

#The values in y_pred_train and y_pred_test are probabilities of whether the person will leave the bank, we will 
# now make these values discrete(i,e, 0 or 1)
y_pred_train = (y_pred_train > 0.5)
y_pred_test = (y_pred_test > 0.5)

# Implementing confusion matrix for model evaluation
from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred_test)

# Printing the accuracies fot train and test set
print("Accuracy for training set = {}".format((cm_train[0, 0] + cm_train[1, 1])/len(X_train)))
print("Accuracy for test set = {}".format((cm_test[0, 0] + cm_test[1, 1])/len(X_test)))
