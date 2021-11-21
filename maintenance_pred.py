# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 18:12:21 2021

@author: doguilmak

dataset: https://www.kaggle.com/shivamb/machine-predictive-maintenance-classification

"""
#%%
# 1. Importing Libraries

import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import time
import warnings
warnings.filterwarnings('ignore')

#%%
# 2. Data Preprocessing

# 2.1. Uploading data
start = time.time()
df = pd.read_csv('predictive_maintenance.csv')
FAILURE_TYPES=df['Failure_Type']

# 2.2. Removing Unnecessary Columns
df.drop(['UDI', 'Product_ID'], axis = 1, inplace = True)

# 2.3. Plot Faiure Types on Histogram
plt.figure(figsize = (12, 12))
sns.set_style('whitegrid')    
sns.histplot(data=FAILURE_TYPES)
plt.title("Failure Types on Histogram")
plt.xlabel("Failure Types")
plt.ylabel("Failures in Total")
#plt.savefig('Plots/hist_failure_types')
plt.show()

# 2.4. Looking for anomalies and duplicated datas
print(df.isnull().sum())
print("\n", df.head(10))
print("\n", df.describe().T)
print("\n{} duplicated.".format(df.duplicated().sum()))
print('\n', df.info())
print('\n', FAILURE_TYPES.value_counts(), '\n')

# 2.5. Label Encoding
from sklearn.preprocessing import LabelEncoder
df = df.apply(LabelEncoder().fit_transform)

# 2.6. Determination of dependent and independent variables
X = df.drop("Failure_Type", axis = 1)
y = df["Failure_Type"]

# 2.7. Splitting test and train 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# 2.8. Scaling datas
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test) # Apply the trained

#%%
# 3 Artificial Neural Network
"""
# 3.1 Loading Created Model
model = load_model('model.h5')

# 3.2 Checking the Architecture of the Model
model.summary()
"""

model = Sequential()

# 3.3. Adding the input layer and the first hidden layer
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 7))

# 3.4. Adding the second hidden layer
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

# 3.5. Adding the output layer
model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))

# 3.6. Compiling the ANN
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# 3.7. Fitting the ANN to the Training set
model_history=model.fit(X_train, y_train, batch_size=200 , epochs = 124, validation_split=0.13)

# 3.8. Predicting the Test set results
y_pred = model.predict(X_test)


# 3.9. Plot accuracy and val_accuracy
print(model_history.history.keys())
model.summary()
model.save('model.h5')
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 12))
sns.set_style('whitegrid')
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('ANN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
plt.savefig('Plots/model_acc')
plt.show()

plt.figure(figsize=(12, 12))
sns.set_style('whitegrid')
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('ANN Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.savefig('Plots/model_loss')
plt.show()


"""
# 3.10. Plotting model 
from keras.utils import plot_model
plot_model(model, "model_input_and_output.png", show_shapes=True)


from ann_visualizer.visualize import ann_viz
try:
    ann_viz(model, view=True, title="", filename="ann")
except:
    print("PDF saved.")
"""

#%%
# 4 XGBoost

# 4.1 Importing Libraries
from xgboost import XGBClassifier

classifier=XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# 4.2. Building Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_pred, y_test)  # Comparing results
print("\nConfusion Matrix(XGBoost):\n", cm2)

# 4.3. Accuracy Score
from sklearn.metrics import accuracy_score
print(f"\nAccuracy score(XGBoost): {accuracy_score(y_test, y_pred)}")

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
