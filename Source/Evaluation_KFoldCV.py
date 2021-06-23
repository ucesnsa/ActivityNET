
##cross-validation_comparison

import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.layers import Dropout
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
import numpy as np

#data
dataset1=pd.read_csv('GroundTruth_CombinedwithPOIs.csv')

col_features = ['ActivityDuration(hr)','StartTime',  'EndTime','Arts & Entertainment','Colleges & Universities','Eating','Nightlife Spots','Outdoors & Recreation','Professional & Other Places','Residences','Shopping','Travel & Transport'] 
col_label = ['LabelledActivities']
col_station = ['ActivityLocation']
df_xy1 = dataset1
df_x1 = dataset1[col_features]
df_y1 = dataset1[col_label]

encoder = LabelEncoder()
encoder.fit(df_y1)
encoded_y = encoder.transform(df_y1)
df_binary_y1 = pd.get_dummies(encoded_y)

sc = StandardScaler()
scaled_x1 = sc.fit_transform(df_x1)
scaled_x1 = sc.transform(df_x1)
df_scaled_x1 = pd.DataFrame(data=scaled_x1)
df_scaled_x1.columns = col_features

#df_x1_train, df_x1_test, df_y1_train, df_y1_test = train_test_split(df_scaled_x1, df_binary_y1, test_size = 0.3, random_state = 0) 
### here i assigned the split to a random state

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 100, kernel_initializer ='uniform', activation = 'relu', input_dim =12))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units = 60, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units = 7, kernel_initializer ='uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',  metrics = ['accuracy'])
    return classifier

X = df_scaled_x1[col_features].values
y = df_binary_y1.values
classifier1 = build_classifier()

from sklearn.model_selection import KFold 
kf = KFold(n_splits=10, random_state=None) 

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index] 
    y_train, y_test = y[train_index], y[test_index]
          
    model = classifier1.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size = 32, epochs = 1000
                            ,verbose=0)
    
    score = classifier1.evaluate(X_test, y_test,  batch_size = 32)
    
    print(score)
    print(np.var(score))
