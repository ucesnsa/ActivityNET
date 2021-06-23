#Create NN model from the combined data
#Two combined data available(SSCD + POIs (used for training and evaluation) & SCD + POIs (used for prediction))
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd
import numpy as np
import uuid
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="ticks", color_codes=True)
sns.set(style="ticks")
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn.metrics as metrics

dataset1=pd.read_csv(r'C:\ActivityNET\Data\GroundTruth_CombinedwithPOIs.csv')
print(dataset1.shape)
dataset1.head(5)

col_features = ['ActivityDuration(hr)','StartTime',  'EndTime','Arts & Entertainment','Colleges & Universities','Eating','Nightlife Spots','Outdoors & Recreation','Professional & Other Places','Residences','Shopping','Travel & Transport'] 

col_label = ['Labelled_Acts']
col_station = ['ActivityLocation']
df_xy1 = dataset1

#multiple scenarios for class imbalance problem
# balance by under sampling 
df_class_0 = dataset1[dataset1['Labelled_Acts'] == 0].sample(427)
df_class_1 = dataset1[dataset1['Labelled_Acts'] == 1].sample(427)
df_class_2 = dataset1[dataset1['Labelled_Acts'] == 2].sample(427)
df_class_3 = dataset1[dataset1['Labelled_Acts'] == 3].sample(427)
df_class_4 = dataset1[dataset1['Labelled_Acts'] == 4].sample(427)
df_class_5 = dataset1[dataset1['Labelled_Acts'] == 5].sample(427)
df_class_6 = dataset1[dataset1['Labelled_Acts'] == 6].sample(427)

# balance by over sampling 
#df_class_0 = dataset1[dataset1['Labelled_Acts'] == 0].sample(2894, replace=True)
#df_class_1 = dataset1[dataset1['Labelled_Acts'] == 1].sample(2894, replace=True)
#df_class_2 = dataset1[dataset1['Labelled_Acts'] == 2].sample(2894, replace=True)
#df_class_3 = dataset1[dataset1['Labelled_Acts'] == 3].sample(2894, replace=True)
#df_class_4 = dataset1[dataset1['Labelled_Acts'] == 4].sample(2894, replace=True)
#df_class_5 = dataset1[dataset1['Labelled_Acts'] == 5].sample(2894, replace=True)
#df_class_6 = dataset1[dataset1['Labelled_Acts'] == 6].sample(2894, replace=True)

#No class imbalance
#df_class_2 = dataset1[dataset1['Labelled_Acts'] == 2]
#df_class_3 = dataset1[dataset1['Labelled_Acts'] == 3]
#df_class_4 = dataset1[dataset1['Labelled_Acts'] == 4]
#df_class_5 = dataset1[dataset1['Labelled_Acts'] == 5]
#df_class_6 = dataset1[dataset1['Labelled_Acts'] == 6]
dataset1 = pd.concat([df_class_0,df_class_1,df_class_2,df_class_3,df_class_4,df_class_5,df_class_6], axis=0)
#without home and work activities
#dataset1 = pd.concat([df_class_2,df_class_3,df_class_4,df_class_5,df_class_6], axis=0)

df_x1 = dataset1[col_features]
df_y1 = dataset1[col_label]

print (dataset1.groupby('Labelled_Acts')['Labelled_Acts'].value_counts())

df_x1.head()
df_x1.shape

encoder = LabelEncoder()
encoder.fit(df_y1)
encoded_y = encoder.transform(df_y1)
df_binary_y1 = pd.get_dummies(encoded_y)
#df_binary_y1.columns = ['Y0','Y1','Y2','Y3']
print (df_binary_y1[0:6])

#scale
sc = StandardScaler()
scaled_x1 = sc.fit_transform(df_x1)
scaled_x1 = sc.transform(df_x1)
print (type(scaled_x1))
df_scaled_x1 = pd.DataFrame(data=scaled_x1)
print (type(df_scaled_x1))
#adding column for station name
df_scaled_x1.columns = col_features
#df_scaled_x1[col_station] = df_station
#seed = 5
df_x1_train, df_x1_test, df_y1_train, df_y1_test = train_test_split(df_scaled_x1, df_binary_y1, test_size = 0.3,random_state=0, shuffle= True)

print (df_x1_train[:6])
print (df_y1_train[:6])

#DNN model
classifier1 = Sequential()
classifier1.add(Dense(units = 100, kernel_initializer ='uniform', activation = 'relu', input_dim =12))
####classifier1.add(Dense(units = 100, kernel_initializer ='uniform', activation = 'relu', input_dim =12))
#classifier1.add(Dropout(0.5))
classifier1.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))
classifier1.add(Dropout(0.5))
classifier1.add(Dense(units = 60, kernel_initializer = 'uniform', activation = 'relu'))
#####classifier1.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'elu'))
#####classifier1.add(Dense(units =12, kernel_initializer = 'uniform', activation = 'relu'))
#####classifier1.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'elu'))
classifier1.add(Dropout(0.5))
classifier1.add(Dense(units = 7, kernel_initializer ='uniform', activation = 'softmax'))

adam = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.005, amsgrad=False)
classifier1.compile(optimizer = 'adam', loss = 'binary_crossentropy',  metrics = ['accuracy'])
model = classifier1.fit(df_x1_train[col_features], df_y1_train, validation_data = (df_x1_test[col_features], df_y1_test), batch_size = 64, epochs = 700)
score=classifier1.evaluate(df_x1_test[col_features], df_y1_test,  batch_size = 64)
print (score)

# list all data in history
print(model.history.keys())

# Plot the history for accuracy
plt.plot(model.history['acc'])
plt.plot(model.history['val_acc'])
plt.title('ANN model')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'testing'], loc='lower right')
plt.show()

#plt.savefig('29__accuracy.jpg')

# Plot the history for loss
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('ANN model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'testing'], loc='upper right')
plt.show()

#save the model
import os
path = r'C:\Users\eng d\Google Drive\SSL\Data\Dec2020'
os.chdir(path)
classifier1.save('trained_model_withHM_withPOI1.h5') 
print(score)
print(np.var(score))

#classifier1.save('trained_model_withHM_withPOI1.h5') 
#Loading the pre-trained model - Select one of the ANN weights that previously trained
import tensorflow as tf
print(tf.__version__)
print("Num GPUs Available: ",len(tf.config.experimental.list_physical_devices('GPU')))
model_filepath = './trained_model_withHM_withPOI1.h5'
classifier1 = tf.keras.models.load_model(model_filepath,custom_objects=None,compile=False)
print ('classifier1')

#Predicting the Test set results
y_pred = classifier1.predict(df_x1_test[col_features])
y_pred = (y_pred > 0.5).astype(int) 
#y_test = y_test.values
print (df_y1_test.shape, y_pred.shape)
print (type(df_y1_test), type(y_pred))

#print (df_y1_test[0:6])
#print (y_pred[0:6])

df_test = pd.DataFrame(df_y1_test)
df_pred = pd.DataFrame(y_pred)

y_test = df_test.idxmax(axis=1)
y_pred = df_pred.idxmax(axis=1)

print (y_test[0:10])
print (y_pred[0:10])

#y_test.to_csv (r'C:\Users\eng d\y_test.csv', index = True, header=True)
#y_pred.to_csv (r'C:\Users\eng d\y_pred.csv', index = True, header=True)

#CM

cm = confusion_matrix(y_test, y_pred)
cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)
#cm = np.round(cm.astype('float') / cm.sum(),2) #(axis=1)[:, np.newaxis])
print(cm)
#fig, ax = plt.subplots(figsize=(6,3))
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm,cmap=plt.cm.Blues)
#gray_r
#plt.title('Spatial-Temporal')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.tight_layout()

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black",
             fontsize=12)
#plt.xlabel('Predicted label')
#plt.ylabel('True label')
#plt.show(
#plt.savefig(r'C:\Users\eng d\Google Drive\SSL\Data\JUNE2020forTAO\Figure\12Aug2020\CM_HW_withPOIs1.tiff', dpi = 300)
plt.show()

#Use the trained model to predict labels from the unlabelled data (SCD)

#unlabelled data
dataset2=pd.read_csv(r'C:\ActivityNET\Data\UnlabelledData_CombinedwithPOIs.csv')
dataset2.fillna(0)
dataset2['Arts & Entertainment'] = dataset2['Arts & Entertainment'].fillna(0)
dataset2 = dataset2.fillna(0)

#print (dataset2.shape)
#print(dataset2[0:10])
#df_x_new= dataset2[col_features][0:14947]

df_x_new= dataset2[col_features].fillna(0)
#print (type(df_x_new))
print('complete')

#scale unlabelled data and predict the first 4 column of unlabelled data
sc = StandardScaler()
df_x_new = pd.DataFrame( data = sc.fit_transform(df_x_new))
df_x_new.columns = col_features
#df_x_new['ActivityLocation'] = dataset2[['ActivityLocation']]
#print (type(df_x_new))
#print (df_x_new)
print (df_x_new[:4])
print (df_x_new.shape)
#print(df_x_new[col_features])

#predict y_new (y2_pred) from x_new
y2_pred=classifier1.predict(df_x_new[col_features])
y2_pred =y2_pred.round(decimals=2, out=None)
#y2_pred = (y2_pred > 0.5).astype(int) 
#y2_pred

# find the index of the max convert confidence 
y2_pred = np.argmax(y2_pred, axis=1)
df_y2_pred = pd.DataFrame(y2_pred)
df_y2_pred

#print (type (df_y2_pred), df_y2_pred.shape)
print (df_y2_pred[0:200])
df_y2_pred.to_csv (r'C:\Users\eng d\Google Drive\SSL\Data\Dec2020\y2_pred.csv', index = True, header=True)

