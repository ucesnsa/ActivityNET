--POI_Activity Consolidation Algorithm--

from Utilities import pandas_to_spark
import pandas as pd
import os
import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession.0
# only run after findspark.init()
path = r'C:\Dev\nilufer\ActivityPOI-Data'
fileName_unlabel = 'Unlabelledactivities1.csv'
fileName_label = 'LabelledData.csv'
fileName_poi = 'POIs (1).xlsx'

os.chdir(path)
xl = pd.ExcelFile(fileName_poi)
# print (xl.sheet_names)
POI_df = xl.parse("Sheet1", converters={'Weekday_Closing1': str, 'Weekday_Opening1': str})
##POI wrangling
##'Weekday_Closing1' +values to be converted into 2400
POI_df.loc[POI_df['Weekday_Closing1'].str.contains('\+'), 'Weekday_Closing1'] = '2400'
POI_df['Weekday_Closing1'] = POI_df['Weekday_Closing1'].astype(int)
POI_df['Weekday_Opening1'] = POI_df['Weekday_Opening1'].astype(int)

labelled_df = pd.read_csv(fileName_label, delimiter=',')
labelled_df.insert(0, 'activity_id', range(1000, 1000 + len(labelled_df)))

labelled_df["StartTime"] = labelled_df["StartTime"] * 100
labelled_df["EndTime"] = labelled_df["EndTime"] * 100

labelled_df['StartTime'] = labelled_df['StartTime'].astype(int)
labelled_df['EndTime'] = labelled_df['EndTime'].astype(int)

# create spark session
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

print('transform pandas DF to spark DF')
labelled_df = labelled_df.rename(columns={'ActivityLocation': 'Station'})

print(labelled_df.shape)
print(labelled_df.columns)
print(POI_df.shape)
print(POI_df.columns)

# Pandas to Spark
poi_sdf = pandas_to_spark(spark, POI_df)
labelled_sdf = pandas_to_spark(spark, labelled_df)

# labelled_df = labelled_df.head(150000)

merged_sdf = labelled_sdf.join(poi_sdf, "Station", "inner")

# merged_sdf = labelled_sdf.join(poi_sdf, labelled_sdf("Station") === poi_sdf("Station"), "inner")

print('Output', merged_sdf.count(), len(merged_sdf.columns))
print(merged_sdf.head(2))

# remove activities that span two days
# merged_sdf = merged_df[merged_df["Activity Start Date/Time"].str[0:2] ==
#                      merged_df["Activity End Date/Time"].str[0:2]]

# filter POIs based on the time of the activity
merged_sdf = merged_sdf[(merged_sdf['Weekday_Opening1'] < merged_sdf['StartTime']) &
                        (merged_sdf['Weekday_Closing1'] > merged_sdf['EndTime'])]

print('Output1', merged_sdf.count(), len(merged_sdf.columns))

merged_sdf = merged_sdf[
    ['UserID', 'activity_id','ActivityDate','ActivityDay', 'category_0', 'checkinsCo', 'Station', 'ActivityDuration', 'ActivityDuration(hr)',
     'StartTime', 'EndTime', 'Labelled_Acts']]
merged_sdf.head(2)

# grp_sdf = merged_sdf.groupby(['UserID','activity_id','Station','ActivityDuration','ActivityDuration(hr)','StartTime','EndTime','Labelled_Acts'])['category_0'].value_counts()

# grp_sdf = merged_sdf.groupBy('UserID','activity_id','Station','ActivityDuration','ActivityDuration(hr)','StartTime','EndTime','Labelled_Acts','category_0').count()

gexprs = ['UserID', 'activity_id','ActivityDate','ActivityDay', 'Station', 'ActivityDuration', 'ActivityDuration(hr)', 'StartTime', 'EndTime',
          'Labelled_Acts']

grp_sdf_cnt = merged_sdf.groupBy(*gexprs).pivot("category_0").count()
grp_sdf_cnt_chkin = merged_sdf.groupBy(*gexprs).pivot("category_0").sum('checkinsCo')

print('Output1', grp_sdf_cnt.count(), len(grp_sdf_cnt.columns))
print('Output2', grp_sdf_cnt_chkin.count(), len(grp_sdf_cnt_chkin.columns))

print('completed')

grp_df_cnt = grp_sdf_cnt.toPandas()
grp_df_cnt_chkin = grp_sdf_cnt_chkin.toPandas()

# create excel writer object
writer = pd.ExcelWriter('output.xlsx')
# write dataframe to excel
grp_df_cnt.to_excel(writer, 'counts')
grp_df_cnt_chkin.to_excel(writer, 'check-ins')
# save the excel
writer.save()
print('DataFrame is written successfully to Excel File.')

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

dataset1=pd.read_csv(r'C:\Users\eng d\Google Drive\SSL\Data\JUNE2020forTAO\Combined_Data1_POIs1.csv')
#dataset1=pd.read_csv(r'C:\Users\eng d\Google Drive\SSL\Data\JUNE2020forTAO\Combined_Labelled_POIs1SA.csv')
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

#y_test.to_csv (r'C:\Users\eng d\Google Drive\SSL\Data\ANN_CV\Outputs\y_test.csv', index = True, header=True)
#y_pred.to_csv (r'C:\Users\eng d\Google Drive\SSL\Data\ANN_CV\Outputs\y_pred.csv', index = True, header=True)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.ticker as ticker
import itertools

##Evaluation
#heuristic_labels = ['H', 'W', 'BW', 'MD', 'AW', 'U']
#trippurpose_labels= ['0','1', '2', '3', '4', '5','6'] 
#trippurpose_labels = [ 'H', 'W','ENT','EAT','SHO','D/P','PT_W']

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

##cross-validation_comparison
#data if you need
dataset1=pd.read_csv('Combined_Labelled_POIs1.csv')

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


#Use the trained model to predict labels from the unlabelled data (SCD)

#unlabelled data
dataset2=pd.read_csv(r'C:\Users\eng d\Google Drive\SSL\Data\Dec2020\Combined_unlabelleddata_POIs1.csv')
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

#predict y_new from x_new
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


#Temporal variable of SCD (predicted)
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

 
dfH=pd.read_csv(r'C:\Users\eng d\Google Drive\TD\Data\Activities.csv')
# extract dfH='HOME activities',dfW= 'WORK activities', dfEAT= 'EAT activities', dfENT = 'ENT activities', dfSHO= 'SHO activities'
# dfPT_W = 'PTW activities', dfD_P= 'D/P activities'

from mpl_toolkits.axes_grid1 import make_axes_locatable
df_StartEnd_Time=pd.read_csv(r'C:\Users\eng d\Google Drive\TD\Data\TD_ActivityStartEndTime.csv')
dfH.head()
dfW.head()
dfEAT.head()
dfSHO.head()
df_StartEnd_Time.head()
#######################################
cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

from pandas.api.types import CategoricalDtype
cat_type = CategoricalDtype(categories=cats, ordered=True)
dfH['ActivityDay'] = dfH['ActivityDay'].astype(cat_type)
dfW['ActivityDay'] = dfW['ActivityDay'].astype(cat_type)
dfENT['ActivityDay'] = dfENT['ActivityDay'].astype(cat_type)
dfEAT['ActivityDay'] = dfEAT['ActivityDay'].astype(cat_type)
dfSHO['ActivityDay'] = dfSHO['ActivityDay'].astype(cat_type)
dfD_P['ActivityDay'] = dfD_P['ActivityDay'].astype(cat_type)
dfPT_W['ActivityDay'] = dfPT_W['ActivityDay'].astype(cat_type)

H_matrix=pd.pivot_table(dfH, values='ActivityDate', index=['ActivityDay'], columns='Duration_Hr', aggfunc='count', fill_value=0)
W_matrix=pd.pivot_table(dfW, values='ActivityDate', index=['ActivityDay'], columns='Duration_Hr', aggfunc='count', fill_value=0)
ENT_matrix=pd.pivot_table(dfENT, values='ActivityDate', index=['ActivityDay'], columns='Duration_Hr', aggfunc='count', fill_value=0)
EAT_matrix=pd.pivot_table(dfEAT, values='ActivityDate', index=['ActivityDay'], columns='Duration_Hr', aggfunc='count', fill_value=0)
SHO_matrix=pd.pivot_table(dfSHO, values='ActivityDate', index=['ActivityDay'], columns='Duration_Hr', aggfunc='count', fill_value=0)
D_P_matrix=pd.pivot_table(dfD_P, values='ActivityDate', index=['ActivityDay'], columns='Duration_Hr', aggfunc='count', fill_value=0)
PT_W_matrix=pd.pivot_table(dfPT_W, values='ActivityDate', index=['ActivityDay'], columns='Duration_Hr', aggfunc='count', fill_value=0)

sns.set_context('paper')
sns.set_style("white")
###start, end = ax.get_xlim()
###ax.xaxis.set_ticks(np.arange(0, 24, 4))

f,((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8),(ax9,ax10,ax11,ax12),(ax13,ax14,ax15,ax16)) = plt.subplots(4,4,sharex=False, sharey=False,figsize=(21, 14))

ax1 = sns.heatmap(H_matrix,cmap="YlGnBu",linewidths=0.5, cbar=False, linecolor='white',xticklabels=4,annot_kws={"size": 24},ax=ax1)
ax1.set_ylabel('')
ax1.set_xlabel('Duration (Hr)')
ax1.set_title('H',fontsize=12)
plt.xlim([0,24])
#start, end = ax.get_xlim(0,24)
ax1.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
ax1.set_xticklabels(['0', '4', '8', '12','16','20', '24'])
#ax10.set_xticklabels(['0', '4', '8', 'Duration (Hr)','20', '24'])

ax2 = sns.heatmap(W_matrix,cmap="YlGnBu",cbar=False,linewidths=0.5, linecolor='white',xticklabels=4,ax=ax2)
ax2.set_ylabel('')
ax2.set_xlabel('Duration (Hr)')
ax2.set_title('W',fontsize=12)
#plt.xlim([0,24])
ax2.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
#ax2.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax2.set_xticklabels(['0', '4', '8', '12','16','20', '24'])

ax3 = sns.heatmap(D_P_matrix,cmap="YlGnBu",linewidths=0.5, linecolor='white', cbar_kws={'label':'Counts'}, xticklabels=4, cbar=False, ax=ax3)
ax3.set_ylabel('')
ax3.set_xlabel('Duration (Hr)')
ax3.set_title('D/P',fontsize=12)
#plt.xlim([0,24])
ax3.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
#ax3.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax3.set_xticklabels(['0', '4', '8','12','16','20', '24'])

ax4 = sns.heatmap(PT_W_matrix,cmap="YlGnBu",linewidths=0.5, linecolor='white', cbar_kws={'label':'Counts'}, xticklabels=4, cbar=True, ax=ax4)
ax4.set_ylabel('')
ax4.set_xlabel('Duration (Hr)')
ax4.set_title('PTW',fontsize=12)
#plt.xlim([0,24])
ax4.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
#ax3.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax4.set_xticklabels(['0', '4', '8','12','16','20', '24'])

#f,((ax5,ax6,ax7,ax8)) = plt.subplots(1,4,sharex=True, sharey=False,figsize=(14, 5))
ax5=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['H_StartTime_Count']),data=df_StartEnd_Time,label = 'Start Time', color = '#225ea8',ax=ax5)
ax5=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['H_EndTime_Count']),data=df_StartEnd_Time, label = 'End Time', color = '#DD4968', ax=ax5)
ax5.set_title('')
ax5.set_ylabel('Counts', fontsize=12)
ax5.set_xlabel('')
#ax2.legend(loc='upper left', loc=0)
#f.legend(loc="upper left")
#ax5.set_title('', size =10)
ax5.legend(loc='upper left', fontsize=8)
plt.xlim([0,24])
#ax5.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax5.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
ax5.set_xticklabels(['4', '8', '12', '16', '20', '24'])
ax5.set_xticklabels(['0', '4', '8', '12','16','20', '24'])

ax6=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['W_StartTime_Count']),data=df_StartEnd_Time,  label = 'Start Time',color = '#225ea8',ax=ax6)
ax6=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['W_EndTime_Count']),data=df_StartEnd_Time, label = 'End Time', color = '#DD4968', ax=ax6)
ax6.set_ylabel('')
ax6.set_xlabel('')
ax6.legend(loc='upper left',fontsize=8)
plt.xlim([0,24])
#ax6.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax6.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
ax6.set_xticklabels(['0', '4', '8', '12','16','20', '24'])

ax7=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['D_P_StartTime_Count']),data=df_StartEnd_Time, label = 'Start Time',color = '#225ea8',ax=ax7)
ax7=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['D_P_EndTime_Count']),data=df_StartEnd_Time,label = 'End Time', color = '#DD4968', ax=ax7)
ax7.set_ylabel('')
ax7.set_xlabel('')
ax7.legend(loc='upper left',fontsize=8)
plt.xlim([0,24])
ax7.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
#ax7.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax7.set_xticklabels(['0', '4', '8', '12','16','20', '24'])

ax8=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['PT_W_StartTime_Count']),data=df_StartEnd_Time, label = 'Start Time',color = '#225ea8',ax=ax8)
ax8=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['PT_W_EndTime_Count']),data=df_StartEnd_Time,label = 'End Time', color = '#DD4968', ax=ax8)
ax8.set_ylabel('')
ax8.set_xlabel('Time (Hr)')
ax8.legend(loc='upper left',fontsize=8)
plt.xlim([0,24])
ax8.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
#ax7.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax8.set_xticklabels(['0', '4', '8', '12','16','20', '24'])

#f,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,sharex=True, sharey=False,figsize=(14, 5))

ax9 = sns.heatmap(ENT_matrix,cmap="YlGnBu",linewidths=0.5, cbar=False, linecolor='white',xticklabels=4,annot_kws={"size": 24},ax=ax9)
ax9.set_ylabel('')
ax9.set_xlabel('Duration (Hr)')
ax9.set_title('ENT',fontsize=12)
plt.xlim([0,24])
#start, end = ax.get_xlim(0,24)
ax9.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
ax9.set_xticklabels(['0', '4', '8', '12','20', '24'])
#ax10.set_xticklabels(['0', '4', '8', 'Duration (Hr)','20', '24'])

ax10 = sns.heatmap(EAT_matrix,cmap="YlGnBu",cbar=False,linewidths=0.5, linecolor='white',xticklabels=4,ax=ax10)
ax10.set_ylabel('')
ax10.set_xlabel('Duration (Hr)')
ax10.set_title('EAT',fontsize=12)
#plt.xlim([0,24])
ax10.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
#ax2.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax10.set_xticklabels(['0', '4', '8', '12','16','20', '24'])

ax11 = sns.heatmap(SHO_matrix,cmap="YlGnBu",linewidths=0.5, linecolor='white', cbar_kws={'label':'Counts'}, xticklabels=4, cbar=True, ax=ax11)
ax11.set_ylabel('')
ax11.set_xlabel('Duration (Hr)')
ax11.set_title('SHO',fontsize=12)
#plt.xlim([0,24])
ax11.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
#ax3.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax11.set_xticklabels(['0', '4', '8','12','16','20', '24'])

#f,((ax5,ax6,ax7,ax8)) = plt.subplots(1,4,sharex=True, sharey=False,figsize=(14, 5))
ax13=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['ENT_StartTime_Count']),data=df_StartEnd_Time,label = 'Start Time', color = '#225ea8',ax=ax13)
ax13=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['ENT_EndTime_Count']),data=df_StartEnd_Time, label = 'End Time', color = '#DD4968', ax=ax13)
ax13.set_title('')
ax13.set_ylabel('Counts', fontsize=12)
ax13.set_xlabel('Time (Hr)')
#ax2.legend(loc='upper left', loc=0)
#f.legend(loc="upper left")
#ax5.set_title('', size =10)
ax13.legend(loc='upper left', fontsize=8)
plt.xlim([0,24])
#ax5.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax13.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
ax13.set_xticklabels(['4', '8', '12', '16', '20', '24'])
ax13.set_xticklabels(['0', '4', '8', '12','16','20', '24'])

ax14=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['EAT_StartTime_Count']),data=df_StartEnd_Time,  label = 'Start Time',color = '#225ea8',ax=ax14)
ax14=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['EAT_EndTime_Count']),data=df_StartEnd_Time, label = 'End Time', color = '#DD4968', ax=ax14)
ax14.set_ylabel('')
ax14.set_xlabel('Time (Hr)')
ax14.legend(loc='upper left',fontsize=8)
plt.xlim([0,24])
#ax6.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax14.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
ax14.set_xticklabels(['0', '4', '8', '12','16','20', '24'])

ax15=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['SHO_StartTime_Count']),data=df_StartEnd_Time, label = 'Start Time',color = '#225ea8',ax=ax15)
ax15=sns.lineplot(x=df_StartEnd_Time['Hr'], y=(df_StartEnd_Time['SHO_EndTime_Count']),data=df_StartEnd_Time,label = 'End Time', color = '#DD4968', ax=ax15)
ax15.set_ylabel('')
ax15.set_xlabel('Time (Hr)')
ax15.legend(loc='upper left',fontsize=8)
plt.xlim([0,24])
ax15.set_xticks([0, 4, 8, 12, 16, 20, 24 ])
#ax7.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
ax15.set_xticklabels(['0', '4', '8', '12','16','20', '24'])

ax12.set_visible(False)
ax16.set_visible(False)

plt.savefig('C:\\Users\\eng d\\Google Drive\\TD\\Figures\\TD_all.tiff',dpi = 300)