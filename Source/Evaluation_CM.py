
##Evaluation
#heuristic_labels = ['H', 'W', 'BW', 'MD', 'AW', 'U']
#trippurpose_labels= ['0','1', '2', '3', '4', '5','6'] 
#trippurpose_labels = [ 'H', 'W','ENT','EAT','SHO','D/P','PT_W']

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.ticker as ticker
import itertools

#Loading the pre-trained model - Select one of the ANN weights that previously trained
import tensorflow as tf
print(tf.__version__)
print("Num GPUs Available: ",len(tf.config.experimental.list_physical_devices('GPU')))
model_filepath = './trained_model_withHM_withPOI1.h5'
classifier1 = tf.keras.models.load_model(model_filepath,custom_objects=None,compile=False)
print ('classifier1')

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
