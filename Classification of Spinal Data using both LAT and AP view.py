#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# l = []
# while True:
#   l.append(0)


# In[ ]:


get_ipython().system('pip install Augmentor')


# In[ ]:


import os
import matplotlib.pyplot as plt
print(os.listdir('drive/My Drive/Training'))


# In[ ]:


################################################################################ Current Model ##################################################

import numpy as np
import os
import pandas as pd
from PIL import Image


data_dir = "drive/My Drive/TestData/"

def load_data(data_dir, start, end):
    dataset = []
    directory = ""
    img_size = (256, 512)
    for i in range(start, end+1):
        print(i)
        if i == 245:
          continue
        folder = "Test (" + str(i) + ")"
        curr_data = []
        
        #AP view
        curr_dir = data_dir + directory + folder + "/AP/"
        ap = Image.open(curr_dir+"AP.jpg").resize(img_size)
        ap = ap.convert('RGB')
        try:
          ap_pedicle = Image.open(curr_dir+"Ap_Pedicle.png").resize(img_size)
        except FileNotFoundError:
          print("Problem in folder: ", folder)  
        
        try:
          ap_spinous_process = Image.open(curr_dir+"Ap_Spinous_Process.png").resize(img_size)
        except FileNotFoundError:
          print("Problem in folder: ", folder)  
        try:
          ap_vertebra = Image.open(curr_dir+"Ap_Vertebra.png").resize(img_size)
        except FileNotFoundError:
          print("Problem in folder: ", folder)  
        
        #LAT view
        curr_dir = data_dir + directory + folder + "/LAT/"
            
        lat = Image.open(curr_dir+"LAT.jpg").resize(img_size)
        lat = lat.convert('RGB')
        lat_anterior_vertebral_line = None
        lat_disk_height = None
        lat_posterior_vertebral_line = None
        lat_spinous_process = None
        lat_vertebra = None

        try:
          lat_anterior_vertebral_line = Image.open(curr_dir+"Lat_Anterior_Vertebral_Line.png").resize(img_size)
        except FileNotFoundError:
          print("Problem in folder: ", folder)  
            
        try:
          lat_disk_height = Image.open(curr_dir+"Lat_Disk_Height.png").resize(img_size)
        except FileNotFoundError:
          print("Problem in folder: ", folder)
            
        try:
          lat_posterior_vertebral_line = Image.open(curr_dir+"Lat_Posterior_Vertebral_Line.png").resize(img_size)
        except FileNotFoundError:
          print("Problem in folder: ", folder)
            
        try:
          lat_spinous_process = Image.open(curr_dir+"Lat_Spinous_Process.png").resize(img_size)
        except FileNotFoundError:
          print("Problem in folder: ", folder)
            
        try:
          lat_vertebra = Image.open(curr_dir+"Lat_Vertebra.png").resize(img_size)
        except FileNotFoundError:
          print("Problem in folder: ", folder)
            
        curr_data = [i,ap, lat, ap_pedicle, ap_spinous_process, ap_vertebra, lat_anterior_vertebral_line, 
                         lat_disk_height, lat_posterior_vertebral_line, lat_spinous_process, lat_vertebra ]
        dataset.append(curr_data)
        
    return dataset

m_test = 301
#load damaged data
print("Loading Test Data....")   
test_data = load_data(data_dir, 1, m_test)

test_df = pd.DataFrame(test_data)
column_names = ['ID', 'AP', 'LAT', 'AP_Pedicle', 'AP_Spinous_Process', "AP_Vertebreta", "Lat_Anterior_Vertebral_Line", 'Lat_Disk_Height', 'Lat_Posterior_Vertebral_Line', 'Lat_Spinous_Process', "Lat_Vertebra"]
test_df.columns = column_names


# In[ ]:


import numpy as np
x_input=[]
y_input=[[], [], [], [], []]

m_test = test_df.values.shape[0]
for i in range(m_test):
    x_input.append(np.asarray(test_df['AP'][i]))


# In[ ]:


from keras.models import model_from_json

# load json and create model
json_file = open('drive/My Drive/model_ap.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("drive/My Drive/model_ap.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
# loaded_model.compile(loss=iou_loss, optimizerAdam(), metrics[iou, dice_coef]])
y_pred = loaded_model.predict(np.array(x_input))


# In[ ]:


x_input = []
for i in range(m_test):
    x_input.append(np.asarray(test_df['LAT'][i]))


# In[ ]:


from keras.models import model_from_json

# load json and create model
json_file = open('drive/My Drive/model_lat.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_lat = model_from_json(loaded_model_json)
# load weights into new model
loaded_model_lat.load_weights("drive/My Drive/model_lat.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
# loaded_model.compile(loss=iou_loss, optimizerAdam(), metrics[iou, dice_coef]])
y_pred_lat = loaded_model_lat.predict(np.array(x_input))


# In[ ]:


print(y_pred[0].shape)


# In[ ]:


y_test = [[], [], []]
y = [[], [], []]
for i in range(len(y_pred)):
  for j in range(y_pred[0].shape[0]):
    y_test[i].append(np.asarray(Image.fromarray(y_pred[i][j].reshape(512, 256)).resize((224, 224))) )
    if test_df.values[j, i+3] is not None:
      y[i].append(np.asarray(test_df.values[j, i+3].resize((224, 224)).convert('L'), dtype=np.int )/255 )
    else:
      y[i].append(None)


# In[ ]:


y_test_lat = [[], [], [], [], []]
y_lat = [[], [], [], [], []]
for i in range(len(y_pred_lat)):
  for j in range(m_test):
    y_test_lat[i].append(np.asarray(Image.fromarray(y_pred_lat[i][j].reshape(512, 256)).resize((224, 224))) )
    if test_df.values[j, i+6] is not None:
      y_lat[i].append(np.asarray(test_df.values[j, i+6].resize((224, 224)).convert('L'), dtype=np.int )/255 )
    else:
      y_lat[i].append(None)


# In[ ]:


print(y[0][0].shape)
plt.imshow(y[0][6])
print(np.max(y[0][0]))


# In[ ]:


plt.imshow(y_pred[0][0].reshape(512, 256))


# In[ ]:


def dice_coef(y_true, y_pred):
  y_true_f = y_true.flatten()
  y_pred_f = y_pred.flatten()
  intersection = np.sum(y_true_f * (y_pred_f))
  return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def jaccard_coef(y_true, y_pred):
  y_true_f = y_true.flatten()
  y_pred_f = y_pred.flatten()
  intersection = np.sum(y_true_f * (y_pred_f))
  return (intersection) / (np.sum(y_true) + np.sum(y_pred) - intersection)

def fp(y_true, y_pred):
  y_true_f = y_true.flatten()
  y_pred_f = y_pred.flatten()
  neg = 0
  fp = 0
  for i in range(len(y_true_f)):
    if(y_true_f[i] <= 0.5 and y_pred_f[i] >= 0.5):
      fp += 1
    if(y_true_f[i] <= 0.5):
      neg += 1
  return fp /neg 

def fn(y_true, y_pred):
  y_true_f = y_true.flatten()
  y_pred_f = y_pred.flatten()
  pos = 0
  fn = 0
  for i in range(len(y_true_f)):
    if(y_true_f[i] >= 0.5 and y_pred_f[i] <= 0.5):
      fn += 1
    if(y_true_f[i] >= 0.5):
      pos += 1
  return (fn+1) / (pos+1) 


# In[ ]:


for j in range(3):
  f = [ ]
  g = [] 
  h = []
  k = []
  for i in range(m_test):
    if (y[j][i]) is not None:
      f.append( dice_coef(y[j][i], y_test[j][i]))
      g.append( jaccard_coef(y[j][i], y_test[j][i]))
      h.append( fp(y[j][i], y_test[j][i]))
      k.append( fn(y[j][i], y_test[j][i]))
      
  f = np.array(f)
  g = np.array(g)
  h = np.array(h)
  k = np.array(k)
  print(np.mean(f))
  print(np.mean(g))
  print(np.mean(h))
  print(np.mean(k))


# In[ ]:



for j in range(5):
  f = [ ]
  g = []
  h = []
  k = []
  for i in range(m_test):
    if y_lat[j][i] is not None:
      f.append( dice_coef(y_lat[j][i], y_test_lat[j][i]))
      g.append( jaccard_coef(y_lat[j][i], y_test_lat[j][i]))
      h.append( fp(y_lat[j][i], y_test_lat[j][i]))
      k.append( fn(y_lat[j][i], y_test_lat[j][i]))
  f = np.array(f)
  g = np.array(g)
  h = np.array(h)
  k = np.array(k)
  print(np.mean(f))
  print(np.mean(g))
  print(np.mean(h))
  print(np.mean(k))


# In[ ]:


y_final = [[], [], []]
for i in range(len(y_pred)):
  for j in range(y_pred[0].shape[0]):
    temp = np.round(np.array(y_test[i][j], dtype=np.int)/255)
    y_final[i].append(np.asarray(temp, dtype=np.bool))


# In[ ]:


y_final_lat = [[], [], [], [], []]
for i in range(len(y_pred_lat)):
  for j in range(m_test):
    temp = np.round(np.array(y_test_lat[i][j], dtype=np.int)/255)
    y_final_lat[i].append(np.asarray(temp, dtype=np.bool))


# In[ ]:



ap_pedicle = []
for j in range(m_test):
  ap_pedicle.append(np.array(y_final[0][j]))
ap_pedicle = np.array(ap_pedicle)

ap_spinous = []
for j in range(m_test):
  ap_spinous.append(np.array(y_final[1][j]))
ap_spinous = np.array(ap_spinous)

ap_vertebra = []
for j in range(m_test):
  ap_vertebra.append(np.array(y_final[2][j]))
ap_vertebra = np.array(ap_vertebra)

lat_ant = []
for j in range(m_test):
  lat_ant.append(np.array(y_final_lat[0][j]))
lat_ant = np.array(lat_ant)

lat_disk = []
for j in range(m_test):
  lat_disk.append(np.array(y_final_lat[1][j]))
lat_disk = np.array(lat_disk)

lat_post = []
for j in range(m_test):
  lat_post.append(np.array(y_final_lat[2][j]))
lat_post = np.array(lat_post)

lat_spi = []
for j in range(m_test):
  lat_spi.append(np.array(y_final_lat[3][j]))
lat_spi = np.array(lat_spi)

lat_vert = []
for j in range(m_test):
  lat_vert.append(np.array(y_final_lat[4][j]))
lat_vert = np.array(lat_vert)


# In[ ]:


data_dict = {}
for i in range(m_test):
  id_num = i+1
  if id_num >= 245:
    id_num += 1
  data_dict[id_num] = {"Ap_Pedicle": ap_pedicle[i], "Ap_Spinous_Process": ap_spinous[i],  "Ap_Vertebra": ap_vertebra[i] , "Lat_Anterior_Vertebral_Line" :  lat_ant[i] , "Lat_Disk_Height" : lat_disk[i], "Lat_Posterior_Vertebral_Line": lat_post[i] , "Lat_Spinous_Process": lat_spi[i] , "Lat_Vertebra" : lat_vert[i]}


# In[ ]:


# !pip install pickle
import pickle as pkl

pkl.dump( data_dict, open( "segmentation.pkl", "wb" ) )


# In[ ]:


import keras.applications.resnet as resnet
from keras import models
import numpy as np 
from keras import backend as keras
from keras.models import load_model
from keras.models import model_from_json


# In[ ]:



# load json and create model
json_file = open('drive/My Drive/model_classification.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("drive/My Drive/model_classification.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
# loaded_model.compile(loss=iou_loss, optimizerAdam(), metrics[iou, dice_coef]])


# In[ ]:


x_input1 = []
x_input2 = []
for i in range(m_test):
    x_input1.append(np.asarray(test_df['AP'][i]))
    x_input2.append(np.asarray(test_df['LAT'][i]))

x_input1 = np.array(x_input1)
x_input2 = np.array(x_input2)


# In[ ]:


base_model = resnet.ResNet50(include_top=False, weights='imagenet', layers= 1, input_shape=(512, 256, 3), pooling=None)

preds1 = base_model.predict(x_input1)
#print(preds1.shape)
preds2 = base_model.predict(x_input2)
#print(preds2.shape)


# In[ ]:


predictions = model.predict([preds1, preds2])


# In[ ]:


label_pred = (np.array(predictions>0.5, dtype=np.int))
label_dict = {}
for i in range(m_test):
  id_num = i+1
  if id_num >= 245:
    id_num += 1
  label_dict[id_num] = label_pred[i][0]


# In[ ]:


pkl.dump( label_dict, open( "classification.pkl", "wb" ) )


# In[ ]:


import numpy as np
x_input=[]
y_input=[]
z_input=[]
j = 0
k = 0

l=0
m=0
for i in range(len(np.asarray(damaged_df['AP']))):
    #print(damaged_df['vAP_Spinous_Process'][i].mode)
    x_input.append(np.asarray(damaged_df['AP'][i]))
    j+=1
    #print(damaged_df['AP'][i])
    a = np.asarray(damaged_df['AP'][i])
    #print(np.asarray(damaged_df['AP'][i]).shape)
    a1 = np.stack((np.asarray(damaged_df['AP_Pedicle'][i]), np.asarray(damaged_df['AP_Spinous_Process'][i]), np.asarray(damaged_df["AP_Vertebreta"][i])), axis = 2)
    #print(a1.shape)
    y_input.append(a1)
    
    z_input.append(1)


for i in range(len(np.asarray(normal_df['AP']))):
    #print(damaged_df['AP_Spinous_Process'][i].mode)
    k+=1
    x_input.append(np.asarray(normal_df['AP'][i]))
    #print(damaged_df['AP'][i])
    a = np.asarray(normal_df['AP'][i])
    #print(np.asarray(damaged_df['AP'][i]).shape)
    a1 = np.stack((np.asarray(normal_df['AP_Pedicle'][i]), np.asarray(normal_df['AP_Spinous_Process'][i]), np.asarray(normal_df["AP_Vertebreta"][i])), axis = 2)
    #print(a1.shape)
    y_input.append(a1)
    z_input.append(0)

print(j)
print(k)
import numpy as np
a_input=[]
b_input=[]
c_input=[]
for i in range(len(np.asarray(damaged_df['LAT']))):
    #print(damaged_df['AP_Spinous_Process'][i].mode)
    l+=1
    a_input.append(np.asarray(damaged_df['LAT'][i]))
    #print(damaged_df['AP'][i])
    a = np.asarray(damaged_df['LAT'][i])
    a1 = np.stack((np.asarray(damaged_df["LAT_Anterior_Vert_Line"][i]), np.asarray(damaged_df['LAT_disk_height'][i]), np.asarray(damaged_df['LAT_Posterior_Vert_Line'][i]), np.asarray(damaged_df['LAT_Spinal_Process'][i]), np.asarray(damaged_df["LAT_Vertebra"][i])), axis = 2)
    #print(a1.shape)
    b_input.append(a1)
    c_input.append(1)


for i in range(len(np.asarray(normal_df['LAT']))):
  a_input.append(np.asarray(normal_df['LAT'][i]))
  a = np.asarray(normal_df['LAT'][i])
  a1 = np.stack((np.asarray(normal_df["LAT_Anterior_Vert_Line"][i]), np.asarray(normal_df['LAT_disk_height'][i]), np.asarray(normal_df['LAT_Posterior_Vert_Line'][i]), np.asarray(normal_df['LAT_Spinal_Process'][i]), np.asarray(normal_df["LAT_Vertebra"][i])), axis = 2)
  b_input.append(a1)
  c_input.append(0)
  m+=1

print(l)
print(m)



# In[ ]:


class_train1 = np.array(np.divide(x_input, 1))
#print(class_train1.shape)
class_train_masks1 = np.array(np.divide(y_input, 1))
class_train2 = np.array(np.divide(a_input, 1))
class_train_masks2 = np.array(np.divide(z_input, 1))
#print(class_train2.shape)
labels = np.array(c_input)
import matplotlib.pyplot as plt

# plt.imshow(class_train1[0, :, :, 1])
# plt.show()
# plt.imshow(class_train2[0, :, :, 1])
# plt.show()

labels2 = np.array(z_input)
#print(np.max(class_train1, axis =1))

a = np.random.permutation(667)
#print(a)
#a = np.random.shuffle(a)
#a = np.random.shuffle(a)

class_train_masks1 = class_train_masks1[a]
class_train_masks2 = class_train_masks2[a]

class_train1 = class_train1[a]#[class_train[a[i])for i in range 667]
labels = labels[a]
labels2 = labels2[a]
class_train2 = class_train2[a]
#print(labels)
#print(labels2)


# In[ ]:


#base_model_ = resnet.ResNet50(include_top=False, weights='imagenet', layers= 1, input_shape=(512, 256, 3), pooling=None)
#base_model_.summary()

#preds1 = base_model.predict(class_train1)
#preds2 = base_model.predict(class_train2)
#del model
#del modelA
#del modelB
from keras.layers import *
import keras.applications.resnet as resnet
from keras import models
from keras import layers
from keras import optimizers
from keras import applications
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import tensorflow as tf
import keras.activations as ac
from keras.models import load_model
import keras.models as models
import keras.losses
base_model = resnet.ResNet50(include_top=False, weights='imagenet', layers= 1, input_shape=(512, 256, 3), pooling=None)
class_train1=np.divide(class_train1, 255)
class_train1=np.divide(class_train2, 255)

preds1 = base_model.predict(class_train1)
#print(preds1.shape)
preds2 = base_model.predict(class_train2)
#print(preds2.shape)


# In[ ]:


### Alternative model CNN
from keras.layers import Input
inx = Input(shape=(512,256,3), name="Input")
x = Conv2D(32,(3,3),padding='same',activation='relu')(inx)
x = Conv2D(32,(3,3),padding='same',activation='relu')(inx)
x = MaxPooling2D()(x)
x = Dropout(0.2)(x)
x = Conv2D(64,(3,3),padding='same',activation='relu')(x)
x = Conv2D(64,(3,3),padding='same',activation='relu')(x)
x = MaxPooling2D()(x)
x = Dropout(0.3)(x)
x = Conv2D(128,(3,3),padding='same',activation='relu')(x)
x = Conv2D(128,(3,3),padding='same',activation='relu')(x)
x = MaxPooling2D()(x)
x = Dropout(0.5)(x)
x = Conv2D(256,(3,3),padding='same',activation='relu')(x)
x = Conv2D(256,(3,3),padding='same',activation='relu')(x)
x = MaxPooling2D()(x)
x = Dropout(0.6)(x)
x = Conv2D(512,(3,3),padding='same',activation='relu')(x)
x = Conv2D(512,(3,3),padding='same',activation='relu')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(128,activation='relu')(x)
outx = Dense(1,activation='sigmoid')(x)
doc = Model(inputs=[inx], outputs=[outx])
doc.compile(optimizer=SGD(lr = 1e-4, momentum = 1e-2), loss='binary_crossentropy', metrics=["binary_accuracy"])
doc.summary()
history = doc.fit(class_train1, labels, batch_size= 4, shuffle = True, validation_split=0.2, epochs = 25)


# In[ ]:


model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
model.summary()
print(labels.shape)

model.fit(class_train1, labels, batch_size = 8, epochs=50, validation_split=0.2, shuffle = True)


# In[ ]:



base_model_A = resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(512, 256, 3), pooling=None)
base_model_B = resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(512, 256, 3), pooling=None)

for i in base_model_A.layers:
  i.trainable = False

for i in base_model_B.layers:
  i.trainable = False

a = Sequential()
b = Sequential()

a.add(base_model_A)
b.add(base_model_B)



inputsA = Input(shape= (16, 8, 2048,))
inputsB = Input(shape= (16, 8, 2048,))

x = Flatten()(a.output)
x = Dropout(.2)(x)
x = Dense(128, activation= 'relu')(x)
modelA = Model(input = a.input, output=x)


y = Flatten()(b.output)
y = Dropout(.2)(y)
y = Dense(128, activation= 'relu')(y)
modelB = Model(input = b.input, output=y)



from keras.layers import *

mergedOut = Concatenate()([modelA.output, modelB.output])
mergedOut = Dropout(.3)(mergedOut)
mergedOut = Dense(128, activation='relu')(mergedOut)
mergedOut = Dropout(.4)(mergedOut)
mergedOut = Dense(64, activation='relu')(mergedOut)
mergedOut = Dropout(.5)(mergedOut)
mergedOut = Dense(1, activation='sigmoid')(mergedOut)
model1 = Model(inputs = [a.input, b.input], output = mergedOut)
model1.summary()
model1.compile(optimizer=Adam(lr=1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
# mergedOut = Dense(8, activation='relu')(mergedOut)
# mergedOut = Dropout(.35)(mergedOut)

# # output layer
# 


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model Accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


####TSNE code
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
AP_fit = preds1[:,:,:,:]
AP_fit = AP_fit.reshape([667,14*6*2048])
LAT_fit = preds2[:,:,:,:]
LAT_fit = LAT_fit.reshape([667,14*6*2048])

svd = TruncatedSVD(n_components=50, random_state=42)
AP_reduced = svd.fit_transform(AP_fit)
LAT_reduced = svd.fit_transform(LAT_fit)
AP_embedded = TSNE(n_components=2, perplexity = 15).fit_transform(AP_reduced)
LAT_embedded = TSNE(n_components=2, perplexity = 15).fit_transform(LAT_reduced)
plt.scatter(AP_embedded[0:321,0], AP_embedded[0:321,1],color = 'red') #Damaged AP - blue
plt.scatter(AP_embedded[321:,0], AP_embedded[321:,1], color = 'blue') #Normal AP - orange

plt.show()

plt.scatter(LAT_embedded[0:321,0], LAT_embedded[0:321,1], color= 'red') #Damaged AP
plt.scatter(LAT_embedded[321:,0], LAT_embedded[321:,1], color='blue') #Normal AP
plt.show()

