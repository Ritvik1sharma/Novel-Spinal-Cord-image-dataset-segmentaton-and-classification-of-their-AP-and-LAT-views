#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install Augmentor')


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:





# In[3]:


import os
import matplotlib.pyplot as plt
print(os.listdir('drive/My Drive/Training'))


# In[4]:


################################################################################ Current Model ##################################################

import numpy as np
import os
import pandas as pd
from PIL import Image


data_dir = "drive/My Drive/Training/"

def load_data(data_dir, directory, start, end):
    dataset = []
    img_size = (256, 512)
    for i in range(start, end+1):
        print(i)
        folder = "ID (" + str(i) + ")"
        curr_data = []
        try:
            #AP view
            curr_dir = data_dir + directory + folder + "/AP/"
            ap = Image.open(curr_dir+"AP.jpg").resize(img_size)
            ap = ap.convert('RGB')
            ap_pedicle = Image.open(curr_dir+"Ap_Pedicle.png").resize(img_size)
            ap_spinous_process = Image.open(curr_dir+"Ap_Spinous_Process.png").resize(img_size)
            ap_vertebra = Image.open(curr_dir+"Ap_Vertebra.png").resize(img_size)

            #LAT view
            curr_dir = data_dir + directory + folder + "/LAT/"
            lat = Image.open(curr_dir+"LAT.jpg").resize(img_size)
            lat = lat.convert('RGB')
            lat_anterior_vertebral_line = Image.open(curr_dir+"Lat_Anterior_Vertebral_Line.png").resize(img_size)
            lat_disk_height = Image.open(curr_dir+"Lat_Disk_Height.png").resize(img_size)
            lat_posterior_vertebral_line = Image.open(curr_dir+"Lat_Posterior_Vertebral_Line.png").resize(img_size)
            lat_spinous_process = Image.open(curr_dir+"Lat_Spinous_Process.png").resize(img_size)
            lat_vertebra = Image.open(curr_dir+"Lat_Vertebra.png").resize(img_size)

            curr_data = [i,ap, lat, ap_pedicle, ap_spinous_process, ap_vertebra, lat_anterior_vertebral_line, 
                         lat_disk_height, lat_posterior_vertebral_line, lat_spinous_process, lat_vertebra ]
            dataset.append(curr_data)
            
        except FileNotFoundError:
            print("Problem in folder: ", folder)
    return dataset

m_damaged = 328
m_normal = 350

#load damaged data
print("Loading Damaged Data....")   
damaged_data = load_data(data_dir, "Damaged/", 1, m_damaged)
#load normal data
print("Loading Normal Data....")
normal_data = load_data(data_dir, "Normal/", 1, m_normal)

m_damaged -= 5
m_normal -= 5

damaged_df = pd.DataFrame(damaged_data)
normal_df = pd.DataFrame(normal_data)

column_names = ['ID', 'AP', 'LAT', 'AP_Pedicle', 'AP_Spinous_Process', "AP_Vertebreta", "Lat_Anterior_Vertebral_Line", 'Lat_Disk_Height', 'Lat_Posterior_Vertebral_Line', 'Lat_Spinous_Process', "Lat_Vertebra"]
damaged_df.columns = column_names
normal_df.columns = column_names


# In[ ]:


import numpy as np
x_input=[]
y_input=[[], [], [], [], []]
m_damaged = damaged_df.values.shape[0]
m_normal = normal_df.values.shape[0]
for i in range(m_damaged):
    #print(damaged_df['AP_Spinous_Process'][i].mode)
    x_input.append(np.asarray(damaged_df['LAT'][i]))
    a1 = [np.asarray(damaged_df['Lat_Anterior_Vertebral_Line'][i], dtype=np.int).reshape(512, 256, 1), 
          np.asarray(damaged_df['Lat_Disk_Height'][i], dtype=np.int).reshape(512, 256, 1), 
          np.asarray(damaged_df['Lat_Posterior_Vertebral_Line'][i], dtype=np.int).reshape(512, 256, 1),
          np.asarray(damaged_df['Lat_Spinous_Process'][i], dtype=np.int).reshape(512, 256, 1),
          np.asarray(damaged_df['Lat_Vertebra'][i], dtype=np.int).reshape(512, 256, 1),
          ]
    for j in range(5):
      y_input[j].append(a1[j])

for i in range(m_normal):
    #print(damaged_df['AP_Spinous_Process'][i].mode)
    x_input.append(np.asarray(normal_df['LAT'][i]))
    a1 = [np.asarray(normal_df['Lat_Anterior_Vertebral_Line'][i], dtype=np.int).reshape(512, 256, 1), 
          np.asarray(normal_df['Lat_Disk_Height'][i], dtype=np.int).reshape(512, 256, 1), 
          np.asarray(normal_df['Lat_Posterior_Vertebral_Line'][i], dtype=np.int).reshape(512, 256, 1),
          np.asarray(normal_df['Lat_Spinous_Process'][i], dtype=np.int).reshape(512, 256, 1),
          np.asarray(normal_df['Lat_Vertebra'][i], dtype=np.int).reshape(512, 256, 1),
        ]
    for j in range(5):
      y_input[j].append(a1[j])


# In[6]:


print(x_input[0].shape)


# In[7]:


plt.imshow(x_input[0])


# In[8]:


plt.imshow(np.asarray(damaged_df['Lat_Anterior_Vertebral_Line'][0], dtype=np.int))


# In[ ]:


img_size = (512, 256)
augment_in = []
for i in range(len(x_input)):
  augment_in.append([x_input[i], y_input[0][i].reshape(img_size).astype(np.uint8)*255, y_input[1][i].reshape(img_size).astype(np.uint8)*255, y_input[2][i].reshape(img_size).astype(np.uint8)*255,
                     y_input[3][i].reshape(img_size).astype(np.uint8)*255, y_input[4][i].reshape(img_size).astype(np.uint8)*255])


# In[ ]:


import Augmentor
p = Augmentor.DataPipeline(augment_in)
p.random_distortion(0.3, 2, 2, 5);
p.skew(0.2);
p.rotate(1, max_left_rotation=5, max_right_rotation=5)
p.zoom_random(1, percentage_area=0.8)


# In[ ]:


def get_batch(gen_images):
  x_in = []
  y_in = [[], [], [], [], []]
  for i in range(len(gen_images)):
    #print(damaged_df['AP_Spinous_Process'][i].mode)
    x_in.append(gen_images[i][0])
    a1 = [(gen_images[i][1]/255).astype(np.int).reshape(512, 256, 1), 
          (gen_images[i][2]/255).astype(np.int).reshape(512, 256, 1), 
          (gen_images[i][3]/255).astype(np.int).reshape(512, 256, 1),
          (gen_images[i][4]/255).astype(np.int).reshape(512, 256, 1),
          (gen_images[i][5]/255).astype(np.int).reshape(512, 256, 1)]
    for i in range(5):
      y_in[i].append(a1[i])
  return x_in, y_in


# In[12]:


import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * (y_pred_f))
    smooth = 1
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

def dice_coef_mod(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * (y_pred_f-0.0002))
    smooth = 1
    return 2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth) 

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * (y_pred_f))
    smooth = 1
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth - intersection) 

def iou_loss(y_true, y_pred):
    return 1-iou(y_true, y_pred)

def unet(input_size = (512, 256, 3)):
    inputs = Input((input_size))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    d = Dropout(0.5)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(d)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    d = Dropout(0.5)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(d)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    d = Dropout(0.5)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(d)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)

    conv_mask1 = Conv2D(128, (1, 1), activation='relu', padding='same')(conv9)
    conv_mask2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv9)
    conv_mask3 = Conv2D(128, (1, 1), activation='relu', padding='same')(conv9)
    conv_mask4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv9)
    conv_mask5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv9)
    
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv_mask1)
    conv11 = Conv2D(1, (1, 1), activation='sigmoid')(conv_mask2)
    conv12 = Conv2D(1, (1, 1), activation='sigmoid')(conv_mask3)
    conv13 = Conv2D(1, (1, 1), activation='sigmoid')(conv_mask4)
    conv14 = Conv2D(1, (1, 1), activation='sigmoid')(conv_mask5)
    

    model = Model(inputs=[inputs], outputs=[conv10, conv11, conv12, conv13, conv14])

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef, iou])

    return model
model = unet()


# In[ ]:


g = p.generator(batch_size=700)
x, y = get_batch(next(g))
x_input.extend(x)
for i in range(5):
  y_input[i].extend(y[i])


# In[14]:



model.fit( np.array(x_input), 
          y_input, 
          batch_size= 4, epochs=30, verbose=1, callbacks=None, validation_split=0.2, validation_data=None, shuffle=True)


# In[ ]:


y_pred = model.predict(np.array(x_input))


# In[16]:


plt.imshow(y_pred[0][20, :, :, 0])


# In[17]:


plt.imshow(y_input[0][20][:, :, 0])


# In[18]:


# serialize model to JSON
model_json = model.to_json()
with open("model_lat.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_lat.h5")
print("Saved model to disk")


# In[ ]:




