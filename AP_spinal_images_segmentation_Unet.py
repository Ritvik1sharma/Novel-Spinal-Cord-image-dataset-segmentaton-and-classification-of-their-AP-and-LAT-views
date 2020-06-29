#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install Augmentor')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


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

m_damaged -=5
m_normal -=5

damaged_df = pd.DataFrame(damaged_data)
normal_df = pd.DataFrame(normal_data)

column_names = ['ID', 'AP', 'LAT', 'AP_Pedicle', 'AP_Spinous_Process', "AP_Vertebreta", "LAT_Anterior_Vert_Line", 'LAT_disk_height', 'LAT_Posterior_Vert_Line', 'LAT_Spinal_Process', "LAT_Vertebra"]
damaged_df.columns = column_names
normal_df.columns = column_names


# In[ ]:


img_size = (512, 256)


# In[ ]:


import numpy as np
x_input=[]
y_input=[[], [], []]
m_damaged = damaged_df.values.shape[0]
m_normal = normal_df.values.shape[0]

for i in range(m_damaged):
    #print(damaged_df['AP_Spinous_Process'][i].mode)
    x_input.append(np.asarray(damaged_df['AP'][i]))
    a1 = [np.asarray(damaged_df['AP_Pedicle'][i], dtype=np.int).reshape(512, 256, 1), 
          np.asarray(damaged_df['AP_Spinous_Process'][i], dtype=np.int).reshape(512, 256, 1), 
          np.asarray(damaged_df['AP_Vertebreta'][i], dtype=np.int).reshape(512, 256, 1)]
    for i in range(3):
      y_input[i].append(a1[i])

for i in range(m_normal):
    #print(damaged_df['AP_Spinous_Process'][i].mode)
    x_input.append(np.asarray(normal_df['AP'][i]))
    a1 = [np.asarray(normal_df['AP_Pedicle'][i], dtype=np.int).reshape(512, 256, 1), 
          np.asarray(normal_df['AP_Spinous_Process'][i], dtype=np.int).reshape(512, 256, 1), 
          np.asarray(normal_df['AP_Vertebreta'][i], dtype=np.int).reshape(512, 256, 1)]
    for i in range(3):
      y_input[i].append(a1[i])


# In[ ]:


plt.imshow(x_input[0])


# In[ ]:


plt.imshow(np.asarray(damaged_df['AP_Pedicle'][0], dtype=np.int))


# In[ ]:





# In[ ]:


augment_in = []
for i in range(len(x_input)):
  augment_in.append([x_input[i], y_input[0][i].reshape(img_size).astype(np.uint8)*255, y_input[1][i].reshape(img_size).astype(np.uint8)*255, y_input[2][i].reshape(img_size).astype(np.uint8)*255])


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
  y_in = [[], [], []]
  for i in range(len(gen_images)):
    #print(damaged_df['AP_Spinous_Process'][i].mode)
    x_in.append(gen_images[i][0])
    a1 = [gen_images[i][1].astype(np.int).reshape(512, 256, 1)/255, 
          gen_images[i][2].astype(np.int).reshape(512, 256, 1)/255, 
          gen_images[i][3].astype(np.int).reshape(512, 256, 1)/255]
    for i in range(3):
      y_in[i].append(a1[i])
  return x_in, y_in


# In[ ]:


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
    return 2*(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) ) 

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

    conv_mask1 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv9)
    d1 = Dropout(0.2)(conv_mask1)
    conv_mask2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv9)
    d2 = Dropout(0.5)(conv_mask2)
    conv_mask3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv9)
    d3 = Dropout(0.1)(conv_mask3)
    
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(d1)
    conv11 = Conv2D(1, (1, 1), activation='sigmoid')(d2)
    conv12 = Conv2D(1, (1, 1), activation='sigmoid')(d3)

    model = Model(inputs=[inputs], outputs=[conv10, conv11, conv12])

    model.compile(optimizer=Adam(lr=1e-4), loss=iou_loss, metrics=[dice_coef, iou])

    return model
model = unet()


# In[ ]:


g = p.generator(batch_size=1024)
x, y = get_batch(next(g))
x_input.extend(x)
for i in range(3):
  y_input[i].extend(y[i])


# In[ ]:





# In[14]:



model.fit( np.array(x_input), 
          y_input, 
          batch_size= 4, epochs=45, verbose=1, callbacks=None, validation_split=0.2, validation_data=None, shuffle=True)


# In[15]:


plt.imshow(x[0])


# In[16]:



plt.imshow(y[0][0].reshape(512, 256))


# In[ ]:


y_pred = model.predict(np.array(x_input))


# In[18]:


print(y_pred[0].shape)
print(y_input[0][0].shape)


# In[19]:


plt.imshow(y_pred[1][40, :, :, 0])


# In[20]:


plt.imshow(y_input[1][40][:, :, 0])


# In[21]:


def dice_calc(y_true, y_pred):
    y_true_f = (y_true).flatten()
    y_pred_f = (y_pred).flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 1
    return (2. * intersection + smooth) / (np.sum(y_true_f)+np.sum(y_pred_f))

print(dice_calc(y_input[0][40][:, :, 0], y_pred[0][40, :, :, 0]))


# In[22]:


# serialize model to JSON
model_json = model.to_json()
with open("model_ap.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_ap.h5")
print("Saved model to disk")


# In[24]:


get_ipython().system('pip install shap')


# In[27]:


print(model.output)


# In[33]:


import shap
import numpy as np

x_input = np.array(x_input)
# select a set of background examples to take an expectation over
background = x_input[np.random.choice(x_input.shape[0], 100, replace=False)]


new_model = Model(inputs=model.input, outputs=model.output[0])
print(new_model.output_shape)
# explain predictions of the model on three images
e = shap.DeepExplainer(new_model, background)
# ...or pass tensors directly
# e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
shap_values = e.shap_values(x_input[1:5])

