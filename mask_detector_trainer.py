#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import os
import matplotlib.pyplot as plt
from imutils import paths

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[20]:


dataset=r'C:\Users\LENOVO\Desktop\Face-Mask-detector-master\project\dataset'
imagePath=list(paths.list_images(dataset))


# In[15]:


imagePath


# In[44]:


data=[]
labels=[]

for i in imagePath:
    label=i.split(os.path.sep)[-2]
    labels.append(label)
    image=load_img(i,target_size=(224,224))
    image=img_to_array(image)
    image=preprocess_input(image)
    data.append(image)


# In[23]:


labels


# In[32]:


data


# In[50]:


lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)


# In[53]:


data=np.array(data,dtype='float32')
labels=np.array(labels)


# In[54]:


labels


# In[55]:


train_X,test_X,train_Y,test_Y=train_test_split(data,labels,test_size=0.20,stratify=labels,random_state=10)


# In[56]:


data.shape


# In[57]:


train_X


# In[58]:


train_Y


# In[59]:


test_X


# In[60]:


test_Y


# In[61]:


train_Y.size


# In[62]:


train_X.shape


# In[63]:


test_Y.shape


# In[ ]:


aug=ImageDataGenerator(rota)

