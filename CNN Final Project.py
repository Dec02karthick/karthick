#!/usr/bin/env python
# coding: utf-8

# # Sports_Image_Classification

# # Import Libraries

# In[3]:


pip install keras==2.15.0


# In[4]:


pip install --upgrade tensorflow


# In[5]:


import tensorflow as tf
print(tf.__version__)


# In[ ]:


get_ipython().system('pip uninstall tensorflow')
get_ipython().system('pip install tensorflow')


# In[ ]:


get_ipython().system('pip install opencv-python')


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import warnings
warnings.simplefilter('ignore')

from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications
import os
import glob
import cv2


# In[3]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, Sequential


# In[4]:


train=glob.glob("C:\\Users\\User\\Downloads\\Dataset CNN Project\\train")


# In[5]:


train


# In[6]:


glob.glob('C:\\Users\\User\\Downloads\\Dataset CNN Project\\train\\sidecar racing\*')


# In[7]:


img = cv2.imread('C:\\Users\\User\\Downloads\\Dataset CNN Project\\train\\sidecar racing\\033.jpg')
print(img.shape)
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img1)


# In[8]:


train_class=os.listdir('C:\\Users\\User\\Downloads\\Dataset CNN Project\\train\\')
train_class


# In[9]:


count_dict1 = {}
img_dict1 = {}

# Loop through classes
for cls in train_class:  # Assuming class_names contains the list of dog classes
    image_path = glob.glob(f'C:\\Users\\User\\Downloads\\Dataset CNN Project\\train\\{cls}/*')
    count_dict1[cls] = len(image_path)

    if image_path:  # Check if image_path is not empty
        img_dict1[cls] = tf.keras.utils.load_img(random.choice(image_path))
count_dict1


# In[10]:


df1 = pd.DataFrame(data={'label':count_dict1.keys(),'count':count_dict1.values()})


# In[11]:


df1


# In[12]:


plt.figure(figsize=(20,8))
sns.barplot(x='label',y='count',data=df1)
plt.xticks(rotation=90)
plt.show()


# In[13]:


import math

num_items = len(img_dict1)
num_cols = 4
num_rows = math.ceil(num_items / num_cols)

plt.figure(figsize=(20, 5 * num_rows))  # Adjust the figure size based on the number of rows

for id, (label, img) in enumerate(img_dict1.items()):
    plt.subplot(num_rows, num_cols, id + 1)
    plt.imshow(img)
    plt.title(f"{label} {img.size}")
    plt.axis('off')


# # Test Data

# In[14]:


test_dir=os.listdir("C:\\Users\\User\\Downloads\\Dataset CNN Project\\test")


# In[15]:


test_dir


# In[16]:


img_dict={}
count_dict={}
for cls in test_dir:
    img_path=glob.glob(f'C:\\Users\\User\\Downloads\\Dataset CNN Project\\test/{cls}/*')
    count_dict[cls]=len(img_path)
    if img_path:
        img_dict[cls]=tf.keras.utils.load_img(random.choice(img_path))
count_dict 


# In[17]:


num_items=len(img_dict)
num_cols=4
num_rows=math.ceil(num_items/num_cols)
plt.figure(figsize=(20, 5* num_rows))
for id ,(label,img) in enumerate (img_dict.items()):
    plt.subplot(num_rows,num_cols, id + 1)
    plt.imshow(img)
    plt.title(f'{label} {img.size}')
    plt.axis('off')


# # Data Preprocessing

# In[18]:


train_data=tf.keras.utils.image_dataset_from_directory('C:\\Users\\User\\Downloads\\Dataset CNN Project\\train',label_mode='categorical',shuffle=False)
test_data=tf.keras.utils.image_dataset_from_directory('C:\\Users\\User\\Downloads\\Dataset CNN Project\\test',shuffle=False,label_mode='categorical')
validation_data=tf.keras.utils.image_dataset_from_directory('C:\\Users\\User\\Downloads\\Dataset CNN Project\\valid',label_mode='categorical',shuffle=False)


# In[19]:


width = 224
height = 224
channels = 3

data_preprocessing = tf.keras.Sequential([
    tf.keras.layers.Resizing(height, width),
    tf.keras.layers.Rescaling(1.0 / 255),

])


# In[20]:


train_ds=train_data.map(lambda x,y:(data_preprocessing(x),y))
test_ds=test_data.map(lambda x,y:(data_preprocessing(x),y))
valid_ds=validation_data.map(lambda x,y:(data_preprocessing(x),y))


# In[21]:


train_ds


# # Custom Model

# In[22]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Initialize the Sequential model
model = Sequential()

# Convolutional layers with MaxPooling
model.add(Conv2D(input_shape=(224,224,3), filters=32, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# Flatten the image into a 1D array using the Flatten layer
model.add(Flatten())

# Fully connected layers
model.add(Dense(units=256, activation="relu"))

# Output layer with 100 units and softmax activation for multi-class classification
model.add(Dense(units=100, activation="softmax"))


# In[23]:


# Display the model summary
model.summary()


# In[24]:


model.compile(optimizer='adam',loss='categorical_crossentropy'
                 ,metrics=['accuracy','Precision','Recall'])


# In[25]:


import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# # Data Augmentation

# In[26]:


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


# In[27]:


batch_size = 40
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator()
test_datagen = test_datagen.flow_from_directory('C:\\Users\\User\\Downloads\\Dataset CNN Project\\test',
                                      class_mode = "categorical",
                                      target_size = (224, 224),
                                      batch_size = batch_size,
                                      shuffle = False,
                                      seed = 42)

# batches of augmented image data
train_generator = train_datagen.flow_from_directory('C:\\Users\\User\\Downloads\\Dataset CNN Project\\train',
                   # this is the target directory
                    target_size=(224, 224),  # all images will be resized 
                    batch_size=batch_size,
                    class_mode='categorical',
shuffle = True,  seed = 42)  
# this is a similar generator, for validation data


# In[28]:


# Assuming validation_generator is your validation data generator
batch_images, batch_labels = next(train_generator)

# Print the shape of the batch
print("Batch images shape:", batch_images.shape)
print("Batch labels shape:", batch_labels.shape)


# In[29]:


val_datagen = ImageDataGenerator()
validation_generator = val_datagen.flow_from_directory(
        'C:\\Users\\User\\Downloads\\Dataset CNN Project\\valid',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle = False,seed = 42 )


# In[30]:


# Assuming validation_generator is your validation data generator
batch_images, batch_labels = next(validation_generator)

# Print the shape of the batch
print("Batch images shape:", batch_images.shape)
print("Batch labels shape:", batch_labels.shape)


# In[31]:


history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=500 // batch_size,
    callbacks=[early_stopping]
)


# In[32]:


import matplotlib.pyplot as plt
plt.plot(history.history["accuracy"])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()


# # Xception Model

# In[33]:


# Import libaries
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.applications.xception import Xception


# In[34]:


base_model_xception = Xception(weights='imagenet', include_top=False, input_shape=(224,224,3))


# In[35]:


# Create a Sequential model
model_xception = Sequential()

# Adding the Xception base model
model_xception.add(base_model_xception)

# Adding Global Average Pooling 2D layer
model_xception.add(GlobalAveragePooling2D())

# Adding Dropout layer with a dropout rate of 0.5
model_xception.add(Dropout(0.25))

# Adding Dropout layer with a dropout rate of 0.3
# model_xception.add(Dropout(0.3))

# Adding the final Dense layer with 100 units and 'softmax' activation
model_xception.add(Dense(100, activation='softmax'))


# In[36]:


# Define RMSprop optimizer with specific parameters
optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-07)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7)


# In[37]:


model_xception.compile(optimizer='adam',loss='categorical_crossentropy'
                 ,metrics=['accuracy','Precision','Recall'])


# In[38]:


model_xception.summary()


# In[39]:


from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# In[ ]:


history2=model_xception.fit(
        train_generator,
        epochs=6,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr])


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history2.history["accuracy"])
plt.plot(history2.history['val_accuracy'])
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()

