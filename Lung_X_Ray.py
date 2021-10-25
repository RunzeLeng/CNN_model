import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import seaborn as sns 
plt.style.use('fivethirtyeight')
import pickle 
import os 
import numpy as np
import cv2
from os import listdir
from matplotlib import image
from PIL import Image
from numpy import array
from keras import backend as K
from keras.utils import np_utils
import keras.backend as tfback


##
def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus()
tf.config.list_logical_devices()


##change data format
##K.set_image_data_format('channels_first') ##(1,200,200)
K.set_image_data_format('channels_last') ##(200,200,1)


## load all images in a directory -- method 1
##loaded_images = list()
##for filename in listdir('/Users/RunzeLeng/Desktop/Project/BP-Lung-Xray/TypeA/'):
 # load image
## img_data = image.imread('/Users/RunzeLeng/Desktop/Project/BP-Lung-Xray/TypeA/' + filename)
 # store loaded image
## loaded_images.append(img_data)
## print('> loaded %s %s' % (filename, img_data.shape))
 
 
## load all images in a directory -- method 2
labels = ['NORMAL','TypeA', 'TypeB', 'TypeC']
img_size = 200
def get_training_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

dataimg = get_training_data('/Users/RunzeLeng/Desktop/Project/BP-Lung-Xray/')
dataimg.shape #(2389, 2) 每一组都少一个


##look of current data
arr=dataimg[0][0]
arr
arr.shape
arr=dataimg[0]
arr
arr.shape

img = Image.fromarray(arr[0])
img.show()


##formatting data
X = []
y = []

for feature, label in dataimg:
    X.append(feature)
    y.append(label)
    

## resize data for deep learning 
##X = np.array(X).reshape(-1, 1, img_size, img_size)##(1,200,200)
X = np.array(X).reshape(-1, img_size, img_size, 1)##(200,200,1)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=32)

X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

##look of current data
X_train.shape
X_test.shape
y_train.shape
y_test.shape
num_classes


##good for balancing out disproportions in the dataset
datagen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=60, 
        zoom_range = 0.1, 
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip=True,  
        vertical_flip=True)  
datagen.fit(X_train)



##model 2
def larger_model2():
    model2 = Sequential()
    model2.add(Conv2D(256, (3, 3), input_shape=(200, 200,1), padding='same'))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model2.add(BatchNormalization(axis=1))

    model2.add(Conv2D(64, (3, 3), padding='same'))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model2.add(BatchNormalization(axis=1))

    model2.add(Conv2D(16, (3, 3), padding='same'))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model2.add(BatchNormalization(axis=1))

    model2.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model2.add(Dropout(0.5))
    model2.add(Dense(64))
    model2.add(Activation('relu'))

    model2.add(Dropout(0.5))
    model2.add(Dense(4))
    model2.add(Activation('softmax'))

    adam = Adam(learning_rate=0.0001)
    model2.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['acc'])
    return model2

model2=larger_model2()
model2.summary()



## model 1
def larger_model():
    # create model
    model = Sequential()
   ## model.add(Conv2D(20, (5, 5), input_shape=(1, 200, 200), activation='relu'))
    model.add(Conv2D(20, (5, 5), input_shape=(200, 200, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(50, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(70, (7, 7), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = larger_model()
model.summary()


## early_stop
early_stop = EarlyStopping(patience=3, monitor='val_loss')


## train and test model 1
##model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=50 ,verbose=1)
model.fit(datagen.flow(X_train, y_train, batch_size=20), validation_data=(X_test, y_test), epochs=20,callbacks=[early_stop], verbose=1)

scores = model.evaluate(X_test, y_test, verbose=1)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
print("Accuracy: %.2f%%" % (scores[1]*100))



## train and test model 2
model2.fit(datagen.flow(X_train, y_train, batch_size=20), validation_data=(X_test, y_test), epochs=20,callbacks=[early_stop], verbose=1)



##model 1 threhold adjustment
pred = model.predict(X_train)
















































    
    
    
    
    
    