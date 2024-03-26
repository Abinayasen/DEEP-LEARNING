import numpy as np
import cv2   #convert images to arrays
import os
import random
import time
from tensorflow.keras.callbacks import TensorBoard

# Directory containing the images
DIRECTORY = r"D:\AI\DEEP LEARNING\keras classification\archive (1)\Dog and Cat .png"
# Categories
CATEGORIES = ["cat", "dog"]

img_size=120

data = []

for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr,(img_size,img_size))
        data.append([img_arr, label])
random.shuffle(data)
X=[]
y=[]

#iterate through the data list
for features, labels in data:
    X.append(features)
    y.append(labels)
X = np.array(X)
y=np.array(y)

#pickle.dump(X, open('X.pkl', 'wb'))       saves data
#pickle.dump(y, open('y.pkl', 'wb'))
#you can get these values by the code
#import pickle
#pickle.load(X, open('X.pkl', 'rb'))
NAME = f'cat-vs-dog-prediction-{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs\\{NAME}\\')

#feature scaling to speed the process
X= X/255
print(X.shape)

#small jist of processing
#input -> multiple filters(convolution layer) -> max pooling -> again covolution -> pool2 ->hidden layers ->output
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128,input_shape= X.shape[1:], activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation= 'softmax'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics =['accuracy'])
model.fit(X, y, epochs =5, validation_split =0.1, batch_size =32, callbacks =[tensorboard])
# tensorboard --logdir=logs/
#in terminal
#check for accuracy and validation

