import os
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
import time
from keras.models import load_model

DIRECTORY = "D:\AI\DEEP LEARNING\DATA\DATA_FLOWER"
CATEGORIES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
img_size=150
data=[]
for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr,(img_size,img_size))
        data.append([img_arr,label])
random.shuffle(data)
X=[]
y=[]
for features,labels in data:
    X.append(features)
    y.append(labels)
X=np.array(X)
y=np.array(y)
X=X/255
print(X.shape)

NAME = f'flower-classification-{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs\\{NAME}\\')

#model creation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Flatten,Dense,Dropout
from keras.optimizers import Adam
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val), batch_size=32, callbacks =[tensorboard])

model.save("D:\AI\DEEP LEARNING/flower_classification.keras")
