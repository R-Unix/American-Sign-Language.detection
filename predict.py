import tensorflow as tf 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.keras.callbacks import TensorBoard
import pickle
#import time
#from tensorflow.keras.callbacks import ModelCheckpoint 
import cv2

CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q' ,'R' , 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

model = tf.keras.models.load_model('/Users/ishan/Desktop/Deep Learning/Trained models/ASL_image_classifier')

img = "/Users/ishan/Desktop/dataset/ASL/alfa_test/P_test.jpg"

img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)

img = img.astype(float)

img = cv2.resize(img,(50,50))

img = np.reshape(img,(1,50,50,1))

pred = model.predict(img)

ans = CATEGORIES[np.argmax(pred)]

print(ans)