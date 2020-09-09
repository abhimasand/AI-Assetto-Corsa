# from numpy.random import seed
# seed(123)
# from tensorflow import set_random_seed
# set_random_seed(123)


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
#from tensorflow.keras.layers.core import Lambda, RepeatVector, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
#from tensorflow.keras.layers.pooling import MaxPooling2D
from tensorflow.keras.layers import concatenate

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)
#import imutils


import cv2
from queue import Queue
from threading import Thread
from mss import mss

height, width = 1000, 1400
input_img = Input((height, width, 3), name='img')

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (input_img)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)

u5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c4)
u5 = concatenate([u5, c3])
c6 = Conv2D(32, (3, 3), activation='relu', padding='same') (u5)
c6 = Conv2D(32, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c2])
c7 = Conv2D(16, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(16, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c1])
c8 = Conv2D(8, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(8, (3, 3), activation='relu', padding='same') (c8)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c8)

model = Model(inputs=[input_img], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy') #, metrics=[mean_iou]) # The mean_iou metrics seens to leak train and test values...

model.load_weights('final-road-seg-model-v2.h5')


bounding_box = {'top': int(1440/2) - int(height/2), 'left': int(2560/2) - int(width/2), 'width': width, 'height': height}

sct = mss()



while True:
	frame = np.array(sct.grab(bounding_box))
	cv2.imshow("Frame", frame[:,:,:3])

	my_preds = model.predict(np.expand_dims(frame[:,:,:3], 0))
	my_preds = my_preds.flatten()
	my_preds = np.array([1 if i >= 0.5 else 0 for i in my_preds])
	my_preds = my_preds.reshape((height, width))*128/255.0
	
	my_preds = np.dstack([my_preds, my_preds, my_preds])
	# print (np.unique(my_preds))
	cv2.imshow("Pred", my_preds)
	# cv2.waitKey(1)

	if (cv2.waitKey(1) & 0xFF) == ord('q'):
		cv2.destroyAllWindows()
		break

# my_preds = model.predict(np.expand_dims(test_images[NUMBER], 0))
# my_preds = my_preds.flatten()
# my_preds = np.array([1 if i >= 0.5 else 0 for i in my_preds])
# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(my_preds.reshape(600, 800))
# ax[0].set_title('Prediction')
# ax[1].imshow(test_masks[NUMBER].reshape(600, 800))
# ax[1].set_title('Ground truth')
