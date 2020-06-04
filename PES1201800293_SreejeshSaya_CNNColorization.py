import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import random
import matplotlib.pyplot as plt

datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

# Generate training data
batch_size = 10
def image_a_b_gen(batch_size, Xtrain):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

def trainNN(model):
    # Get images
        Xtrain = []
        for filename in os.listdir('Train'):
            Xtrain.append(img_to_array(load_img('Train/'+filename)))
        Xtrain = np.array(Xtrain, dtype=float)
        Xtrain = 1.0/255*Xtrain

        # Train model      
        tensorboard = TensorBoard(log_dir="Log/output", histogram_freq=1)
        history = model.fit(image_a_b_gen(batch_size, Xtrain),
                            callbacks=[tensorboard],
                            epochs=150,
                            steps_per_epoch=90)
   
def createModel():
    model = Sequential()
    model.add(InputLayer(input_shape=(256, 256, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.compile(optimizer='rmsprop', loss='mse')
    return model

weightFile = 'modelPerson.h5'
weightFile = os.path.abspath(weightFile)
model = createModel()
if os.path.isfile(weightFile): #file exists
    model.load_weights(weightFile)
    print('Model Successfully Loaded!')

model.summary()

trainNN(model)

X = []
for filename in os.listdir('Eval'):
    X.append(img_to_array(load_img('Eval/'+filename)))
X = np.array(X, dtype=float)
X = 1.0/255*X

Xtest = rgb2lab(1.0/255*X)[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0/255*X)[:,:,:,1:]
Ytest = Ytest / 128
print(model.evaluate(Xtest, Ytest, batch_size=10))

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("modelPerson.h5")

colorMe = []
for filename in os.listdir('Test'):
    colorMe.append(img_to_array(load_img('Test/'+filename)))
colorMe = np.array(colorMe, dtype=float)
colorMe = rgb2lab(1.0/255*colorMe)[:,:,:,0]
colorMe = colorMe.reshape(colorMe.shape+(1,))

# Test model
output = model.predict(colorMe)
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = colorMe[i][:,:,0]
    imsave("Result/resBW_"+str(i)+".png", lab2rgb(cur))
    cur[:,:,1:] = output[i]
    ab = np.zeros((256,256,3))
    ab[:,:,1:] = output[i]
#     imsave("Result/resAB+"+str(i)+".png", lab2rgb(ab))
    imsave("Result/res_"+str(i)+".png", lab2rgb(cur))

r = np.zeros((256, 256, 3))
r[:,:,1:] = output[0]
r[:,:,0] = colorMe[0][:,:,0]
plt.imshow(lab2rgb(r))