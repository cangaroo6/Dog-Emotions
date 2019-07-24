import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import No_Preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.utils import shuffle
from keras import Sequential
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Conv2D, Flatten, BatchNormalization, AveragePooling2D, Activation
from keras.optimizers import SGD

# size of images
img_width = 100
img_height = 100
# path for results
model_path="../models/"

# ---------------------------------------------------------------------------------------

def create_model():
  model = Sequential()

  model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(100, 100, 1)))
  model.add(BatchNormalization())
  model.add(Conv2D(filters=16, kernel_size=(7, 7), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
  model.add(Dropout(.5))

  model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same'))
  model.add(BatchNormalization())
  model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
  model.add(Dropout(.5))

  model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
  model.add(Dropout(.5))

  model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
  model.add(Dropout(.5))

  model.add(Flatten())
  model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.1)))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(4, activation='softmax'))

  sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

  return model

# ---------------------------------------------------------------------------------------

# read image csv
data = pd.read_csv('../prep_images_rotated.csv')
data = data.sample(frac=1).reset_index(drop=True)

# print some information
print('Different emotional states: ' + str(np.unique(data.emotion)))
print("Number of Examples for both states")
print(data.emotion.value_counts())

# pixel array is an ndarray containing all pixels of each image
helper = No_Preprocessing(img_width, img_height)
pixels = helper.extract_and_prepare_pixels(data[['pixels']])

x = pixels
x = x.reshape((-1, 100, 100, 1))
y = to_categorical(data.emotion.reset_index(drop=True))

# build train/test datasets
x_shuffle, y_shuffle = shuffle(x, y)  # Mix samples
x_train, x_test, y_train, y_test = train_test_split(x_shuffle, y_shuffle, test_size=0.1)  # split for train and test block

# train on all
# x_train = x_shuffle
# y_train = y_shuffle

# plot part of the set
fig, ax = plt.subplots(5, 5, figsize=(7, 7))
j = -1
k = 0
for i in range(0, 25):
  if i % 5 == 0:
    j += 1
    k = 0
  print(pixels[i])
  ax[j, k].imshow(pixels[i].reshape(100, 100), cmap="gray", interpolation='nearest')
  ax[j, k].set_title(y[i])
  ax[j, k].axis('off')
  k += 1
plt.savefig(model_path + 'exampleInput.png', dpi=300)

model = create_model()
model.summary()

history = model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=16)

scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# save trained classifier
model.save(model_path + 'classifier.h5')

# ---------------------------------------------------------------------------------------
# plotting

# summarize history for accuracy
plt.close()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model_path + 'accuracy.png', dpi=300)

# summarize history for loss
plt.close()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model_path + 'loss.png', dpi=300)



