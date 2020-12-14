import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # to preprocess the images

# Preprocessing the Training Set
# Image Data Augmentations

# Train Data gen object will apply all the transformation on the images of the dataset
# to prevent over-fitting
# preprocessing training dataset
train_datagen = ImageDataGenerator(
  rescale=1. / 255,  # Feature Scaling to each pixel
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
)

# importing the training dataset
training_set = train_datagen.flow_from_directory(
  '../../data/dataset-cnn/training_set',
  target_size=(64, 64),
  batch_size=32,
  class_mode='binary'  # cat or dog binary outcome
)

# preprocessing and importing the test set
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_set = test_datagen.flow_from_directory(
  '../../data/dataset-cnn/test_set',
  target_size=(64, 64),
  batch_size=32,
  class_mode='binary'
)

# Building the CNN
# Initialising the CNN
cnn = tf.keras.Sequential()

# Convolution Layers, filters -> number of filters in the convolutional layer
# our input shape is 64 by 64 because we imported in that size
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Pooling -> Max-Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# add a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening Layer
cnn.add(tf.keras.layers.Flatten())

# Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output layer - 1 Node required for binary classification
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training CNN, adam for stochastic gradient descent
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# loading the test image
test_image = image.load_img("../../data/dataset-cnn/single_prediction/cat_or_dog_1.jpg", target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)  # add an extra dimension corresponding to the batch

result = cnn.predict(test_image)

# encode 1 -> dog, 0 -> cat
print(training_set.class_indices)

prediction = ''
if result[0][0] == 0:
  prediction = 'cat'
else:
  prediction = 'dog'

print(prediction)
