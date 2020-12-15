import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

cnn = tf.keras.models.load_model('cnn_model.h5')

# loading the test image
test_image = image.load_img("../../data/dataset-cnn/single_prediction/cat_or_dog_1.jpg", target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)  # add an extra dimension corresponding to the batch

result = cnn.predict(test_image)

# encode 1 -> dog, 0 -> cat
# print(training_set.class_indices)

prediction = ''
if result[0][0] == 0:
  prediction = 'cat'
else:
  prediction = 'dog'

print(prediction)
