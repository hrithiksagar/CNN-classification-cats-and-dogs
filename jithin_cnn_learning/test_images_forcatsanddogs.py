from keras.models import load_model
import numpy as np
from keras_preprocessing import image

model= load_model('learning_cats_and_dogs_classification_jithin.hdf5')
test_image= image.load_img('C:/Users/hrith/Downloads/cat-and-dog/testi/c.jpg',target_size=(32,32))
test_image= image.img_to_array(test_image)
test_image= np.expand_dims(test_image, axis=0)
prediction= model.predict(test_image)
if (prediction[0][0]==0):
    print('cat')
elif(prediction[0][0]==1):
    print('dog')

