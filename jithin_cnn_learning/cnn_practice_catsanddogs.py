from keras.layers import Dense, Flatten , Convolution2D, MaxPooling2D
from keras import Sequential
from keras_preprocessing.image import ImageDataGenerator

classifier = Sequential()
classifier.add(Convolution2D(32,3,3, input_shape=(32,32,3),activation='relu'))#relu=rectifier linear fiuntion
classifier.add(MaxPooling2D(pool_size=(2,2)))#pooling layer for kernal
classifier.add(Flatten())#flatenning the inputs
classifier.add(Dense(64,activation='relu'))#creating ANN and flanetting outputs are now inputs for ANN
classifier.add(Dense(1,activation='sigmoid'))#says prob of what ans is
classifier.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')#adam=sgd,in we hace more than 2 inputs then loss = 'catogirical_crossentropy'
#layers are finished till here NN is done
#to increase accuracy add deepNN oincrease dense to 64 or 128 or anything up to you
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'C:/Users/hrith/Downloads/cat-and-dog/training_set/training_set',
        target_size=(32,32),
        batch_size=25,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'C:/Users/hrith/Downloads/cat-and-dog/test_set/test_set',
        target_size=(32, 32),
        batch_size=25,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=2000,
        epochs=10,
        validation_data=test_set,
        validation_steps=800)

classifier.save('learning_cats_and_dogs_classification_jithin.hdf5')
classifier.save_weights('myweights.h5')
print(train_set.class_indices)

