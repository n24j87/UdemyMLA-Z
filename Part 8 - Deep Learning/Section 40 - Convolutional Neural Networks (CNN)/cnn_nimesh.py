# -*- coding: utf-8 -*-

#part - 1 Building the CNN

# importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

# Initialize CNN
classifier = Sequential()

# CNN model Steps - Convolution -> Max pooling -> Flattening -> Full connection

# Step 1 convolution
classifier.add(Conv2D(filters=16, kernel_size=3, padding='same', input_shape=(64, 64, 3), activation='relu'))

# Step - 2 pooling
classifier.add(MaxPool2D(pool_size=2))

classifier.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
classifier.add(MaxPool2D(pool_size=2))

# Step - 3 Flattening
classifier.add(Flatten())

# Step - 4 Dense Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.summary()

# Compile the model
classifier.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part - 2 Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000//32,
                         epochs = 20,
                         validation_data = test_set,
                         nb_val_samples = 2000, use_multiprocessing=True)