import csv
import cv2
import numpy as np 
import sklearn
import os

# Load data
samples = []
with open('./mac-simulator/data-4-01-copy/driving_log.csv') as csvfile:
#with open('./udacity-data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

images = []
measurements = []
for line in samples:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    #print(filename)
    current_path = './mac-simulator/data-4-01-copy/IMG/' + filename
    #current_path = './udacity-data/IMG/' + filename
    #print(current_path)
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    #print(image.shape)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D 

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# Create generator to create augmented images
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                image.append(center_iamge)
                angles.append(center_angle)
                
            # Trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffule(X_train, y_train)

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320 # Trimmed image format

# Create nvidia's deep neural network model architecture 
model = Sequential()
# normalize image values between -1. : 1.
model.add(Lambda(lambda x: (x / 127.5) - 1., 
    input_shape=(row, col, ch),
    output_shape=(row, col, ch)))
'''
50 rows pixels from the top of the image
20 rows pixels from the bottom of the image
0 columns of pixels from the left of the image
0 columns of pixels from the right of the image
'''
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
# valid border mode shuold get rid of a couple each way, whereas same keeps
model.add(Convolution2D(24,5,5, subsample=(2, 2), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
model.add(Flatten()) 
model.add(Dropout(.5))
model.add(Activation('relu'))

model.add(Dense(120))
model.add(Activation('relu'))

model.add(Dense(84))
model.add(Activation('relu'))

model.add(Dense(1))

# comple with normal adam optimizer (loss .001) and return
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, batch_size = 512, validation_split=0.2, shuffle=True, nb_epoch=4)
model.save('model-data4-01-copy-nongen.h5')
print ('Model saved')
exit()
