import csv
import cv2
import sklearn
import numpy as np

lines = []
#with open('/Users/rajivkarki/Desktop/driving_log.csv') as csvfile:
with open('dvl.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# images = []
# measurements = []
# for line in lines:
#     for i in range(3):
#         source_path = line[i]
#         filename = source_path.split('/')[-1]
#         # local computer path
#         current_path = source_path
#         # AWS path
#         #current_path = '../data/IMG' + filename
#         image = cv2.imread(current_path)
#         images.append(image)
#         measurement = float(line[3])
#         measurements.append(measurement)
#
# # Data Augmentation to remove left turn bias
# augmented_images, augmented_measurements =[], []
# for image, measurement in zip(images, measurements):
#     augmented_images.append(image)
#     augmented_measurements.append(measurement)
#     augmented_images.append(cv2.flip(image, 1))
#     augmented_measurements.append(measurement*-1.0)

# Keras only accepts numpy arrays
#X_train = np.array(augmented_images)
#y_train = np.array(augmented_measurements)

#--------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                print (batch_sample)
                print ('***********')
                print (batch_sample[0])
                exit()
                filename = batch_sample[0].split('/')[-1]
                #current_path = filename
                current_path = './SDC_data/IMG/' + filename
                #current_path = ../data/IMG' + filename
                #print (current_path)
                #print (batch_sample[0].split('/')[-1])
                #print (batch_sample[1])
                #print (batch_sample[3])
                exit()
                center_image = cv2.imread(current_path)

                center_angle = float(batch_sample[3])
                #print (center_image)
                #exit()
                images.append(center_image)
                angles.append(center_angle)

            #X_train = np.array(images)
            #print (X_train.shape)
            #exit()
            # Data Augmentation to remove left turn bias
            augmented_images, augmented_measurements =[], []
            for image, measurement in zip(images, angles):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement*-1.0)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            #print (X_train.shape)
            #exit()
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
#------------------------------------------------------------------

# Regression network
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D

model = Sequential()
# Normalising and mean center
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160,320,3)))
# Crop top 70% and bottom 25%
model.add(Cropping2D(cropping=((70,25),(0,0))))

# 2. NVIDIA architecture
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# 1. Simple Convolution layers
# model.add(Convolution2D(6,5,5,activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6,5,5,activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=3)
model.save('model.h5')
exit()
