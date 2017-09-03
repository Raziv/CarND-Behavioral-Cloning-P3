import csv
import cv2
import numpy as np

lines = []
#with open('/Users/rajivkarki/Desktop/driving_log.csv') as csvfile:
with open('SDC_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


images = []
measurements = []
for line in lines:
    #print (line)
    source_path = line[0] #center camera
    #print (source_path)
    filename = source_path.split('/')[-1]
    # local computer path
    current_path = source_path
    # AWS path
    #current_path = '../data/IMG' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

# Keras only accespt numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)

# Regression network
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D

model = Sequential()
# Normalising and mean center
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
#Conlution layers
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
exit()
