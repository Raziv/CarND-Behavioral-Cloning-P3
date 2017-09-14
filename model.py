import csv
import cv2
import random
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Regression network
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Convolution2D
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout

epoch_size = 5
batch_size = 32
csv_path = '../data1combo/driving_log.csv'
img_path = '../data1combo/IMG/'

# Randomly alter brightness of an image
def random_image_brightness(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    img[:,:,2] = img[:,:,2] * np.random.uniform()
    img[:,:,2][img[:,:,2]>255]  = 255
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    return img

# Convert image from BGR to RGB
def BGR_to_RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def RGB_to_YUV(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def process_batch(batch_samples):
    images = []
    angles = []

    for batch_sample in batch_samples:
        path_center = img_path + batch_sample[0].split('/')[-1]
        path_left = img_path + batch_sample[1].split('/')[-1]
        path_right = img_path + batch_sample[2].split('/')[-1]

        img_center = np.asarray(cv2.imread(path_center))
        img_left = np.asarray(cv2.imread(path_left))
        img_right = np.asarray(cv2.imread(path_right))

        correction = 0.25
        steering_center = float(batch_sample[3])
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        images.append(img_center)
        angles.append(steering_center)
        images.append(img_left)
        angles.append(steering_left)
        images.append(img_right)
        angles.append(steering_right)

    return images, angles

def augment_image(images, angles):
    augmented_images, augmented_measurements = [], []

    for image, angle in zip(images, angles):
        if (abs(angle) > 0.04):
            img1 = RGB_to_YUV(BGR_to_RGB(image))
            augmented_images.append(img1)
            augmented_measurements.append(angle)
            # Generate 5 new images per input image
            for i in range(5):
                if (angle < -0.3 or angle > 0.3):
                    # Image read from cv2 is in BGR format
                    img = BGR_to_RGB(image)
                    img = random_image_brightness(img)
                    img = RGB_to_YUV(img)
                    # Flip image around y-axis
                    img = cv2.flip(img, 1)
                    ang = (angle * -1.0)

                    augmented_images.append(img)
                    augmented_measurements.append(ang)

    return augmented_images, augmented_measurements

def generator(samples):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset+batch_size]
            images, angles = process_batch(batch_samples)
            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

# Read lines from csv file
def load_image_from_csv(csv_path):
    lines = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    return lines

# Split input data into train/valid samples
def split_input(lines):
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)

    return train_samples, validation_samples

def build_model():
    model = Sequential()
    # Normalising and mean center
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160,320,3)))
    # Crop top 70% and bottom 25%
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    # NVIDIA architecture
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    #model.summary()

    return model

# Compile and train the model using the generator function
def train_model(model, train_samples, validation_samples):
    model.compile(loss='mse', optimizer=Adam(lr=0.0001))
    model.fit_generator(generator(train_samples), samples_per_epoch=len(train_samples), validation_data=generator(validation_samples),
                        nb_val_samples=len(validation_samples), nb_epoch=epoch_size)
    model.save('model.h5')
    exit()

# Load train/valid samples, build and train the model
def main():
    lines = load_image_from_csv(csv_path)
    train_samples, validation_samples = split_input(lines)
    model = build_model()
    train_model(model, train_samples, validation_samples)

if __name__ == '__main__':
    main()
