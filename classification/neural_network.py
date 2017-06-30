import os
import csv
import time
import keras
import random
import numpy as np
from scipy import misc
from typing import List, Tuple, Dict
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from preprocessing.images_reader import ImagesReader


def run(size: Tuple[int, int], dataset_dir: str):
    epochs = 100
    batch_size = 128
    classes_count = 257
    input_shape = size[0] * size[1] * 3

    (x_train, y_train) = __read_train_dataset__(size, dataset_dir)
    (x_test, y_test) = __read_test_dataset__(size, dataset_dir)

    y_train = keras.utils.to_categorical(y_train, classes_count)

    print('*****************************')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('*****************************')

    model = Sequential()

    model.add(Dense(2048, input_dim=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    for i in range(0, 2):
        model.add(Dense(1024))
        model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Dense(classes_count))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.002, patience=3, verbose=1, mode='auto')
    tensorBoard = keras.callbacks.TensorBoard(log_dir=os.path.join('./logs', str(time.time())))

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[tensorBoard])

    prediction_result = model.predict_classes(x_test) + 1

    print()
    print('Saving results')
    with open('nn_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['image', 'class'])
        for i in range(0, len(x_test)):
            writer.writerow([y_test[i], prediction_result[i]])


def __read_train_dataset__(size: Tuple[int, int], dataset_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    reader = ImagesReader(dataset_dir)
    images = reader.read_train_images()

    flatten_images = []
    for image_class in images:
        for image in images[image_class]:
            flatten_images.append((int(image_class), image.path))
    random.shuffle(flatten_images)

    images_count = len(flatten_images)

    x = np.empty(shape=(images_count, size[0] * size[1] * 3), dtype=np.float32)
    y = np.empty(shape=(images_count,), dtype=np.int16)

    for (i, (image_class, image_path)) in enumerate(flatten_images):
        x[i, :] = __get_np_image__(image_path)
        y[i] = image_class - 1

    return x, y


def __read_test_dataset__(size: Tuple[int, int], dataset_dir: str) -> Tuple[List[List[int]], List[str]]:
    reader = ImagesReader(dataset_dir)
    images = reader.read_test_images()
    random.shuffle(images)

    images_count = len(images)

    x = np.empty(shape=(images_count, size[0] * size[1] * 3), dtype=np.float32)
    y = [None] * images_count

    for i, image_metadata in enumerate(images):
        x[i, :] = __get_np_image__(image_metadata.path)
        y[i] = image_metadata.file_name

    return x, y


def __get_np_image__(image_path: str) -> np.ndarray:
    return misc.imread(image_path).flatten() / 255
