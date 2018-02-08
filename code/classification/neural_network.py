import os
import csv
import time
import keras
import random
import numpy as np
from scipy import misc
from keras.models import Sequential
from typing import List, Tuple, Dict
from preprocessing.images_reader import ImagesReader
from preprocessing.image_metadata import ImageMetadata
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, GaussianDropout, Flatten, Activation, BatchNormalization


def run_NN(size: Tuple[int, int], dataset_dir: str):
    epochs = 60
    batch_size = 256
    classes_count = 256
    input_shape = size[0] * size[1] * 3

    (train, validation) = __split_train_validation_set__(dataset_dir)

    (x_train, y_train) = __read_train_dataset__(train, size)
    (x_validation, y_validation) = __read_train_dataset__(validation, size)
    (x_test, y_test) = __read_test_dataset__(size, dataset_dir)

    y_train = keras.utils.to_categorical(y_train, classes_count)
    y_validation = keras.utils.to_categorical(y_validation, classes_count)

    print('*****************************')
    print('x_train shape:', x_train.shape)
    print('x_validation shape:', x_validation.shape)
    print('x_test shape:', x_test.shape)
    print('*****************************')

    model = Sequential()

    model.add(Dense(input_shape, input_shape=(input_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(classes_count))
    # model.add(BatchNormalization())
    model.add(Activation('softmax'))

    print('Compilling NN model')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_acc', min_delta=0.001, patience=5, verbose=1, mode='auto')
    tensor_board = keras.callbacks.TensorBoard(
        log_dir=os.path.join('../logs', str(time.time())))

    model.fit(
        x_train,
        y_train,
        shuffle=True,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_validation, y_validation),
        callbacks=[early_stopping, tensor_board])

    prediction_result = model.predict_classes(x_test) + 1

    print()
    print('Saving results')
    if not os.path.exists('../output'):
        os.makedirs('../output')
    with open(os.path.join('../output', 'nn_results.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['image', 'class'])
        for i in range(0, len(x_test)):
            writer.writerow([y_test[i], prediction_result[i]])


def run_CNN(size: Tuple[int, int], dataset_dir: str):
    epochs = 100
    batch_size = 256
    classes_count = 256
    input_shape = (size[0], size[1], 3)

    (train, validation) = __split_train_validation_set__(dataset_dir)

    (x_train, y_train) = __read_train_dataset__(train, size)
    (x_validation, y_validation) = __read_train_dataset__(validation, size)
    (x_test, y_test) = __read_test_dataset__(size, dataset_dir)

    x_train = x_train.reshape(x_train.shape[0], size[0], size[1], 3)
    x_validation = x_validation.reshape(
        x_validation.shape[0], size[0], size[1], 3)
    x_test = x_test.reshape(x_test.shape[0], size[0], size[1], 3)

    y_train = keras.utils.to_categorical(y_train, classes_count)
    y_validation = keras.utils.to_categorical(y_validation, classes_count)

    print('*****************************')
    print('x_train shape:', x_train.shape)
    print('x_validation shape:', x_validation.shape)
    print('x_test shape:', x_test.shape)
    print('*****************************')

    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(0.0125)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(GaussianDropout(0.5))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.0125)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(GaussianDropout(0.5))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.0125)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(GaussianDropout(0.5))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(GaussianDropout(0.8))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(GaussianDropout(0.5))

    model.add(Dense(classes_count))
    model.add(Activation('softmax'))

    print('Compilling CNN model')
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_acc', min_delta=0.001, patience=5, verbose=1, mode='auto')
    tensor_board = keras.callbacks.TensorBoard(
        log_dir=os.path.join('../logs', str(time.time())))

    model.fit(
        x_train,
        y_train,
        shuffle=True,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_validation, y_validation),
        callbacks=[early_stopping, tensor_board])

    prediction_result = model.predict_classes(x_test) + 1

    print()
    print('Saving results')
    if not os.path.exists('../output'):
        os.makedirs('../output')
    with open(os.path.join('../output', 'nn_results.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['image', 'class'])
        for i in range(0, len(x_test)):
            writer.writerow([y_test[i], prediction_result[i]])


def __split_train_validation_set__(dataset_dir: str) -> Tuple[Dict[str, ImageMetadata], Dict[str, ImageMetadata]]:
    train_dataset = {}
    validation_dataset = {}

    reader = ImagesReader(dataset_dir)
    images = reader.read_train_images()

    for image_class in images:
        class_images = list(images[image_class])
        random.shuffle(class_images)

        train_samples = round(len(class_images) * 0.75)

        train_dataset[image_class] = class_images[:train_samples]
        validation_dataset[image_class] = class_images[train_samples:]

    return train_dataset, validation_dataset


def __read_train_dataset__(images: Dict[str, ImageMetadata], size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:

    flatten_images = []
    for image_class in images:
        for image in images[image_class]:
            flatten_images.append((int(image_class), image.path))
    random.shuffle(flatten_images)

    images_count = len(flatten_images)

    x_dataset = np.empty(
        shape=(images_count, size[0] * size[1] * 3), dtype=np.float32)
    y_dataset = np.empty(shape=(images_count,), dtype=np.int16)

    for (i, (image_class, image_path)) in enumerate(flatten_images):
        x_dataset[i, :] = __get_np_image__(image_path)
        y_dataset[i] = image_class - 1

    return x_dataset, y_dataset


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
