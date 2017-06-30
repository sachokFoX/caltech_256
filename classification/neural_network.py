import os
import csv
import time
import keras
import random
import numpy as np
from scipy import misc
from typing import List,Tuple
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

def run(size: Tuple[int, int], dataset_dir: str):
    epochs = 30
    batch_size = 128
    classes_count = 257
    input_shape = size[0] * size[1] * 3

    (x_train, y_train) = __read_dataset__(size, os.path.join(dataset_dir, 'train'))
    (x_test, y_test) = __read_dataset__(size, os.path.join(dataset_dir, 'test'))
    (x_validation, y_validation) = __read_validation_dataset__(size, os.path.join(dataset_dir, 'validate'))

    y_train = keras.utils.to_categorical(y_train, classes_count)
    y_test = keras.utils.to_categorical(y_test, classes_count)

    print('*****************************')
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)
    print('x_validation shape:', x_validation.shape)
    print('*****************************')

    model = Sequential()

    model.add(Dense(2048, input_dim=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    for i in range(0, 5):
        model.add(Dense(512))
        model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(classes_count))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.02, patience=2, verbose=1, mode='auto')
    tensorBoard = keras.callbacks.TensorBoard(log_dir=os.path.join('./logs', str(time.time())), write_graph=True, write_images=True)

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[earlyStopping, tensorBoard])

    # test_result = model.evaluate(x_test, y_test, batch_size=batch_size)

    prediction_result = model.predict_classes(x_validation) + 1

    # print()
    # print('Test score:', test_result[0])
    # print('Test accuracy:', test_result[1])

    print('Saving results')
    with open('nn_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['image', 'class'])
        for i in range(0, len(x_validation)):
            writer.writerow([y_validation[i], prediction_result[i]])

def __read_dataset__(size: Tuple[int, int], dataset_dir: str) -> Tuple[List[List[int]], List[int]]:
    images = [img for img in os.listdir(dataset_dir) if img.endswith('.jpg')]
    random.shuffle(images)
    images_count = len(images)

    x = np.empty(shape=(images_count, size[0] * size[1] * 3), dtype=np.float32)
    y = np.empty(shape=(images_count, ), dtype=np.int16)

    for i, image in enumerate(images):
        x[i, :] = __get_np_image__(os.path.join(dataset_dir, image))
        y[i] = __get_class_id__(image)
        # print(image)

    return (x, y)

def __read_validation_dataset__(size: Tuple[int, int], dataset_dir: str) -> Tuple[List[List[int]], List[str]]:
    images = [img for img in os.listdir(dataset_dir) if img.endswith('.jpg')]
    images_count = len(images)

    x = np.empty(shape=(images_count, size[0] * size[1] * 3), dtype=np.float32)
    y = []

    for i, image in enumerate(images):
        x[i, :] = __get_np_image__(os.path.join(dataset_dir, image))
        y.append(image)

    return (x, y)

def __get_np_image__(image_path: str) -> np.ndarray:
    return misc.imread(image_path).flatten() / 255

def __get_class_id__(image_name: str) -> str:
    return int(image_name.split('_')[0]) - 1
