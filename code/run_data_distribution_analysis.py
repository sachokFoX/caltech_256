import time
import constants as cons
import matplotlib.pyplot as plt
from preprocessing.images_reader import ImagesReader

start_time = time.time()

print('reading images...')
reader = ImagesReader(cons.PREPROCESSED_DATASET_DIR)

train_images = reader.read_train_images()

classes = [None] * len(train_images)
samples = [None] * len(train_images)

for i, image_class in enumerate(train_images):
    classes[i] = image_class
    samples[i] = len(train_images[image_class])

plt.plot(classes, samples)
plt.show()

end_time = time.time()
print('done in {:.2f}s'.format(end_time - start_time))