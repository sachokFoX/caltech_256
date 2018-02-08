import time
import constants as cons
from preprocessing.images_reader import ImagesReader
from preprocessing.preprocessor import PreProcessor

start_time = time.time()

print('reading images...')
reader = ImagesReader(cons.ORIGINAL_DATASET_DIR)

train_images = reader.read_train_images()
test_images = reader.read_test_images()

print('preprocessing images...')
preprocessor = PreProcessor(cons.PREPROCESSED_DATASET_DIR, True, 250)
preprocessor.preprocess(train_images, test_images, cons.SIZE)

end_time = time.time()
print('done in {:.2f}s'.format(end_time - start_time))
