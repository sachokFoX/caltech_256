import os
import time
from termcolor import cprint
from preprocessing.images_reader import ImagesReader
from preprocessing.preprocessor import PreProcessor
import classification.neural_network as nn

start_time = time.time()

size = (32, 32)
base_dir = '/Users/bohdan/work/uds-club'
original_dataset_dir = os.path.join(base_dir, 'dataset_caltech_256')
preprocessed_dataset_dir = os.path.join(base_dir, 'dataset_caltech_256_preprocessed')

# cprint('reading images...', 'green')
# reader = ImagesReader(original_dataset_dir)
#
# train_images = reader.read_train_images()
# test_images = reader.read_test_images()
#
# cprint('preprocessing images...', 'green')
# preprocessor = PreProcessor(preprocessed_dataset_dir)
# preprocessor.preprocess(train_images, test_images, size)

nn.run(size, preprocessed_dataset_dir)

end_time = time.time()
cprint('done in {:.2f}s'.format(end_time - start_time), 'green')
