import os
import time
from termcolor import cprint
from preprocessing.images_reader import ImagesReader
from preprocessing.preprocessor import PreProcessor


def main():
    start_time = time.time()

    base_dir = '/Users/bohdan/work/uds-club'
    original_dataset_dir = os.path.join(base_dir, 'dataset_caltech_256/train')
    preprocessed_dataset_dir = os.path.join(base_dir, 'dataset_caltech_256_preprocessed')

    cprint('reading images...', 'green')
    reader = ImagesReader(original_dataset_dir)
    images = reader.read_images()

    cprint('preprocessing images...', 'green')
    PreProcessor.preprocess(images, preprocessed_dataset_dir)

    end_time = time.time()
    cprint('done in {:.2f}s'.format(end_time - start_time), 'green')


# main
main()
