import os
import Augmentor
from PIL import Image
from typing import Dict, List, Tuple
from preprocessing.image_metadata import ImageMetadata


class PreProcessor:
    def __init__(self, output_dir: str, is_augmentation_enabled=True, min_samples_per_class=100) -> None:
        self.__output_dir = output_dir
        self.__is_augmentation_enabled = is_augmentation_enabled
        self.__min_samples_per_class = min_samples_per_class

    def preprocess(self, train_images: Dict[str, ImageMetadata], test_images: List[ImageMetadata], size: Tuple[int, int]) -> None:

        test_dataset_dir = os.path.join(self.__output_dir, 'test')
        train_dataset_dir = os.path.join(self.__output_dir, 'train')

        self.__create_dir_if_not_exists(test_dataset_dir)
        self.__create_dir_if_not_exists(train_dataset_dir)

        print('processing test dataset...')
        for i in test_images:
            self.__process_image__(i, size, test_dataset_dir)

        print('processing train dataset...')
        for image_class in train_images:
            path = os.path.join(train_dataset_dir, image_class)
            for i in train_images[image_class]:
                self.__process_image__(i, size, path)

        if self.__is_augmentation_enabled:
            print('augmenting train dataset...')
            for image_class in train_images:
                if len(train_images[image_class]) < self.__min_samples_per_class:
                    print('augmenting class %s...' % image_class)
                    path = os.path.join(train_dataset_dir, image_class)
                    pipeline = Augmentor.Pipeline(source_directory=path, output_directory='')
                    pipeline.flip_left_right(probability=0.4)
                    pipeline.rotate(probability=0.7, max_left_rotation=5, max_right_rotation=5)
                    pipeline.sample(self.__min_samples_per_class - len(train_images[image_class]))

    # @staticmethod
    # def __split_train_validation_set__(images) -> ():
    #     train_dataset = {}
    #     validation_dataset = {}
    #
    #     for k in images:
    #         class_images = list(images[k])
    #         random.shuffle(class_images)
    #
    #         train_samples = round(len(class_images) * 0.8)
    #
    #         train_dataset[k] = class_images[:train_samples]
    #         validation_dataset[k] = class_images[train_samples:]
    #
    #     return train_dataset, validation_dataset

    @staticmethod
    def __process_image__(image: ImageMetadata, size: Tuple[int, int], output_dir: str) -> None:
        path = os.path.join(output_dir, image.file_name)
        PreProcessor.__create_dir_if_not_exists(output_dir)

        img = Image.open(image.path)
        img = img.resize(size, Image.BICUBIC)
        img = img.convert("RGB")

        img.save(path)

    @staticmethod
    def __create_dir_if_not_exists(dataset_dir: str):
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
