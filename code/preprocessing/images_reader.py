import os
from PIL import Image
from typing import Dict, List
from preprocessing.image_metadata import ImageMetadata


class ImagesReader:
    def __init__(self, base_path: str) -> None:
        self.__basePath = base_path

    def read_train_images(self) -> Dict[str, List[ImageMetadata]]:
        images = {}
        dataset_dir = os.path.join(self.__basePath, 'train')

        for root, dirs, files in os.walk(dataset_dir, topdown=False):
            if root not in [self.__basePath, dataset_dir]:
                files = [img for img in files if img.endswith('.jpg') or img.endswith('.JPEG')]
                class_id = self.__get_class_id__(root)
                images[class_id] = []

                for name in files:
                    image = self.__get_image_metadata__(os.path.join(root, name))
                    images[class_id].append(image)

        return images

    def read_test_images(self) -> List[ImageMetadata]:
        images = []
        dataset_dir = os.path.join(self.__basePath, 'test')

        files = [img for img in os.listdir(dataset_dir) if img.endswith('.jpg') or img.endswith('.JPEG')]

        for name in files:
            image = self.__get_image_metadata__(os.path.join(dataset_dir, name))
            images.append(image)

        return images

    @staticmethod
    def __get_image_metadata__(image_path: str) -> ImageMetadata:
        image = Image.open(image_path)
        return ImageMetadata(image.filename, (image.width, image.height), image.layers, image.mode)

    @staticmethod
    def __get_class_id__(dir_path: str) -> str:
        class_id = dir_path.split(os.sep)[-1].split('.')[0]
        return class_id
