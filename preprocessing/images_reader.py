import os
from PIL import Image
from typing import Dict
from common.size import Size
from preprocessing.image_metadata import ImageMetadata


class ImagesReader:
    def __init__(self, base_path: str) -> None:
        self.__basePath = base_path

    def read_images(self) -> Dict[str, ImageMetadata]:
        images = {}

        for root, dirs, files in os.walk(self.__basePath, topdown=False):
            if root is not self.__basePath:
                class_id = self.__get_class_id__(root)
                images[class_id] = []

                for name in files:
                    if not name.startswith('.'):
                        image = Image.open(os.path.join(root, name))
                        images[class_id].append(ImageMetadata(image.filename, Size(image.width, image.height), image.layers, image.mode))

        return images

    @staticmethod
    def __get_class_id__(dir_path: str):
        class_id = dir_path.split(os.sep)[-1].split('.')[0]
        return class_id
