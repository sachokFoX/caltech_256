import os
import math
import random
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
from preprocessing.image_metadata import ImageMetadata


class PreProcessor:
    @staticmethod
    def preprocess(train_images: Dict[str, ImageMetadata], validate_images: List[ImageMetadata], size: Tuple[int, int], output_dir: str) -> None:

        train_dataset_dir = os.path.join(output_dir, 'train')
        test_dataset_dir = os.path.join(output_dir, 'test')
        validate_dataset_dir = os.path.join(output_dir, 'validate')

        if not os.path.exists(train_dataset_dir):
            os.makedirs(train_dataset_dir)

        if not os.path.exists(test_dataset_dir):
            os.makedirs(test_dataset_dir)

        if not os.path.exists(validate_dataset_dir):
            os.makedirs(validate_dataset_dir)

        datasets = PreProcessor.__split_dataset__(train_images)

        Parallel(n_jobs=2)(delayed(PreProcessor.__process_image__)
                           (i, size, train_dataset_dir) for i in datasets[0])
        Parallel(n_jobs=2)(delayed(PreProcessor.__process_image__)
                           (i, size, test_dataset_dir) for i in datasets[1])
        Parallel(n_jobs=2)(delayed(PreProcessor.__process_image__)
                           (i, size, validate_dataset_dir) for i in validate_images)

    @staticmethod
    def __split_dataset__(images: Dict[str, ImageMetadata]) -> Tuple[List[str], List[str]]:
        train_dataset = []
        test_dataset = []

        for k in images:
            class_images = list(images[k])
            random.shuffle(class_images)
            train_samples = round(len(class_images) * 0.8)
            train_dataset.extend(class_images[:train_samples])
            test_dataset.extend(class_images[train_samples:])

        return (train_dataset, test_dataset)

    @staticmethod
    def __process_image__(image: ImageMetadata, size: Tuple[int, int], output_dir: str) -> None:
        path = os.path.join(output_dir, PreProcessor.__get_file_name__(image.path))
        img = Image.open(image.path)

        img = img.resize(size, Image.BICUBIC)
        img = img.convert("RGB")

        # contrast
        enh = ImageEnhance.Contrast(img)
        enh.enhance(1.3)

        # sharpness
        enh = ImageEnhance.Sharpness(img)
        enh.enhance(1.3)

        img.save(path)

    @staticmethod
    def __get_file_name__(original_path: str) -> str:
        file_def = os.path.basename(original_path)
        class_def = os.path.dirname(original_path).split(os.sep)[-1].split('.')

        if file_def.find('train') > 0:
            return '{:s}_{:s}_{:s}'.format(class_def[0], class_def[1], file_def)
        else:
            return file_def
