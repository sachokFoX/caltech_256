import os
import random
from PIL import Image
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
from preprocessing.image_metadata import ImageMetadata


class PreProcessor:
    @staticmethod
    def preprocess(images: Dict[str, ImageMetadata], output_dir: str) -> None:

        train_dataset_dir = os.path.join(output_dir, 'train')
        test_dataset_dir = os.path.join(output_dir, 'test')

        if not os.path.exists(train_dataset_dir):
            os.makedirs(train_dataset_dir)

        if not os.path.exists(test_dataset_dir):
            os.makedirs(test_dataset_dir)

        datasets = PreProcessor.__split_dataset__(images)

        Parallel(n_jobs=2)(delayed(PreProcessor.__process_image__)
                           (i, train_dataset_dir) for i in datasets[0])
        Parallel(n_jobs=2)(delayed(PreProcessor.__process_image__)
                           (i, test_dataset_dir) for i in datasets[1])

    @staticmethod
    def __split_dataset__(images: Dict[str, ImageMetadata]) -> Tuple[List[str], List[str]]:
        train_dataset = []
        test_dataset = []

        for k in images:
            class_images = list(images[k])
            random.shuffle(class_images)
            train_samples = round(len(class_images) * 0.6)
            train_dataset.extend(class_images[:train_samples])
            test_dataset.extend(class_images[train_samples:])

        return (train_dataset, test_dataset)

    @staticmethod
    def __process_image__(image: ImageMetadata, output_dir: str) -> None:
        path = os.path.join(output_dir, PreProcessor.__get_file_name__(image.path))
        img = Image.open(image.path)
        img = img.resize((64, 64))
        img = img.convert("RGB")
        img.save(path)

    @staticmethod
    def __get_file_name__(original_path: str) -> str:
        file_def = os.path.basename(original_path)
        class_def = os.path.dirname(original_path).split(os.sep)[-1].split('.')
        return '{:s}_{:s}_{:s}'.format(class_def[0], class_def[1], file_def)
