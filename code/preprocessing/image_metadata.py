import os
from typing import Tuple


class ImageMetadata:
    def __init__(self, path: str, size: Tuple[int, int], layers: int, mode: str) -> None:
        self.path = path
        self.file_name = self.__get_file_name__(path)
        self.size = size
        self.layers = layers
        self.mode = mode

    def __str__(self):
        return '{:s} ({:d}x{:d}) {:d} {:s}'.format(self.path, self.size[0], self.size[1], self.layers, self.mode)

    @staticmethod
    def __get_file_name__(original_path: str) -> str:
        file_def = os.path.basename(original_path)
        return file_def
