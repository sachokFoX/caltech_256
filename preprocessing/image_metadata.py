from common.size import Size


class ImageMetadata:
    def __init__(self, path: str, size: Size, layers: int, mode: str) -> None:
        self.path = path
        self.size = size
        self.layers = layers
        self.mode = mode

    def __str__(self):
        return '{:s} {:s} {:d} {:s}'.format(self.path, self.size, self.layers, self.mode)
