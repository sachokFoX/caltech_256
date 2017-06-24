class Size:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def __format__(self, format_spec):
        return self.__str__()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '({:d}x{:d})'.format(self.width, self.height)
