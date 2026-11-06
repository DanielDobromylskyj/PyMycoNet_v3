import numpy as np


class DataBuffer:
    def __init__(self, data=None, size=None):
        if data is None and size is None:
            raise ValueError("Either data or size must be provided")

        if data is not None:
            self.__values = np.array(data, dtype=np.float32)

        if size is not None:
            self.__values = np.zeros(size, dtype=np.float32)

    def get_buffer(self):
        return self.__values

    def set_buffer(self, buffer: np.ndarray):
        self.__values = buffer
