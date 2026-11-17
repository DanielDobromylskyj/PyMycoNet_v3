from buffer import DataBuffer
from ..generic.relu import ReLU as GenericReLU


class ReLU(GenericReLU):
    def __init__(self):
        super().__init__()

    @staticmethod
    def load(zip_file, layer):
        return ReLU()