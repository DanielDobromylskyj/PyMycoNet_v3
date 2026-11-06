
from .layer import Layer

class ActivationLayer(Layer):
    def __init__(self):
        pass

    def initialize_core(self, core_class, generate=True):
        return core_class()

    def apply_gradients(self, *args):
        return

    def serialize(self):
        return {}, b""





