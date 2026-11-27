import numpy as np
import math

from .layer import Layer


class ConvolutedLayer(Layer):
    __weights: list[np.ndarray]
    __biases: list[np.ndarray]

    def __init__(self,input_width: int, input_height: int,
                 kernel_width: int, kernel_height: int,
                 filter_count: int,
                 step_x: int, step_y: int, generate=True):

        self.input_width = input_width
        self.input_height = input_height

        self.kernel_width = kernel_width
        self.kernel_height = kernel_height

        self.filter_count = filter_count

        self.step_x = step_x
        self.step_y = step_y

        self.__weights, self.__biases = [], []

        if generate:
            for _ in range(self.filter_count):
                w, b = generate_values(kernel_height * kernel_width, 1)

                self.__weights.append(w.reshape(kernel_height, kernel_width))
                self.__biases.append(b)

    def calculate_output_dims(self) -> tuple[int, int, int]:
        """ Returns: tuple: filter_count, (input_x-kernel_x) / step_x, (input_y-kernel_y) / step_y """
        return (
            self.filter_count,
            (self.input_height - self.kernel_height) // self.step_y + 1,
            (self.input_width - self.kernel_width) // self.step_x + 1
        )

    def initialize_core(self, core_class, generate=True):
        loaded = core_class(self.input_width, self.input_height, self.kernel_width, self.kernel_height, self.filter_count, self.step_x, self.step_y, generate=generate)

        if not generate:  # Pass the values loaded into the generic layer to the core specific layer (Like GPU)
            loaded.set_values(self.__weights, self.__biases)

        return loaded

    def set_values(self, weights, biases):
        self.__weights = weights
        self.__biases = biases

    def get_weights(self) -> list[np.ndarray] | np.ndarray:
        return self.__weights

    def get_biases(self) -> list[np.ndarray] | np.ndarray:
        return self.__biases


    @staticmethod
    def deserialize(metadata: dict, serialized_data: bytes):
        values = Layer.deserialize_data(serialized_data, metadata['value_meta'])
        return values["weights"], values["biases"]


    @staticmethod
    def load(zip_file, layer):
        """ Loads and returns a dense layer / self """
        blob = zip_file.read(layer['blob'])
        meta = layer['meta']

        new = ConvolutedLayer(
            meta['input_width'], meta["input_height"],
            meta['kernel_width'], meta['kernel_height'],
            meta['filter_count'],
            meta['step_x'], meta['step_y'],
            generate=False
        )

        values = new.deserialize(meta, blob)
        new.set_values(*values)

        return new


    def serialize(self):
        value_meta, blob = self.serialize_data({
            "weights": np.array(self.get_weights()),
            "biases":  np.array(self.get_biases())
        })

        return {
            "input_width": self.input_width,
            "input_height": self.input_height,
            "kernel_width": self.kernel_width,
            "kernel_height": self.kernel_height,
            "step_x": self.step_x,
            "step_y": self.step_y,
            "filter_count": self.filter_count,
            "value_meta": value_meta
        }, blob


    def __str__(self) -> str:
        return f"{self.__class__.__name__}(in={self.input_width}x{self.input_height})"


def generate_values(input_dim, output_dim):
    limit = np.sqrt(6 / (input_dim + output_dim))

    weights = np.random.uniform(
        -limit,
        limit,
        size=(output_dim, input_dim)
    )

    biases = np.zeros(output_dim, dtype=np.float32)

    if not isinstance(weights, np.ndarray): raise ValueError

    return np.astype(weights, np.float32), biases
