import numpy as np

from .layer import Layer


class DenseLayer(Layer):
    __weights: np.ndarray
    __biases: np.ndarray

    def __init__(self, input_dim, output_dim, generate=True):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def initialize_core(self, core_class, generate=True):
        loaded = core_class(self.input_dim, self.output_dim, generate=generate)

        if not generate:  # Pass the values loaded into the generic layer to the core specific layer (Like GPU)
            loaded.set_values(self.__weights, self.__biases)

        return loaded

    def set_values(self, weights, biases):
        self.__weights = weights
        self.__biases = biases

    def get_weights(self) -> np.ndarray:
        raise NotImplementedError

    def get_biases(self) -> np.ndarray:
        raise NotImplementedError

    def compare_to(self, dense_layer):
        return {
            "weights": np.array_equal(self.get_weights(), dense_layer.get_weights()),
            "biases": np.array_equal(self.get_biases(), dense_layer.get_biases())
        }

    @staticmethod
    def deserialize(metadata: dict, serialized_data: bytes):
        values = Layer.deserialize_data(serialized_data, metadata['value_meta'])
        return values["weights"], values["biases"]

    @staticmethod
    def load(zip_file, layer):
        """ Loads and returns a dense layer / self """
        raise NotImplementedError

    def serialize(self):
        value_meta, blob = self.serialize_data({
            "weights": self.get_weights(),
            "biases": self.get_biases()
        })

        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "value_meta": value_meta
        }, blob

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(in={self.input_dim}, out={self.output_dim})"


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
