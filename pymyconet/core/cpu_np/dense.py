import numpy as np

from ..generic.dense import DenseLayer as GenericDenseLayer

from pymyconet.buffer import DataBuffer

class DenseLayer(GenericDenseLayer):
    def __init__(self, input_dim: int, output_dim: int, generate=True):
        super().__init__(input_dim, output_dim, generate=generate)


    def forward(self, input_values: DataBuffer):
        raw_values = input_values.get_buffer()
        if not isinstance(raw_values, np.ndarray): raise TypeError("Input must be a numpy array")

        if raw_values.shape[0] != self.input_dim:
            raw_values = raw_values.flatten()

            if raw_values.shape[0] != self.input_dim:
                raise ValueError(f"Bad Input Shape! {raw_values.shape[0]} != {self.input_dim}")


        outputs = self.__weights @ raw_values + self.__biases

        return DataBuffer(outputs)

    def backward(self, input_buffer: DataBuffer, previous_error_buffer: DataBuffer, learning_rate: float):
        previous_errors = previous_error_buffer.get_buffer()
        inputs = input_buffer.get_buffer()

        if len(previous_errors.shape) != 1:
            previous_errors.flatten()

        if len(inputs.shape) != 1:
            inputs = inputs.flatten()

        bias_gradients = np.empty((self.output_dim,), dtype=np.float32)
        weight_gradients = np.empty((self.output_dim, self.input_dim), dtype=np.float32)
        next_gradients = np.zeros((self.input_dim,), dtype=np.float32)

        for output_index in range(self.output_dim):
            delta = previous_errors[output_index]

            for input_index in range(self.input_dim):
                weight_grad = delta * inputs[input_index] * learning_rate
                next_error_grad = delta * self.__weights[output_index, input_index]

                weight_gradients[output_index, input_index] = weight_grad
                next_gradients[input_index] += next_error_grad

            bias_gradients[output_index] = delta * learning_rate

        return {
            "weight_gradients": DataBuffer(weight_gradients),
            "bias_gradients": DataBuffer(bias_gradients),
            "next_errors": DataBuffer(next_gradients)
        }

    def get_weights(self):
        return self.__weights

    def get_biases(self):
        return self.__biases

    def apply_gradients(self, *args):
        weight_grad, bias_grad = args

        self.__weights += weight_grad
        self.__biases += bias_grad
