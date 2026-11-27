import numpy as np

from ..generic.convoluted import ConvolutedLayer as GenericConvLayer

from pymyconet.buffer import DataBuffer

class ConvolutedLayer(GenericConvLayer):
    def __init__(self, input_width: int, input_height: int,
                 kernel_width: int, kernel_height: int,
                 filter_count: int,
                 step_x: int, step_y: int, generate=True):

        super().__init__(input_width, input_height, kernel_width, kernel_height, filter_count, step_x, step_y, generate=generate)

    def get_weights(self):
        return self.__weights

    def get_biases(self):
        return self.__biases

    def forward(self, input_values) -> DataBuffer:
        """
        Takes in a 2D array, and runs multiple filters over it, moving the kernel, a (X, Y) dense network over it,
        moving by (step_x, step_y) until the while image is covered.
        Returns {filter_count} smaller 2D concatenated arrays (3D for NumPy)

        Output size *should be*: (All divisions,  round up)
         ([input_x-kernel_x] / step_x) * ([input_y-kernel_y] / step_y) * filter_count

        :param input_values:
        :return: DataBuffer:
        """

        shape = self.calculate_output_dims()
        _, output_height, output_width = shape

        outputs = np.empty(shape, dtype=np.float32)

        buffer = input_values.get_buffer()
        weights = self.get_weights()
        biases = self.get_biases()

        for filter_index in range(self.filter_count):
            kernel = weights[filter_index]

            for x_offset in range(0, self.input_width - self.kernel_width + 1, self.step_x):
                for y_offset in range(0, self.input_height - self.kernel_height + 1, self.step_y):
                    section = buffer[
                        y_offset:y_offset + self.kernel_height,
                        x_offset:x_offset + self.kernel_width
                    ]

                    result = np.sum(section * kernel) + biases[filter_index][0]

                    output_x = x_offset // self.step_x
                    output_y = y_offset // self.step_y

                    outputs[filter_index, output_y, output_x] = result

        return DataBuffer(data=outputs)

    def backward(self, input_buffer, previous_error_buffer, learning_rate: float) -> dict:
        previous_errors: np.ndarray = previous_error_buffer.get_buffer()
        inputs = input_buffer.get_buffer()

        output_dims = self.calculate_output_dims()

        if previous_errors.shape != 3:
            previous_errors = previous_errors.reshape(output_dims)

        if inputs.shape != 2:
            inputs = inputs.reshape((self.input_height, self.input_width))

        weight_grads = np.zeros((self.filter_count, self.kernel_height, self.kernel_width))
        bais_grads = np.zeros((self.filter_count,))
        next_grads = np.zeros((self.input_height, self.input_width))

        for filter_index in range(self.filter_count):
            for x_offset in range(0, self.input_width - self.kernel_width + 1, self.step_x):
                for y_offset in range(0, self.input_height - self.kernel_height + 1, self.step_y):
                    delta = previous_errors[filter_index, y_offset//self.step_y, x_offset//self.step_x]
                    bais_grads[filter_index] += delta * learning_rate

                    for kernel_x in range(self.kernel_width):
                        for kernel_y in range(self.kernel_height):
                            weight_grad = delta * inputs[y_offset + kernel_y, x_offset + kernel_x] * learning_rate
                            next_error_grad = delta * self.__weights[filter_index][kernel_y, kernel_x]

                            weight_grads[filter_index, kernel_y, kernel_x] += weight_grad
                            next_grads[y_offset + kernel_y, x_offset + kernel_x] += next_error_grad

        return {
            "weight_gradients": DataBuffer(weight_grads),
            "bias_gradients": DataBuffer(bais_grads),
            "next_errors": DataBuffer(next_grads)
        }

    def apply_gradients(self, *args):
        weight_grad, bias_grad = args

        self.__weights += weight_grad
        self.__biases += bias_grad
