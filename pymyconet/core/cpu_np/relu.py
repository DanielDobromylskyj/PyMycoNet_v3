from buffer import DataBuffer
from ..generic.relu import ReLU as GenericReLU


class ReLU(GenericReLU):
    def __init__(self):
        super().__init__()

    def forward(self, input_values: DataBuffer) -> DataBuffer:
        raw_values = input_values.get_buffer()

        outputs = raw_values * (raw_values > 0)  # each value is clamped to 0 or greater

        return DataBuffer(outputs)

    def backward(self, input_buffer: DataBuffer, previous_error_buffer: DataBuffer, learning_rate: float) -> dict:
        next_error = previous_error_buffer.get_buffer() * (input_buffer.get_buffer() > 0)

        return {
            "weight_gradients": DataBuffer([]),
            "bias_gradients": DataBuffer([]),
            "next_errors": DataBuffer(next_error)
        }

    @staticmethod
    def load(zip_file, layer):
        return ReLU()