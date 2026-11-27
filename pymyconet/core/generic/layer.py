import io
import numpy as np

import inspect

class Layer:
    network: "network.Network"

    def initialize_core(self, core_class, generate=True):
        raise NotImplementedError(f"{self.__class__} -> {inspect.currentframe().f_code.co_name}(...)")

    def forward(self, input_values):
        raise NotImplementedError(f"{self.__class__} -> {inspect.currentframe().f_code.co_name}(...)")

    def backward(self, input_buffer, previous_error_buffer, learning_rate: float) -> dict:
        raise NotImplementedError(f"{self.__class__} -> {inspect.currentframe().f_code.co_name}(...)")

    def apply_gradients(self, *args):
        raise NotImplementedError(f"{self.__class__} -> {inspect.currentframe().f_code.co_name}(...)")

    @staticmethod
    def load(zip_file, layer):
        raise NotImplementedError(f"GenericLayer -> {inspect.currentframe().f_code.co_name}(...)")

    def serialize(self) -> tuple[dict, bytes]:
        raise NotImplementedError(f"{self.__class__} -> {inspect.currentframe().f_code.co_name}(...)")

    @staticmethod
    def deserialize(metadata: dict, serialized_data: bytes):
        raise NotImplementedError(f"GenericLayer -> {inspect.currentframe().f_code.co_name}(...)")

    @staticmethod
    def serialize_data(data: dict[str, np.ndarray]):
        meta = {}

        buffer = io.BytesIO()
        for key, value in data.items():
            value = value.astype(dtype=np.float32)
            byte_data = value.tobytes()
            meta[key] = (buffer.tell(), len(byte_data), value.shape)
            buffer.write(value.tobytes())

        return meta, buffer.getvalue()

    @staticmethod
    def deserialize_data(data, meta: dict[str, int]):
        return {
            key: np.frombuffer(
                data[pointer:pointer+length], dtype=np.float32
            ).reshape(shape)
            for key, (pointer, length, shape) in meta.items()
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"

    def hook(self, net):
        self.network = net

        if hasattr(self, "on_hook"):
            self.on_hook()


    def release(self):
        pass

    def is_next(self):
        pass