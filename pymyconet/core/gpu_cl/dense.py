import pyopencl as cl
import numpy as np

from buffer import DataBuffer
from ..generic.dense import DenseLayer as GenericDenseLayer
from .cl_manager import Manager, create_or_get_manager, GPU_Buffer


mf = cl.mem_flags

class DenseLayer(GenericDenseLayer):
    def __init__(self, input_dim: int, output_dim: int, generate=True):
        super().__init__(input_dim, output_dim, generate=generate)
        self.manager: None | Manager = None

        self.__weight_buffer: None | cl.Buffer = None
        self.__bias_buffer: None | cl.Buffer = None

    def initialize_buffers(self) -> tuple[cl.Buffer, cl.Buffer]:
        if not isinstance(self.manager, Manager): raise

        weights_buffer = cl.Buffer(
            self.manager.context,
            mf.COPY_HOST_PTR | mf.READ_WRITE,
            hostbuf=self.get_weights()
        )

        biases_buffer = cl.Buffer(
            self.manager.context,
            mf.COPY_HOST_PTR | mf.READ_WRITE,
            hostbuf=self.get_biases()
        )

        return weights_buffer, biases_buffer

    def buffers_are_ready(self):
        return isinstance(self.__weight_buffer, cl.Buffer) and isinstance(self.__bias_buffer, cl.Buffer)

    def ready_buffers(self, is_preload: bool = False):
        if not isinstance(self.manager, Manager): raise

        if self.buffers_are_ready():
            return

        try:
            weights, biases = self.initialize_buffers()
            self.set_buffers(weights, biases)
        except MemoryError:
            if not is_preload:
                raise

    def release(self):
        if isinstance(self.__weight_buffer, cl.Buffer):
            self.__weight_buffer.release()

        if isinstance(self.__bias_buffer, cl.Buffer):
            self.__bias_buffer.release()

    def set_buffers(self, w, b):
        self.release()

        self.__weight_buffer = w
        self.__bias_buffer = b

    def on_hook(self):
        self.manager: Manager = create_or_get_manager(self.network)

        if not isinstance(self.manager, Manager): raise

        try:
            weights, biases = self.initialize_buffers()
            self.set_buffers(weights, biases)
        except MemoryError:
            pass

    def is_next(self):
        self.ready_buffers(is_preload=True)

    def forward(self, input_values: DataBuffer | GPU_Buffer) -> GPU_Buffer:
        self.ready_buffers(is_preload=False)

        if not isinstance(input_values, GPU_Buffer):
            if not isinstance(self.manager, Manager): raise
            input_values = GPU_Buffer.from_data_buffer(self.manager, input_values)

        # todo
