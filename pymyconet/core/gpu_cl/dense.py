import pyopencl as cl
import numpy as np

from buffer import DataBuffer
from ..generic.dense import DenseLayer as GenericDenseLayer
from .cl_manager import Manager, BufferManager, create_or_get_manager, GPU_Buffer


mf = cl.mem_flags

def print_cl_buffer(queue, buf: None | cl.Buffer):
    if buf is None:
        print("Buffer Is NoneType!")
        return

    data = np.empty(buf.size // 4, dtype=np.float32)
    cl.enqueue_copy(queue, data, buf).wait()
    print("NaNs:", np.isnan(data).sum(), data)


class DenseLayer(GenericDenseLayer, BufferManager):
    def __init__(self, input_dim: int, output_dim: int, generate=True):
        super().__init__(input_dim, output_dim, generate=generate)
        self.manager: None | Manager = None

        self.__forward_kernel = None
        self.__reduce_kernel = None
        self.__backward_kernel = None

    def load_kernels(self):
        if not isinstance(self.manager, Manager):
            raise ReferenceError("Couldn't find networks manager. Did it get hooked?")

        self.__forward_kernel, self.__reduce_kernel, self.__backward_kernel = self.manager.get_kernels(
            "gpu_cl", "kernels/dense.c", ["forward", "reduce", "backward"]
        )


    def forward(self, input_values: DataBuffer | GPU_Buffer) -> GPU_Buffer:
        self.ready_buffers(is_preload=False)

        if not isinstance(self.manager, Manager):
            raise ReferenceError("Couldn't find networks manager. Did it get hooked?")

        if self.__forward_kernel is None: raise ReferenceError("Missing Kernel!")
        if self.__reduce_kernel is None: raise ReferenceError("Missing Kernel!")

        if not isinstance(input_values, GPU_Buffer):
            input_values = GPU_Buffer.from_data_buffer(self.manager, input_values)

        output_values = GPU_Buffer.empty(self.manager, (self.output_dim,))
        output_unreduced_values = GPU_Buffer.empty(self.manager, (self.output_dim, self.input_dim))

        self.__forward_kernel(
            self.manager.queue, (self.input_dim, self.output_dim), None,

            input_values.get_gpu(),
            output_unreduced_values.get_gpu(),

            self.weight_buffer,
            self.bias_buffer,

            np.int32(self.input_dim),
            np.int32(self.output_dim)
        ).wait()

        self.__reduce_kernel(
            self.manager.queue, (self.output_dim,), None,

            output_unreduced_values.get_gpu(),
            output_values.get_gpu(),

            np.int32(self.input_dim),
            np.int32(self.output_dim)
        ).wait()

        output_unreduced_values.release()

        return output_values

    def backward(self, input_buffer, previous_error_buffer, learning_rate: float) -> dict:
        self.ready_buffers(is_preload=False)

        if self.__backward_kernel is None: raise ReferenceError("Missing Kernel!")
        if self.__reduce_kernel is None: raise ReferenceError("Missing Kernel!")

        if not isinstance(self.manager, Manager):
            raise ReferenceError("Couldn't find networks manager. Did it get hooked?")

        if not isinstance(input_buffer, GPU_Buffer):
            input_buffer = GPU_Buffer.from_data_buffer(self.manager, input_buffer)

        if not isinstance(previous_error_buffer, GPU_Buffer):
            previous_error_buffer = GPU_Buffer.from_data_buffer(self.manager, previous_error_buffer)

        next_errors_unreduced = GPU_Buffer.empty(self.manager, (self.output_dim, self.input_dim))
        weight_gradients = GPU_Buffer.empty(self.manager, (self.output_dim, self.input_dim))

        next_errors = GPU_Buffer.empty(self.manager, (self.input_dim,))
        bias_gradients = GPU_Buffer.empty(self.manager, (self.output_dim,))

        self.__backward_kernel(
            self.manager.queue, (self.input_dim, self.output_dim), None,

            input_buffer.get_gpu(),
            self.weight_buffer,
            previous_error_buffer.get_gpu(),

            next_errors_unreduced.get_gpu(),
            bias_gradients.get_gpu(),
            weight_gradients.get_gpu(),

            np.int32(self.input_dim),
            np.int32(self.output_dim),
            np.float32(learning_rate)
        )

        self.__reduce_kernel(
            self.manager.queue, (self.input_dim,), None,

            next_errors_unreduced.get_gpu(),
            next_errors.get_gpu(),

            np.int32(self.output_dim),
            np.int32(self.input_dim)
        ).wait()

        return {
            "weight_gradients": weight_gradients,
            "bias_gradients": bias_gradients,
            "next_errors": next_errors
        }