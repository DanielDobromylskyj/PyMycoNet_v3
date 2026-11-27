import numpy as np
import pyopencl as cl

from ..generic.convoluted import ConvolutedLayer as GenericConvLayer
from .cl_manager import Manager, BufferManager, create_or_get_manager, GPU_Buffer

mf = cl.mem_flags


class ConvolutedLayer(GenericConvLayer, BufferManager):
    def __init__(self, input_width: int, input_height: int,
                 kernel_width: int, kernel_height: int,
                 filter_count: int,
                 step_x: int, step_y: int, generate=True):

        super().__init__(input_width, input_height, kernel_width, kernel_height, filter_count, step_x, step_y, generate=generate)


        self.weight_buffer: None | cl.Buffer = None
        self.bias_buffer: None | cl.Buffer = None

        self.__forward_kernel = None
        self.__backward_kernel = None
        self.__reduce_generic = None
        self.__reduce_next_errors = None # Bane of my existence

    def load_kernels(self):
        self.__forward_kernel, self.__backward_kernel, self.__reduce_generic, self.__reduce_next_errors = self.manager.get_kernels(
            "gpu_cl", "kernels/convoluted.c", ["forward", "backward", "reduce_generic", "reduce_next_errors"]
         )

    def forward(self, input_values) -> GPU_Buffer:
        self.ready_buffers(is_preload=False)

        if not isinstance(input_values, GPU_Buffer):
            input_values = GPU_Buffer.from_data_buffer(self.manager, input_values)

        shape = self.calculate_output_dims()
        output_buffer = GPU_Buffer.empty(self.manager, shape)

        if self.__forward_kernel is None: raise ReferenceError("Missing Kernel!")

        self.__forward_kernel(
            self.manager.queue, shape, None,

            input_values.get_gpu(),
            output_buffer.get_gpu(),
            self.weight_buffer,
            self.bias_buffer,

            np.int32(self.input_width),
            np.int32(self.input_height),

            # Output Shape (_, h, w)
            np.int32(shape[2]),
            np.int32(shape[1]),

            np.int32(self.kernel_width),
            np.int32(self.kernel_height),

            np.int32(self.step_x),
            np.int32(self.step_y),

            np.int32(self.filter_count)
        ).wait()

        return output_buffer

    def backward(self, input_buffer, previous_error_buffer, learning_rate: float) -> dict:
        self.ready_buffers(is_preload=False)

        if not isinstance(self.manager, Manager):
            raise ReferenceError("Couldn't find networks manager. Did it get hooked?")

        if self.__backward_kernel is None: raise ReferenceError("Missing Kernel!")
        if self.__reduce_generic is None: raise ReferenceError("Missing Kernel!")
        if self.__reduce_next_errors is None: raise ReferenceError("Missing Kernel!")

        if not isinstance(input_buffer, GPU_Buffer):
            input_buffer = GPU_Buffer.from_data_buffer(self.manager, input_buffer)

        if not isinstance(previous_error_buffer, GPU_Buffer):
            previous_error_buffer = GPU_Buffer.from_data_buffer(self.manager, previous_error_buffer)

        filter_count, output_height, output_width = self.calculate_output_dims()

        weight_gradients_unreduced = GPU_Buffer.empty(self.manager, (
            output_height, output_width, self.filter_count, self.kernel_height, self.kernel_width
        ))
        bias_gradients_unreduced = GPU_Buffer.empty(self.manager, (
            output_height, output_width, self.filter_count
        ))
        # This will require a special reduction, As It needs to turn into an 'input_height * input_width' buffer
        next_error_gradients_unreduced = GPU_Buffer.empty(self.manager, (
            output_height, output_width, self.filter_count, self.kernel_height, self.kernel_width
        ))

        weight_gradients = GPU_Buffer.empty(self.manager, (self.filter_count, self.kernel_height, self.kernel_width))
        bias_gradients = GPU_Buffer.empty(self.manager, (self.filter_count,))
        next_errors = GPU_Buffer.empty(self.manager, (self.input_height, self.input_width))

        self.__backward_kernel(
            self.manager.queue, (filter_count, output_height, output_width), None,

            input_buffer.get_gpu(),
            previous_error_buffer.get_gpu(),
            self.weight_buffer,

            weight_gradients_unreduced.get_gpu(),
            bias_gradients_unreduced.get_gpu(),
            next_error_gradients_unreduced.get_gpu(),

            np.int32(self.input_width),
            np.int32(self.input_height),

            np.int32(output_width),
            np.int32(output_height),

            np.int32(self.kernel_width),
            np.int32(self.kernel_height),

            np.int32(self.step_x),
            np.int32(self.step_y),

            np.int32(self.filter_count),
            np.float32(learning_rate)
        )

        # Reduce Weights & Biases
        reduction_count = output_height * output_width
        weight_output_count = self.filter_count * self.kernel_height * self.kernel_width
        bias_output_count = self.filter_count

        self.__reduce_generic(
            self.manager.queue, (weight_output_count,), None,

            weight_gradients_unreduced.get_gpu(),
            weight_gradients.get_gpu(),

            np.int32(reduction_count),
            np.int32(weight_output_count)
        )

        self.__reduce_generic(
            self.manager.queue, (bias_output_count,), None,

            bias_gradients_unreduced.get_gpu(),
            bias_gradients.get_gpu(),

            np.int32(reduction_count),
            np.int32(bias_output_count)
        )

        self.__reduce_next_errors(
            self.manager.queue,
            (self.input_width, self.input_height),
            None,

            next_error_gradients_unreduced.get_gpu(),
            next_errors.get_gpu(),

            np.int32(self.input_width),
            np.int32(self.input_height),

            np.int32(output_width),
            np.int32(output_height),

            np.int32(self.kernel_width),
            np.int32(self.kernel_height),

            np.int32(self.step_x),
            np.int32(self.step_y),

            np.int32(self.filter_count)
        )

        return {
            "weight_gradients": weight_gradients,
            "bias_gradients": bias_gradients,
            "next_errors": next_errors
        }