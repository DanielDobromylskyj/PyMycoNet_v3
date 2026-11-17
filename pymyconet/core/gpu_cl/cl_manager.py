import pyopencl as cl
import numpy as np

from pymyconet.buffer import DataBuffer

class GPU_Buffer(DataBuffer):
    def __init__(self, manager: "Manager", cl_buffer: cl.Buffer, dtype: np.dtype, shape: tuple[int] | None = None):
        super().__init__()
        self.manager = manager

        self.cl_buffer = cl_buffer
        self.dtype = dtype
        self.shape = shape if shape is not None else (cl_buffer.size // dtype.itemsize,)

    def get_buffer(self) -> np.ndarray:
        output = np.empty(self.shape, dtype=self.dtype)
        cl.enqueue_copy(self.manager.queue, output, self.cl_buffer).wait()
        return output

    def set_buffer(self, buffer: np.ndarray):
        cl.enqueue_copy(self.manager.queue, self.cl_buffer, buffer).wait()

    @staticmethod
    def from_data_buffer(manager: "Manager", data: DataBuffer):
        buffer = data.get_buffer()

        if buffer.dtype is not np.float32:
            buffer = buffer.astype(np.float32)

        cl_buf = cl.Buffer(manager.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=buffer)

        return GPU_Buffer(
            manager, cl_buf, np.float32
        )


def is_manager(network):
    return hasattr(network, "_gpu_cl_manager")

def get_manager(network):
    return getattr(network, "_gpu_cl_manager")

def set_manager(network, manager):
    setattr(network, "_gpu_cl_manager", manager)

def create_or_get_manager(network, *args, **kwargs) -> "Manager":
    if is_manager(network):
        return get_manager(network)

    manager = Manager(*args, **kwargs)

    set_manager(network, manager)

    return manager


class Manager:
    def __init__(self, device: cl.Device | None=None):
        if device is None:
            platform = cl.get_platforms()[0]
            device = platform.get_devices()[0]

        self.device = device
        self.context = cl.Context(devices=[device])
        self.queue = cl.CommandQueue(self.context, device=device)

    def global_memory_size(self) -> int:
        return self.device.global_mem_size
