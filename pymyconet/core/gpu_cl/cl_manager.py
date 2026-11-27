import pyopencl as cl
import numpy as np
import importlib.resources
import typing

from pymyconet.buffer import DataBuffer

if typing.TYPE_CHECKING:
    from pymyconet.network import Network

mf = cl.mem_flags

def size_from_shape(numpy_shape: tuple[int]):
    if len(numpy_shape) == 0:
        return 0

    total = 1
    for size in numpy_shape:
        total *= size

    return total


class GPU_Buffer(DataBuffer):
    def __init__(self, manager: "Manager", cl_buffer: cl.Buffer, dtype: np.dtype, shape: tuple[int]):
        super().__init__(size=size_from_shape(shape) * dtype().nbytes)
        self.manager = manager

        self.cl_buffer = cl_buffer
        self.dtype = dtype
        self.shape = shape if shape is not None else (cl_buffer.size // dtype.itemsize,)

    def release(self):
        self.cl_buffer.release()
        del self

    def get_buffer(self) -> np.ndarray:
        output = np.empty(self.shape, dtype=self.dtype)
        cl.enqueue_copy(self.manager.queue, output, self.cl_buffer).wait()
        return output

    def get_gpu(self) -> cl.Buffer:
        return self.cl_buffer

    def set_buffer(self, buffer: np.ndarray):
        cl.enqueue_copy(self.manager.queue, self.cl_buffer, buffer).wait()

    @staticmethod
    def empty(manager: "Manager", shape: tuple[int, ...]):
        cl_buf = cl.Buffer(manager.context, cl.mem_flags.READ_WRITE, size=(size_from_shape(shape) * np.float32().itemsize))

        return GPU_Buffer(
            manager, cl_buf, np.float32, shape
        )

    @staticmethod
    def from_data_buffer(manager: "Manager", data: DataBuffer):
        buffer = data.get_buffer()

        if buffer.dtype is not np.float32:
            buffer = buffer.astype(np.float32)

        cl_buf = cl.Buffer(manager.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=buffer)

        return GPU_Buffer(
            manager, cl_buf, np.float32, buffer.shape
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


class BufferManager:
    __weights: np.ndarray
    __biases: np.ndarray

    weight_buffer: cl.Buffer
    bias_buffer: cl.Buffer

    network: "Network"
    manager: "Manager"

    def get_weights(self):
        return self.__weights

    def get_biases(self):
        return self.__biases


    def initialize_buffers(self) -> tuple[cl.Buffer, cl.Buffer]:
        if not isinstance(self.manager, Manager):
            raise ReferenceError("Couldn't find networks manager. Did it get hooked?")

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
        return isinstance(self.weight_buffer, cl.Buffer) and isinstance(self.bias_buffer, cl.Buffer)

    def ready_buffers(self, is_preload: bool = False):
        if not isinstance(self.manager, Manager):
            raise ReferenceError("Couldn't find networks manager. Did it get hooked?")

        if self.buffers_are_ready():
            return

        try:
            weights, biases = self.initialize_buffers()
            self.set_buffers(weights, biases)

        except MemoryError:
            if not is_preload:
                raise

    def release(self):
        if isinstance(self.weight_buffer, cl.Buffer):
            self.weight_buffer.release()

        if isinstance(self.bias_buffer, cl.Buffer):
            self.bias_buffer.release()

    def set_buffers(self, w, b):
        self.release()

        self.weight_buffer = w
        self.bias_buffer = b

    def load_kernels(self):
        pass

    def on_hook(self):
        self.manager: Manager = create_or_get_manager(self.network)

        if not isinstance(self.manager, Manager):
            raise ReferenceError("Failed to create / get a manager for this layer!")

        self.load_kernels()

        try:
            weights, biases = self.initialize_buffers()
            self.set_buffers(weights, biases)
        except MemoryError:
            pass

    def is_next(self):
        self.ready_buffers(is_preload=True)



class Manager:
    def __init__(self, device: cl.Device | None=None):
        if device is None:
            platform = cl.get_platforms()[0]
            device = platform.get_devices()[0]

        self.device = device
        self.context = cl.Context(devices=[device])
        self.queue = cl.CommandQueue(self.context, device=device)

        self.__program_lookup: dict[str, cl.Program] = {}
        self.__kernel_lookup: dict[tuple[str, str, str], cl.Kernel] = {}

    def load_program(self, mode: str, path: str) -> cl.Program:
        file_path = str(importlib.resources.files("pymyconet") / "core" / mode / path)

        if file_path in self.__program_lookup:
            return self.__program_lookup[file_path]

        with open(file_path, "r") as f:
            prg_src = f.read()

        program = cl.Program(self.context, prg_src).build()
        self.__program_lookup[file_path] = program

        return program


    def get_kernels(self, mode: str, path: str, kernels: list[str]) -> list[cl.Kernel]:
        loaded_kernels = []

        for kernel in kernels:
            key = (mode, path, kernel)

            if key not in self.__kernel_lookup:
                program = self.load_program(mode, path)

                if not hasattr(program, kernel):
                    raise NameError(f"Kernel '{kernel}' not found in {mode}:{path}")

                self.__kernel_lookup[key] = cl.Kernel(program, kernel)

            loaded_kernels.append(self.__kernel_lookup[key])

        return loaded_kernels


    def global_memory_size(self) -> int:
        return self.device.global_mem_size
