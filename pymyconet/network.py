import json
import zipfile
import numpy as np

from core.generic.activation import ActivationLayer
from core.generic.layer import Layer
from .hardware import get_hardware_option, load_core_segment, get_core_path
from .buffer import DataBuffer

from .loader import import_from_generic_v1

class Network:
    def __init__(self, layout: list, compute_mode: str = "cpu", verbose: bool = True, auto_gen_weights=True):
        self.__verbose = verbose

        self.__mode = get_hardware_option(compute_mode)
        self.__core_path = get_core_path(self.__mode)
        if self.__verbose: print("[Network] Using:", self.__mode)

        self.__generic_layers = layout
        self.__layers: list[Layer] = self.__load_core_layers(auto_gen_weights)
        if self.__verbose: print("[Network] Loaded core")


    def grab_layer(self, i):
        return self.__layers[i]

    def ensure_ready(self):
        if not self.__layers:
            self.__layers = self.__load_core_layers(False)

        if self.__verbose:
            print("[Network] Ready")

    def __load_core_layers(self, generate_default_values: bool):
        return [
            layer.initialize_core(
                load_core_segment(
                    self.__core_path + "." + layer.__class__.__module__.split(".")[-1],
                    layer.__class__.__name__
                ), generate_default_values
            )
            for layer in self.__generic_layers
        ]


    def _forward(self, input_values: DataBuffer) -> DataBuffer:
        working_values = input_values
        for layer in self.__layers:
            working_values = layer.forward(working_values)

        return working_values

    def _forward_saver(self, input_values: DataBuffer) -> list[DataBuffer]:
        data = [input_values]
        working_values = input_values

        for layer in self.__layers:
            working_values = layer.forward(working_values)
            data.append(working_values)

        return data


    def forward(self, input_values: DataBuffer | np.ndarray | list | tuple, get_layer_values: bool = False) -> list[DataBuffer] | DataBuffer:
        forward_func = self._forward if not get_layer_values else self._forward_saver
        if isinstance(input_values, DataBuffer):
            return forward_func(input_values)

        elif isinstance(input_values, np.ndarray):
            return forward_func(DataBuffer(input_values))

        elif isinstance(input_values, list) or isinstance(input_values, tuple):
            return forward_func(DataBuffer(np.array(input_values, dtype=np.float32)))

        else:
            raise ValueError(f"Bad Input Type!")

    def _backward(self, layer_inputs: list[DataBuffer], error_values: DataBuffer, learning_rate: float) -> dict[str, list[DataBuffer]]:
        weight_gradients = []
        bias_gradients = []

        for i in range(len(self.__layers)-1, -1, -1):
            layer = self.__layers[i]

            values: dict[str, DataBuffer] = layer.backward(
                layer_inputs[i],
                error_values,
                learning_rate
            )

            error_values = values["next_errors"]

            weight_gradients.append(
                values["weight_gradients"]
            )
            bias_gradients.append(
                values["bias_gradients"]
            )

        return {
            "weight_gradients": weight_gradients,
            "bias_gradients": bias_gradients
        }


    def backward(self, layer_inputs: list[DataBuffer], error_values: DataBuffer | np.ndarray, learning_rate: float):
        if isinstance(error_values, DataBuffer):
            return self._backward(layer_inputs, error_values, learning_rate)

        elif isinstance(error_values, np.ndarray):
            return self._backward(layer_inputs, DataBuffer(error_values), learning_rate)

        else:
            raise ValueError(f"Bad Input Type!")

    def train(self, input_values: list[np.ndarray | DataBuffer], target_values: list[np.ndarray], learning_rate: float) -> list[dict[str, list[DataBuffer]]]:
        if len(input_values) != len(target_values):
            raise ValueError("Input / Target lists are not the same length!")

        layer_values = [
            self.forward(input_value, get_layer_values=True)
            for input_value in input_values
        ]

        all_gradients = []

        for i, layer_data in enumerate(layer_values):
            if isinstance(layer_data, DataBuffer):
                raise ValueError("Internal forward did NOT return layer data")

            errors = target_values[i] - layer_data[-1].get_buffer()

            gradients = self.backward(layer_data, errors, learning_rate)
            all_gradients.append( gradients )

        return all_gradients

    def normalize_many_gradients(self, gradients: list[dict[str, list[DataBuffer]]]) -> dict[str, list[np.ndarray]]:
        output_weights = [
            x.get_buffer() for x in gradients[0]["weight_gradients"]
        ]
        output_biases = [
            x.get_buffer() for x in gradients[0]["bias_gradients"]
        ]

        for gradient in gradients[1:]:
            for j, layer_grad in enumerate(gradient):
                output_weights[j] += (gradient["weight_gradients"][j].get_buffer()) / len(gradients)
                output_biases[j] += (gradient["bias_gradients"][j].get_buffer()) / len(gradients)

        return {
            "weight_gradients": output_weights,
            "bias_gradients": output_biases
        }

    def apply_gradients(self, gradients: dict[str, list[np.ndarray]]):
        for i in range(len(self.__layers)):
            layer = self.__layers[len(self.__layers) - i-1]

            layer.apply_gradients(
                gradients["weight_gradients"][i],
                gradients["bias_gradients"][i]
            )

    @staticmethod
    def load(path: str, compute_mode: str = "cpu", verbose: bool = True):
        core_option = get_hardware_option(compute_mode)
        core_path = get_core_path(compute_mode)

        with zipfile.ZipFile(path, "r", compression=zipfile.ZIP_DEFLATED) as z:
            metadata = json.loads(z.read("metadata.json"))
            version = metadata["version"]

            if version == 1:
                layers = import_from_generic_v1(core_path, metadata, z)

            else:
                raise NotImplementedError("Unknown File Format / Version!")

        n = Network(layers, core_option, verbose, auto_gen_weights=False)
        return n

    def print(self):
        print(f"> PyMycoNet: {self.__mode} ({len(self.__layers)} layers)")
        last_layer = None
        for layer in self.__layers:
            if isinstance(layer, ActivationLayer) and last_layer is not None:
                print(" ->", str(layer), end="")
            else:
                if last_layer is None:
                    print(str(layer), end="")
                else:
                    print("\n" + str(layer), end="")

            last_layer = layer

        print()


    def save(self, path: str):
        SAVE_VERSION = 1

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            layers = []
            for i, layer in enumerate(self.__layers):
                meta, blob = layer.serialize()

                path = f"layer-{i}.bin"
                z.writestr(path, blob)

                layers.append({
                    "blob": path,
                    "meta": meta,
                    "type": layer.__class__.__name__,
                    "file": layer.__class__.__module__.split(".")[-1]
                })

            z.writestr("metadata.json", json.dumps({
                "version": SAVE_VERSION,
                "layers": layers
            }))




