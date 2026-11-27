import numpy as np

from pymyconet import Network
from pymyconet import DenseLayer, ConvolutedLayer, ReLU

demo_input = (np.random.random((100, 100)) * 2) - 1

net = Network([
    ConvolutedLayer(100, 100, 5, 5, 8, 2, 2),
    DenseLayer(48*48*8, 2)
], compute_mode="cpu")

#print(net.forward(demo_input).get_buffer())

values_cpu = net.train([demo_input], [np.array([0, 1])], 0.1)

net.save("reference.net")

gpu_net = Network.load("reference.net", "gpu")

#print(gpu_net.forward(demo_input).get_buffer())

values_gpu = gpu_net.train([demo_input], [np.array([0, 1])], 0.1)

for layer_idx, (cpu_layer, gpu_layer) in enumerate(zip(values_cpu, values_gpu)):
    for gradient_type in ("weight_gradients", "bias_gradients", "debug_next_errors"):
        for i, (cpu_buf, gpu_buf) in enumerate(zip(cpu_layer[gradient_type], gpu_layer[gradient_type])):
            cpu = cpu_buf.get_buffer()
            gpu = gpu_buf.get_buffer()

            diff = cpu - gpu
            abs_diff = np.abs(diff)

            print(f"Layer {1-i} | {gradient_type}")
            print(f"  Max absolute error: {np.max(abs_diff):.10g}")
            print(f"  Mean absolute error: {np.mean(abs_diff):.10g}")
            print(f"  RMS error:          {np.sqrt(np.mean(diff ** 2)):.10g}")
            print()