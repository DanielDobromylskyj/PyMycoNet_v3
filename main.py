import numpy as np

from pymyconet import Network
from pymyconet import DenseLayer, ReLU

net = Network([
    DenseLayer(2, 4),
    DenseLayer(4, 1)
], compute_mode="cpu")

print(net.forward([5, 1]).get_buffer())

net.save("reference.net")

gpu_net = Network.load("reference.net", "gpu")

gpu_net.forward([])

print(gpu_net.forward([5, 1]).get_buffer())