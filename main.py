import numpy as np

from pymyconet import Network
from pymyconet import DenseLayer, ReLU

net = Network([
    DenseLayer(2, 5),
    ReLU(),
    DenseLayer(5, 2),
    ReLU()
])

net.save("test.nn")

result = net.forward([1, 2])

print("Input(1, 2) -> ", result.get_buffer())

for i in range(100_000):
    data = net.train(
        [np.array([1, 2]), np.array([2, 3])],
        [np.array([2, 4]),   np.array([4, 6])],
        learning_rate=0.005
    )

    n_data = net.normalize_many_gradients(data)
    net.apply_gradients(n_data)

result = net.forward([1, 2])
print("Input(1, 2) -> Want(2, 4)", result.get_buffer())

result = net.forward([2, 3])
print("Input(2, 3) -> Want(4, 6)", result.get_buffer())

