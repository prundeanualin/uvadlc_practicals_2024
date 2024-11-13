import numpy as np

from modules import SoftMaxModule
from train_mlp_numpy import accuracy

# if __name__ == '__main__':
#     a = np.arange(100).reshape(10, 10)
#     print(f"a is: \n{a}")
#     b = np.max(a, axis=1, keepdims=True)
#     print(f"b is: \n{b}")
#     print(f"b shape is: {b.shape}")
#
#
#     # Test SoftMax implementation
#     softmax = SoftMaxModule()
#
#     input_1_sample = np.array([1, 2, 3, 4, 5]).reshape(1, -1)
#     # result for 1 sample: [[0.01165623 0.03168492 0.08612854 0.23412166 0.63640865]]
#     input_2_samples = np.array([
#         [1, 2, 3, 4, 5],
#         [2, 5, 9, 11, 16]
#     ])
#
#     input = input_2_samples
#     print(f"Shape of input: {input.shape}")
#     result = softmax.forward(input)
#     print(f"{result}")
#
#
#     # Test accuracy implementation
#     predictions = np.array([
#         [0.8, 0.02, 0.1],
#         [0.1, 0.2, 0.5]
#     ])
#     targets = np.array([
#         [1, 0, 0],
#         [0, 1, 0]
#     ])
#     result = accuracy(predictions, targets)
#     print(f"Accuracy is : {result}")