################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, elu_alpha=1):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.elu_alpha = elu_alpha
        # initial layer
        layers = [LinearModule(n_inputs, n_hidden[0], input_layer=True)]
        # include the output layer in the list of hidden layers
        n_hidden = n_hidden + [n_classes]
        for hidden_layer_idx in range(1, len(n_hidden)):
            # activation function between hidden layers
            act_fn = ELUModule(elu_alpha)
            # another hidden layer
            hidden_layer = LinearModule(n_hidden[hidden_layer_idx - 1], n_hidden[hidden_layer_idx])
            layers.extend([act_fn, hidden_layer])
        # final softmax activation
        layers.append(SoftMaxModule())
        self.layers = layers
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Propagate the input through the layers
        for layer in self.layers:
            # print(f"For layer {layer}")
            # print(f"Input: {x}")
            x = layer.forward(x)
            # print(f"Output: {x}")
        out = x
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Propagate the gradient through the layers and the modules, from last to first
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        #######################
        # END OF YOUR CODE    #
        #######################

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        for layer in self.layers:
            layer.clear_cache()
        #######################
        # END OF YOUR CODE    #
        #######################

    def save_model_weights(self, path):
        self.clear_cache()
        print(f"Saving the model weights to {path}")
        linear_layer_idx = 0
        weights_dict = {}
        for layer in self.layers:
            if LinearModule == type(layer):
                weights_dict[f'linear_{linear_layer_idx}-weight'] = layer.params['weight']
                weights_dict[f'linear_{linear_layer_idx}-bias'] = layer.params['bias']
                linear_layer_idx += 1
        np.savez_compressed(path, **weights_dict)

    def load_model_weights(self, path):
        print(f"Loading model weights from {path}")
        loaded_weights_dict = np.load(path, allow_pickle=True)
        linear_layer_idx = 0
        for layer in self.layers:
            if LinearModule == type(layer):
                layer.params['weight'] = loaded_weights_dict[f'linear_{linear_layer_idx}-weight']
                layer.params['bias'] = loaded_weights_dict[f'linear_{linear_layer_idx}-bias']
                linear_layer_idx += 1