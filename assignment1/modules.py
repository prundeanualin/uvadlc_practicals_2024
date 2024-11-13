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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import math

import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization.
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.in_features = in_features
        self.out_features = out_features
        self.input = None

        # Initialization
        # Initialize biases to 0 with shape (1 x out_features), a bias for each neuron
        self.params['bias'] = np.zeros((1, out_features))
        # Kaiming initialization for weights
        if input_layer:
            # no ReLU applied to the inputs, variance is linear
            self.params['weight'] = np.random.normal(size=(out_features, in_features),
                                                     loc=0,
                                                     scale=1 / math.sqrt(in_features))
        else:
            # ReLU applied on the inputs, compensate for its variance
            self.params['weight'] = np.random.normal(size=(out_features, in_features),
                                                     loc=0,
                                                     scale=math.sqrt(2) / math.sqrt(in_features))
        # Initialize gradients to 0
        self.grads['weight'] = np.zeros((self.out_features, self.in_features))
        self.grads['bias'] = np.zeros((1, out_features))
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # Input should be batched and have size (batch_size x n_features)
        self.input = x # stored for computing the gradient later
        out = x @ self.params['weight'].T + self.params['bias']

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.grads['weight'] = dout.T @ self.input
        self.grads['bias'] = dout.sum(axis=0)
        dx = dout @ self.params['weight']
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.input = None
        # Reset gradients
        self.grads['weight'] = np.zeros_like(self.grads['weight'])
        self.grads['bias'] = np.zeros_like(self.grads['bias'])
        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.input = None

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.input = x
        out = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # The derivative of ELU
        d_elu = np.where(self.input > 0, np.ones_like(self.input), self.forward(self.input) + self.alpha)
        dx = np.multiply(dout, d_elu)
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.input = None
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def __init__(self):
        self.output = None

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        max_val_per_sample = np.max(x, axis=1, keepdims=True)
        y = np.exp(x - max_val_per_sample)
        out = y / np.sum(y, axis=1, keepdims=True)
        self.output = out
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # using the formula for dL / dZ from the assignment 1.e), which is defined based on dL / dY
        num_classes = self.output.shape[1]
        inner_parenthesis_result = np.multiply(dout, self.output)
        outer_parenthesis_result = dout - inner_parenthesis_result @ np.ones((num_classes, num_classes))
        dx = np.multiply(self.output, outer_parenthesis_result)
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.output = None
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        num_samples = y.shape[0] # since y is of shape (batch_size x num_classes)
        out = - np.sum(np.multiply(y, np.log(x))) / num_samples
        #######################
        # END OF YOUR CODE    #
        #######################
        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        num_samples = y.shape[0] # since y is of shape (batch_size x num_classes)
        dx = - (y / (x)) / num_samples
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx
