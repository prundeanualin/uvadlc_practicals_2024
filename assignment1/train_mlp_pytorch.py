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
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import time
from pprint import pprint

import numpy as np

import plot_utils
from utils import tqdm_
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    prediction_max_prob = torch.argmax(predictions, dim=1)
    target_class = torch.argmax(targets, dim=1)
    num_samples = len(predictions)
    accuracy = (prediction_max_prob == target_class).sum() / num_samples
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader, device, loss_fn: nn.CrossEntropyLoss = None, desc=''):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
      device: The device to evaluate the model on.
      loss_fn: The loss function to compute the loss. If missing, will only compute the accuracy
      desc: Description to be used in the progress bar
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.
      avg_loss: scalar float, the average loss of the model on the dataset.
                If the loss function is missing, this will be None

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model.eval()
    accuracy_per_batch = []
    loss_per_batch = []
    with torch.no_grad():
        for images, labels in tqdm_(data_loader, total=len(data_loader), desc=desc):
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            # If the loss function is present, will also calculate the loss for this dataset
            if loss_fn:
                batch_loss = loss_fn(predictions, labels).item()
                loss_per_batch.append(batch_loss)
            batch_accuracy = accuracy(predictions, labels).item()
            accuracy_per_batch.append(batch_accuracy)
        avg_accuracy = sum(accuracy_per_batch) / len(accuracy_per_batch)
        avg_loss = None
        if loss_fn:
            avg_loss = sum(loss_per_batch) / len(loss_per_batch)
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy, avg_loss


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir, debug):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
      debug: Run training in debug mode, with smaller dataset
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation.
      logging_dict: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))

    start_time = time.time()

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir, debug=debug)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    train_set = cifar10_loader['train']
    validation_set = cifar10_loader['validation']
    test_set = cifar10_loader['test']

    # 10 classes in CIFAR10 dataset
    num_classes = 10
    # Images in CIFAR10 are 3x32x32 => 3072 flattened array
    input_shape = 3072
    best_weights_save_path = 'best_weights_pytorch.pt'

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Initialize the model
    model = MLP(input_shape, hidden_dims, num_classes, use_batch_norm=use_batch_norm).to(device)
    print(f"Initialized model: {model}")
    # Initialize the loss
    loss_module = nn.CrossEntropyLoss()
    # Initialize the optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_losses = []
    best_val_loss = math.inf
    val_losses = []
    val_accuracies = []

    print("Start training...")
    for epoch in range(epochs):
        epoch_loss_per_batch = []
        tq_iter = tqdm_(train_set, total=len(train_set), desc=f'Train Epoch {epoch + 1}/{epochs}')
        for images, labels in tq_iter:
            # Send input and labels to device
            images, labels = images.to(device), labels.to(device)

            # Model inference
            predictions = model(images)

            # Compute the loss
            loss = loss_module(predictions, labels)
            # Save the loss
            epoch_loss_per_batch.append(loss.item())

            # Reset the gradient
            optimizer.zero_grad()

            # Compute & back-propagate the gradient of the loss function
            loss.backward()

            # SGD update of the model
            optimizer.step()

            # Put the batch loss in the progress bar description
            postfix: dict[str, str] = {"Loss": f"{epoch_loss_per_batch[-1]:.4f}"}
            tq_iter.set_postfix(postfix)

        # Compute the average training loss for this epoch
        epoch_loss = sum(epoch_loss_per_batch)/len(epoch_loss_per_batch)
        train_losses.append(epoch_loss)

        # Evaluate the model on the validation set, computing both loss and acc
        valid_acc, valid_loss = evaluate_model(model, validation_set, device, loss_module, desc=f'Valid Epoch {epoch + 1}/{epochs}')
        val_losses.append(valid_loss)
        val_accuracies.append(valid_acc)
        print(f"\nEpoch {epoch + 1}: Train loss: {epoch_loss:.3f}  |  Valid loss: {valid_loss:.3f}  |  Valid accuracy: {valid_acc * 100:.2f}%")

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            print(f"New best validation loss: {best_val_loss:.3f}!")
            model.save_model(best_weights_save_path)
    print("Done training!")

    # Load the best model
    model.load_model(best_weights_save_path)

    # Evaluate the best model on the test set
    test_accuracy, _ = evaluate_model(model, test_set, device, desc='Test')
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Running time: {duration:.2f} seconds")

    logging_dict = {
        'train/loss': train_losses,
        'val/loss': val_losses,
        'val/acc': val_accuracies,
    }
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    parser.add_argument('--debug', action='store_true', help='Use only 200 samples from the dataset for training')

    args = parser.parse_args()
    kwargs = vars(args)
    pprint(args)
    if args.debug:
        print("!!! DEBUG MODE !!!")

    best_model, _, _, logging_dict = train(**kwargs)
    plot_suffix = 'pytorch'
    if args.use_batch_norm:
        plot_suffix += '_batchnorm'
    # Feel free to add any additional functions, such as plotting of the loss curve here
    plot_utils.plot_train_valid_losses_per_epoch(logging_dict['train/loss'],
                                                 logging_dict['val/loss'],
                                                 suffix=plot_suffix)
