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
This module implements utility functions for downloading and reading CIFAR10 data.
You don't need to change anything here.
"""
import torch
import numpy as np

# tools used or loading cifar10 dataset
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Subset
from torchvision import transforms

def get_dataloader(dataset, batch_size, return_numpy=False):
    collate_fn = numpy_collate_fn if return_numpy else None
    train_dataloader      = DataLoader(dataset=dataset["train"], batch_size=batch_size, shuffle=True, drop_last=True,
                                       collate_fn=collate_fn)
    validation_dataloader = DataLoader(dataset=dataset["validation"], batch_size=batch_size, shuffle=False, drop_last=False,
                                       collate_fn=collate_fn)
    test_dataloader       = DataLoader(dataset=dataset["test"], batch_size=batch_size, shuffle=False, drop_last=False,
                                       collate_fn=collate_fn)
    return {"train": train_dataloader, "validation": validation_dataloader, "test": test_dataloader}


def numpy_collate_fn(batch):
    # flatten the 3x32x32 images into a one-dimensional 3072 array
    imgs = torch.stack([b[0].flatten() for b in batch], dim=0).numpy()
    # one hot encode the labels
    labels = np.array([np.eye(10)[b[1]] for b in batch], dtype=np.int32)
    return imgs, labels


def read_data_sets(data_dir, validation_size=5000, debug=False):
    """
    Returns the dataset readed from data_dir.
    Uses or not uses one-hot encoding for the labels.
    Subsamples validation set with specified size if necessary.
    Args:
      data_dir: Data directory.
      validation_size: Size of validation set
      debug: Load only a small portion of the training set if True
    Returns:
      Dictionary with Train, Validation, Test Datasets
    """

    mean = (0.491, 0.482, 0.447)
    std  = (0.247, 0.243, 0.262)

    data_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])

    train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=data_transforms)
    test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=data_transforms)

    # Subsample the validation set from the train set
    if not 0 <= validation_size <= len(train_dataset):
        raise ValueError("Validation size should be between 0 and {0}. Received: {1}.".format(
            len(train_dataset), validation_size))
    train_dataset, validation_dataset = random_split(train_dataset,
                                                     lengths=[len(train_dataset) - validation_size, validation_size],
                                                     generator=torch.Generator().manual_seed(42))
    if debug:
        train_dataset = Subset(train_dataset, range(0, 200))
    return {'train': train_dataset, 'validation': validation_dataset, 'test': test_dataset}


def get_cifar10(data_dir='data/', validation_size=5000, debug=False):
    """
    Prepares CIFAR10 dataset.
    Args:
      data_dir: Data directory.
      validation_size: Size of validation set
      debug: Load only a fraction of the data if True
    Returns:
      Dictionary with Train, Validation, Test Datasets
    """
    return read_data_sets(data_dir, validation_size, debug)
