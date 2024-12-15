import multiprocessing

import torch

from assignment3.part1.cnn_encoder_decoder import CNNDecoder
from assignment3.part1.train_pl import VAE
from mnist import mnist

# if __name__ == '__main__':
#     multiprocessing.set_start_method("fork")
#
#     train, val, test = mnist(num_workers=1)
#     x = next(iter(train))
#     print(len(train))
#     print(x[0].shape)
#
#     print(x[0][0])
#
#     vae = VAE(num_filters=32, z_dim=20, lr=1e-3)
#     res = vae.sample(4)
#
#     print(f" [DEBUG] res is: {res.shape}")
#     z_dim = 20
#     decoder = CNNDecoder(z_dim=z_dim)
#     z = torch.randn(8, z_dim)
#
#     output = decoder(z)
