import torch
from gpt import GPT
import logging as l

if __name__ == '__main__':
    l.basicConfig(level=l.DEBUG, format='%(levelname)s: %(message)s')

    ############### Test GPT IMPL ################
    # vocab_size = 15
    # x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).int()
    # config = GPT.get_default_config()
    # config.__dict__.update({
    #     'model_type': 'gpt-nano',
    #     'vocab_size': vocab_size,
    #     'block_size': 5,
    #     'abs_emb': True
    # })
    # model = GPT(config)
    # model(x)
