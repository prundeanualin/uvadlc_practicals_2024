# For better visualization of progress bar
from functools import partial

from tqdm import tqdm

tqdm_ = partial(tqdm,
                dynamic_ncols=True,
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]')