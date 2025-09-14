import os
import random
import numpy as np
import torch

def compare(a, b, start=None, end=None):
    s = 0 if start is None else start
    e = len(a) if end is None else end
    return {"range": (s, e), "a": a[s:e], "b": b[s:e]}

def seed_everything(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.set_rng_state(torch.cuda.get_rng_state())
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
