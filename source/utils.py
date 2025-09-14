import os
import random
import numpy as np
import torch
import time

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a:i for i,a in enumerate(AA)}

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

def tprint(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def one_hot(seq, max_len=None):
    if max_len is None:
        max_len = len(seq)
    x = np.zeros((max_len, len(AA)), dtype=np.float32)
    L = min(len(seq), max_len)
    for i in range(L):
        a = seq[i]
        if a in AA_TO_IDX:
            x[i, AA_TO_IDX[a]] = 1.0
    return x

def apply_mutations(seq, mutations):
    s = list(seq)
    for pos, mut in mutations:
        if 0 <= pos < len(s):
            s[pos] = mut
    return "".join(s)

def _project_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, ".."))

def get_results_dir(subdir):
    base = os.environ.get("ANALYSIS_RESULTS_DIR")
    if base is None:
        base = os.path.join(_project_root(), subdir)
    os.makedirs(base, exist_ok=True)
    return base

def save_plot(fig, filename, subdir, dpi):
    import time
    if filename is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        filename = f"plot-{ts}.png"
    out_dir = get_results_dir(subdir=subdir)
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path