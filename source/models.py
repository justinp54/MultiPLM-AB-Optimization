import numpy as np
import torch
import warnings
from esm import pretrained

_model_cache = {}

def _resolve(name):
    m = {
        "esm1b": "esm1b_t33_650M_UR50S",
        "esm1v1": "esm1v_t33_650M_UR90S_1",
        "esm1v2": "esm1v_t33_650M_UR90S_2",
        "esm1v3": "esm1v_t33_650M_UR90S_3",
        "esm1v4": "esm1v_t33_650M_UR90S_4",
        "esm1v5": "esm1v_t33_650M_UR90S_5",
    }
    return m.get(name, name)

def _softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

class Model:
    def __init__(self, name, repr_layer=[-1], model_path=None):
        k = (name, tuple(repr_layer), model_path)
        if k in _model_cache:
            m, a = _model_cache[k]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if model_path:
                    m, a = pretrained.load_model_and_alphabet_local(model_path)
                else:
                    m, a = pretrained.load_model_and_alphabet(_resolve(name))
            m = m.eval()
            _model_cache[k] = (m, a)
        self.model = m
        self.alphabet = a
        self.batch_converter = a.get_batch_converter()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.repr_layer_ = repr_layer

    def _to_device(self, t):
        if torch.cuda.is_available():
            return t.cuda(non_blocking=True)
        return t

    def encode(self, seq):
        data = [("seq", seq)]
        _, _, toks = self.batch_converter(data)
        toks = self._to_device(toks)
        with torch.no_grad():
            out = self.model(
                toks,
                repr_layers=set([l if l >= 0 else self.model.num_layers + 1 + l for l in self.repr_layer_]),
                return_contacts=False,
            )
        rep_key = max(out["repr_layers"].keys())
        rep = out["repr_layers"][rep_key][0].detach().cpu().numpy()
        return rep

    def logits(self, seq):
        data = [("seq", seq)]
        _, _, toks = self.batch_converter(data)
        toks = self._to_device(toks)
        with torch.no_grad():
            out = self.model(toks, repr_layers=set(), return_contacts=False)
            x = out["logits"][0].detach().cpu().numpy()
        return x

    def probs(self, seq):
        return _softmax(self.logits(seq), axis=-1)

    def decode(self, embedding):
        t = embedding
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).float()
        t = t.to(next(self.model.parameters()).device)
        with torch.no_grad():
            x = self.model.lm_head(t).detach().cpu().numpy()
        return x
