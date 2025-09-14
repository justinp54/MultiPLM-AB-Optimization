import numpy as np
from .models import Model, _softmax

def make_models(model_names, repr_layer=[-1], model_path=None):
    return [Model(name=n, repr_layer=repr_layer, model_path=model_path) for n in model_names]

def predict_one(seq, model):
    lg = model.logits(seq)
    pr = _softmax(lg, axis=-1)
    return lg, pr

def ensemble_logits(logits_list):
    s = None
    for lg in logits_list:
        s = lg if s is None else s + lg
    return s / float(len(logits_list))

def reconstruct_multi_models(seq, model_names, model_path=None):
    ms = make_models(model_names, repr_layer=[-1], model_path=model_path)
    per = []
    lgs = []
    for m, n in zip(ms, model_names):
        lg, pr = predict_one(seq, m)
        per.append({"name": n, "logits": lg, "probs": pr})
        lgs.append(lg)
    ens_lg = ensemble_logits(lgs)
    ens_pr = _softmax(ens_lg, axis=-1)
    return {"models": per, "ensemble": {"logits": ens_lg, "probs": ens_pr}}
