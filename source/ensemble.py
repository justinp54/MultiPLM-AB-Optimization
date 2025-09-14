import numpy as np
from .models import Model, _softmax
from .utils import one_hot, apply_mutations

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

def votes_top_mutations(seq, model_names, topk=1):
    ms = make_models(model_names, repr_layer=[-1], model_path=None)
    L = len(seq)
    counts = {}
    for m in ms:
        logits = m.logits(seq)
        for i in range(L):
            row = logits[i+1]
            best_idx = int(np.argmax(row))
            mut = m.alphabet.get_tok(best_idx)
            if len(mut) == 1 and mut != seq[i]:
                key = (i, seq[i], mut)
                counts[key] = counts.get(key, 0) + 1
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    muts = [(pos, mut) for (pos, wt, mut), c in items[:topk]]
    return muts, counts
