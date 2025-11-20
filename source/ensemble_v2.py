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

def votes_top_mutations(seq, model_names, topk=1,prob_threshold = 0.0, margin_threshold = 0.0,
                        temperature = None, sample = None, topk_sample = None, 
                        topp_sample = None, min_model = 1):
    def filter_top_k(p,k):
        if k is None or k<=0 or k>=len(p):
            return p;
        idx = np.argsort(p)[::-1]
        keep = idx[:k]
        out = np.zeros_like(p)
        out[keep] = p[keep]
        s = out.sum()
        return out if s == 0 else out/s

    def filter_top_p(p,topp):
        if topp is None or topp <=0 or topp>=1:
            return p
        idx = np.argsort(p)[::-1]
        csum = np.cumsum(p[idx])
        
        cutoff = np.where(csum<=topp)[0]
        if len(cutoff) == 0:
            keep = idx[0]
        else:
            keep = idx[:cutoff[-1]+1]
        out = np.zeros_like(p)
        out[keep] = p[keep]
        s = out.sum()
        return out if s == 0 else out/s

    def sample_without_replacement(p, num):
        p = p.copy()
        chosen = []
        for _ in range(num):
            s = p.sum()
            if s == 0:
                break;
            j = np.random.choie(len(p),p = p)
            chosen.append(j)
            
            p[j] = 0.0
            s2 = p.sum()
            if s2 == 0:
                break
            p/=s2
        return chosen
     
    ms = make_models(model_names, repr_layer=[-1], model_path=None)
    L = len(seq)
    counts = {}
    weights = {}
    for m in ms:
        logits = m.logits(seq)
        if temperature is not None and temperature>0:
            logits = logits/temperature
        probs_all = _softmax(logits,axis = -1)
        
        for i in range(L):
            row = probs_all[i+1]
            wt = seq[i]
            best_idx = int(np.argmax(row))
            mut = m.alphabet.get_tok(best_idx)
            if len(mut) == 1 and mut != seq[i]:
                key = (i, seq[i], mut)
                counts[key] = counts.get(key, 0) + 1
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    muts = [(pos, mut) for (pos, wt, mut), c in items[:topk]]
    return muts, counts
