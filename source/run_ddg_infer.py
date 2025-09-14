# source/run_ddg_infer.py
import os
import argparse
import numpy as np
import torch

from .ddg_predictor import AbAgNet
from .utils import seed_everything, one_hot, apply_mutations

def _resolve_default_weights():
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, ".."))
    candidates = [
        os.path.join(project_root, "AffinityModel", "ddg_predictor.pt"),
        os.path.join(os.getcwd(), "AffinityModel", "ddg_predictor.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

def _parse_mutations(mut_args, index_base=1):
    muts = []
    for token in mut_args:
        if ":" not in token:
            raise ValueError(f"Invalid mutation format: {token} (use pos:AA, e.g., 30:A)")
        pos_s, aa = token.split(":", 1)
        if not pos_s.isdigit():
            raise ValueError(f"Position must be an integer: {pos_s}")
        pos = int(pos_s)
        if index_base == 1:
            pos -= 1
        if pos < 0:
            raise ValueError(f"Position after index-base conversion is negative: {token}")
        if not isinstance(aa, str) or len(aa) != 1:
            raise ValueError(f"Mutant amino acid must be a single letter: {token}")
        muts.append((pos, aa))
    return muts

def _make_batch_tensors(ab_seq, ag_seq, muts, max_len):
    ab_wt = one_hot(ab_seq, max_len)
    ag_wt = one_hot(ag_seq, max_len)

    X_ab, X_ag, X_ab_mut = [], [], []
    for (pos, aa) in muts:
        ab_m = apply_mutations(ab_seq, [(pos, aa)])
        X_ab.append(ab_wt)
        X_ag.append(ag_wt)
        X_ab_mut.append(one_hot(ab_m, max_len))

    X_ab = torch.tensor(np.stack(X_ab), dtype=torch.float32)
    X_ag = torch.tensor(np.stack(X_ag), dtype=torch.float32)
    X_ab_mut = torch.tensor(np.stack(X_ab_mut), dtype=torch.float32)
    ag_mut = X_ag.clone()
    return X_ab, X_ag, X_ab_mut, ag_mut

def _batched_predict(model, device, X_ab, X_ag, X_ab_mut, ag_mut, batch_size=32):
    N = X_ab.size(0)
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            ab_b = X_ab[i:j].to(device, non_blocking=True)
            ag_b = X_ag[i:j].to(device, non_blocking=True)
            abm_b = X_ab_mut[i:j].to(device, non_blocking=True)
            agm_b = ag_mut[i:j].to(device, non_blocking=True)
            y, _, _, _, _ = model(ab_b, ag_b, abm_b, agm_b)  # (B, 1)
            outs.append(y.squeeze(-1).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)

def main():
    p = argparse.ArgumentParser(description="Predict ΔΔG for antibody mutations using AbAgNet")
    p.add_argument("--ab", type=str, required=True, help="Antibody amino-acid sequence (WT)")
    p.add_argument("--ag", type=str, required=True, help="Antigen amino-acid sequence (WT)")
    p.add_argument("--mut", type=str, nargs="+", required=True, help='Mutations like: 30:A 57:Y')
    p.add_argument("--index-base", type=int, default=1, choices=[0, 1],
                   help="Index base for mutation positions (default: 1 = positions are 1-based)")
    p.add_argument("--max-len", type=int, default=1024, help="Pad/crop one-hot length")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cuda", type=str, default=None, help='GPU id (e.g., "0")')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weights", type=str, default=None, help="Path to ddg_predictor.pt")
    p.add_argument("--out", type=str, default=None, help="Optional CSV output path")
    args = p.parse_args()

    # Device & seed
    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse mutations and build batch tensors
    mutations = _parse_mutations(args.mut, index_base=args.index_base)
    if len(mutations) == 0:
        raise ValueError("No valid mutations were provided.")
    X_ab, X_ag, X_ab_mut, ag_mut = _make_batch_tensors(args.ab, args.ag, mutations, args.max_len)

    # Model & weights
    model = AbAgNet().to(device)
    weight_path = args.weights if args.weights else _resolve_default_weights()
    if weight_path and os.path.isfile(weight_path):
        state = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state, strict=False)

    # Predict
    preds = _batched_predict(
        model, device, X_ab, X_ag, X_ab_mut, ag_mut, batch_size=args.batch_size
    )

    offset = 1 if args.index_base == 1 else 0
    for (pos0, aa), val in zip(mutations, preds):
        print(f"{pos0 + offset},{aa},{float(val):.6f}")

    if args.out:
        import pandas as pd
        rows = []
        for (pos0, aa), val in zip(mutations, preds):
            rows.append({"position": pos0 + offset, "mut_aa": aa, "ddg": float(val)})
        df = pd.DataFrame(rows, columns=["position", "mut_aa", "ddg"])
        df.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
