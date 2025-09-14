import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .ensemble import reconstruct_multi_models, votes_top_mutations
from .utils import tprint, seed_everything, save_plot

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze user-provided protein sequence(s) with ESM ensemble voting and entropy plotting"
    )
    # user-provided sequences (one or more)
    p.add_argument(
        "--seq", type=str, nargs="+", required=True,
        help="One or more protein sequences (e.g., --seq SEQ1 SEQ2 ...)."
    )
    # optional names for sequences (matched by order)
    p.add_argument(
        "--name", type=str, nargs="*", default=None,
        help="Optional labels for the sequences (matched by order)."
    )
    p.add_argument(
        "--models", type=str, nargs="+",
        default=["esm1b", "esm1v1", "esm1v2", "esm1v3", "esm1v4", "esm1v5"],
        help="ESM model names to ensemble."
    )
    p.add_argument("--cuda", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ofname", type=str, default=None,
        help="If provided, save mutation voting results to this CSV file."
    )
    p.add_argument(
        "--save-plot", action="store_true",
        help="If set, save entropy plots into Analysis_results/ via utils.save_plot()."
    )
    p.add_argument(
        "--plot-prefix", type=str, default="entropy",
        help="Filename prefix for saved plots (default: entropy)"
    )
    p.add_argument(
        "--topk", type=int, default=0,
        help="Print top-k voted mutations per sequence (0 = print all)."
    )
    return p.parse_args()

def _entropy_bits(p, axis=-1, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -(p * (np.log(p) / np.log(2.0))).sum(axis=axis)

def _plot_entropy(entropy, title=None):
    fig = plt.figure(figsize=(10, 4))
    x = np.arange(1, len(entropy) + 1)
    plt.plot(x, entropy, marker="o", linewidth=1.5, markersize=3)
    plt.xlabel("Sequence Position")
    plt.ylabel("Shannon Entropy (bits)")
    if title:
        plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    return fig

def _sanitize_filename(s: str) -> str:
    if not isinstance(s, str):
        return "unknown"
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in s.strip()]
    out = "".join(keep)
    return out or "unknown"

def main():
    args = parse_args()
    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    seed_everything(args.seed)

    seqs = args.seq
    names = args.name if args.name is not None else []
    if len(names) < len(seqs):
        # pad names if fewer than sequences
        names = names + [f"seq{i}" for i in range(len(names), len(seqs))]

    all_rows = []  # for optional CSV

    for i, seq in enumerate(seqs):
        name = names[i]
        tprint(f"Analyzing {name} (length={len(seq)})")

        # ---- 1) mutation voting (across models) ----
        muts_top, counts = votes_top_mutations(seq, args.models)  # (list, dict)
        # counts: dict[(pos, wt, mut)] -> n_models
        items = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        # print summary
        if args.topk > 0:
            items_to_show = items[:args.topk]
        else:
            items_to_show = items
        tprint(f"Top mutations for {name}:")
        for (pos, wt, mut), c in items_to_show:
            print(f"  {wt}{pos+1}{mut}  (supported by {c} models)")

        # collect rows for CSV (if requested)
        for (pos, wt, mut), c in items:
            all_rows.append([name, seq, f"{wt}{pos+1}{mut}", c])

        # ---- 2) entropy plot (optional) ----
        if args.save_plot:
            ens = reconstruct_multi_models(seq, args.models)
            probs = ens["ensemble"]["probs"]
            L = len(seq)
            # ESM usually has BOS at index 0
            if probs.shape[0] >= L + 1:
                p_seq = probs[1:L+1]
            else:
                p_seq = probs[:L]
            ent = _entropy_bits(p_seq, axis=-1)
            fig = _plot_entropy(ent, title=f"Entropy per Position — {name}")
            fname = f"{args.plot_prefix}_{_sanitize_filename(name)}.png"
            out_path = save_plot(fig, filename=fname)
            plt.close(fig)
            tprint(f"Saved plot: {out_path}")

    # ---- optional CSV output ----
    if args.ofname:
        df = pd.DataFrame(all_rows, columns=[
            "Name",
            "Sequence",
            "Mutation",
            "Number of Language Models",
        ])
        df.to_csv(args.ofname, index=False)
        tprint(f"Saved CSV: {args.ofname}")

if __name__ == "__main__":
    main()
