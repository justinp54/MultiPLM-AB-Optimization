import os
import argparse
from .ensemble import reconstruct_multi_models
from .utils import seed_everything

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=str, required=True)
    p.add_argument("--models", type=str, nargs="+", default=["esm1b","esm1v1","esm1v2","esm1v3","esm1v4","esm1v5"])
    p.add_argument("--cuda", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    seed_everything(args.seed)

    out = reconstruct_multi_models(args.seq, args.models)
    print("models:", [x["name"] for x in out["models"]])
    print("ensemble logits:", out["ensemble"]["logits"].shape)
    print("ensemble probs:", out["ensemble"]["probs"].shape)

if __name__ == "__main__":
    main()
