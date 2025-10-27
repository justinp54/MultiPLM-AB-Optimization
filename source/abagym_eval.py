import os, re, argparse
import pandas as pd
import numpy as np
import torch
from .align2 import extract_seq, site_to_key, build_key
from .ensemble import votes_top_mutations, reconstruct_multi_models
from .utils import seed_everything

print("[LOAD]", __file__, flush=True) 

VALID_CHAINS = ["H","L"]
STRUCT_PATTERN = "{pdbid}_chothia.pdb"

def parse_args():
    p = argparse.ArgumentParser("Full evaluation of model based on AbaGym dataset")
    p.add_argument("--abagym_csv",type = str, required = True)
    p.add_argument("--struct_dir", type = str, default = "../data")
    p.add_argument("--models", type = str, nargs="+", default=["esm1b","esm1v1","esm1v2","esm1v3","esm1v4","esm1v5"])
    p.add_argument("--topk_votes",type = int, default = 50)
    p.add_argument("--cuda", type = str, default = None)
    p.add_argument("--seed",type = int, default = 0)
    return p.parse_args()

def parse_pdbid(profile_key: str):
    s = str(profile_key).strip()
    m = re.search(r'([0-9][A-Za-z0-9]{3})$', s)   # 끝의 4문자(숫자 + 영숫자 3)
    return m.group(1).lower() if m else None


def make_path(struct_dir: str, pdbid: str):
    path = os.path.join(struct_dir, STRUCT_PATTERN.format(pdbid = pdbid))
    return path

def site2key(num: int, icode: str):
    return f"{int(num)}{(icode or '').strip()}"

def positive_dms(df_full: pd.DataFrame, profile: str, chain: str):
    sub = df_full[(df_full["PDB_file"].astype(str) == profile)&(df_full["chains"].astype(str) == chain)].copy()
    
    sub["site_key"] = sub["site"].astype(str).apply(site_to_key)
    sub["mutation"] = sub["mutation"].astype(str).str.upper().str.strip()
    sub = sub[(sub["DMS_score"] >= 0)].copy()
    return sub[["PDB_file","chains","site_key","mutation","DMS_score"]]

def negativee_dms(df_full: pd.DataFrame, profile: str, chain: str):
    sub = df_full[(df_full["PDB_file"].astype(str) == profile)&(df_full["chains"].astype(str) == chain)].copy()
    
    sub["site_key"] = sub["site"].astype(str).apply(site_to_key)
    sub["mutation"] = sub["mutation"].astype(str).str.upper().str.strip()
    sub = sub[(sub["DMS_score"] < 0)].copy()
    return sub[["PDB_file","chains","site_key","mutation","DMS_score"]]

def main():
    print("[INFO] Starting AbAgym evaluation ...")
    args = parse_args()
    
    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    df_full = pd.read_csv(args.abagym_csv)
    df_ab_only = df_full[df_full["chains"].astype(str).str.upper().isin(VALID_CHAINS)].copy()
    
    # profiles = sorted(set(df_ab_only["PDB_file"].astype(str)))
    profiles = ['G6_27_30A_corrected_4zfg', 'G6_27_30A_corrected_4zff']
    summary_rows,detail_rows = [],[]
    
    for prof in profiles:
        pdbid = parse_pdbid(prof)
        struct_path = make_path(args.struct_dir, pdbid)
        if not struct_path:
            print(f"no structure file found: {prof} ({pdbid})")
            continue
        for ch in VALID_CHAINS:
            try: 
                ab_seq, numbering = extract_seq(struct_path,ch)
            except Exception as e:
                print(f"no chain structure file found: {prof} :{ch} ({e})")
                continue
            key = build_key(numbering)
            
            mut_votes, _ = votes_top_mutations(ab_seq, model_names = args.models,topk = args.topk_votes)
            pred_set = set()
            for pos0, aa in mut_votes:
                idx1 = pos0+1
                if 1<= idx1 <= len(numbering):
                    sk = site2key(*numbering[idx1-1])
                    pred_set.add((sk,aa))
                    
            pos_sub = positive_dms(df_ab_only, prof, ch)
            pos_sub = pos_sub[pos_sub["site_key"].isin(key.keys())].copy()
            pos_set = set(zip(pos_sub["site_key"], pos_sub["mutation"]))
            
            neg_sub = positive_dms(df_ab_only, prof, ch)
            neg_sub = neg_sub[neg_sub["site_key"].isin(key.keys())].copy()
            neg_set = set(zip(neg_sub["site_key"], neg_sub["mutation"]))
            
            good_hits = pred_set & pos_set
            bad_hits = pred_set & neg_set
            hits = good_hits |  bad_hits
            n_pred, n_pos, n_good_hit, n_bad_hit = len(pred_set), len(pos_set), len(good_hits), len(bad_hits)
            good_prec = (n_good_hit/n_pred) if n_pred else 0.0
            bad_prec = (n_bad_hit/n_pred) if n_pred else 0.0
            pred_eff = (n_good_hit/(n_bad_hit+n_good_hit)) if (n_bad_hit+n_good_hit) else 0.0
            
            summary_rows.append({"profile_key": prof, "pdbid": pdbid, "chain": ch
                                 , "n_pred": n_pred, "n_pos": n_pos, "improved_mutation_hit": n_good_hit, "declined_mutation_hit": n_bad_hit
                                 ,"imporved_mutation_precision": good_prec,"declined_mutation_precision": bad_prec, "model_prediction_efficiency": pred_eff})
            for sk, aa in good_hits:
                detail_rows.append({"profile_key": prof, "chain": ch, "type": "GOOD_HIT", "site_key": sk, "mutation": aa})
            for sk, aa in bad_hits:
                detail_rows.append({"profile_key": prof, "chain": ch, "type": "BAD_HIT", "site_key": sk, "mutation": aa})
            for sk, aa in (pred_set - hits):
                detail_rows.append({"profile_key": prof, "chain": ch, "type": "PRED_ONLY", "site_key": sk, "mutation": aa})
            for sk, aa in (pos_set - hits):
                detail_rows.append({"profile_key": prof, "chain": ch, "type": "DMS_ONLY", "site_key": sk, "mutation": aa})
            
            print(f"[DONE] {prof}:{ch} | pred={n_pred} dms+={n_pos} good_hit={n_good_hit} bad_hit={n_bad_hit} prediction_efficiency={pred_eff}")
            
        df_sum = df_sum = pd.DataFrame(summary_rows).sort_values(["profile_key","chain"])
        df_det = pd.DataFrame(detail_rows).sort_values(["profile_key","chain","type","site_key","mutation"])
        
        df_sum.to_csv("eval_summary.csv",index = False)
        df_det.to_csv("eval_details.csv",index = False)
        print(f"[SAVE] {prof}:{ch} updated eval_summary.csv / eval_details.csv")
        
if __name__ == "__main__":
    print("[MAIN] entering main()", flush=True)
    main()