from Bio.PDB import PDBParser, Polypeptide
import re
import pandas as pd


def extract_seq(structure_path: str, chain_id: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", structure_path)
    model = next(structure.get_models())
    chain = model[chain_id]

    aa, numbering = [], []
    for res in chain.get_residues():
        hetflag, resseq, icode = res.id
        if hetflag != " ":
            continue
        three = str(res.get_resname())
        try:
            one_index = Polypeptide.three_to_index(three)
            one = Polypeptide.index_to_one(one_index)
        except KeyError:
            continue
        aa.append(one)
        numbering.append((int(resseq), (icode or "").strip()))
    return "".join(aa), numbering

def site_to_key(site_str: str):
    m = re.fullmatch(r'(\d+)([A-Za-z]?)', str(site_str).strip())
    return f"{int(m.group(1))}{ m.group(2).upper() if m.group(2) else '' }"

def build_key(numbering): # numbering = list of tuples(resseq, icode)
    mp = {}
    for i, (resseq, icode) in enumerate(numbering, start=1):
        key = str(resseq) + (icode if icode else "")
        mp[key] = i
    return mp

def unique_wt(df_sub: pd.DataFrame):
    t = df_sub[df_sub["wildtype"].astype(str).str.len()==1].copy()
    t["site_str"] = t["site"].astype(str)
    wt_by_site = (t.groupby("site_str")["wildtype"]
                    .agg("first").astype(str).str.upper())
    return wt_by_site.to_dict()

def check_agreement(df_sub, ref_seq, numbering):
    site2idx = build_key(numbering) #PDB numbering, 1-based indexing
    wt_dict = unique_wt(df_sub) #AbaGym numbering

    rows = []
    ok = bad = miss = 0
    for site, wt in wt_dict.items():
        idx = site2idx.get(site)  # 1-based index
        if idx is None:
            rows.append({"site": site, "wt": wt, "status": "NO_INDEX", "ref_idx": None, "ref_aa": None})
            miss += 1
            continue
        
        ref_aa = ref_seq[idx-1] #string 0-based index
        if ref_aa == wt:
            rows.append({"site": site, "wt": wt, "status": "OK", "ref_idx": idx, "ref_aa": ref_aa})
            ok += 1
        else:
            rows.append({"site": site, "wt": wt, "status": "MISMATCH", "ref_idx": idx, "ref_aa": ref_aa})
            bad += 1

    df = pd.DataFrame(rows).sort_values(["status","ref_idx"], na_position="last")
    total = ok + bad + miss
    print(f"Agreement summary: OK={ok}, MISMATCH={bad}, NO_INDEX={miss}, TOTAL={total}")
    return df