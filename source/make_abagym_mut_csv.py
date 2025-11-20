import pandas as pd
from pathlib import Path

def make_abagym_ab_only_mut_csv():
    in_path = Path("../data/AbAgym_data_non-redundant.csv")
    if not in_path.exists():
        print(f"[SKIP] input csv file does not exist")
        return
    
    df = pd.read_csv(in_path)
    valid_chains = ["H","L"]
    df = df[df["chains"].astype(str).str.upper().isin(valid_chains)].copy()
    wanted = ['PDB_file', 'chains', 'site', 'wildtype', 'mutation', 'DMS_score', 
              'MinMax_normalized_DMS_score','Rank_quartile_normalized_DMS_score']
    df_out = df[wanted].copy()
    
    out_path = Path("../data/Abagym_antibody_only_dataset.csv")
    
    df_out.to_csv(out_path,index = False)
    print(f"Antibody only AbaGym csv file created")
    
def make_abagym_mut_csv():
    in_path = Path("../data/AbAgym_data_non-redundant.csv")
    if not in_path.exists():
        print(f"[SKIP] input csv file does not exist")
        return
    
    df = pd.read_csv(in_path)
    wanted = ['PDB_file', 'chains', 'site', 'wildtype', 'mutation', 'DMS_score', 
              'MinMax_normalized_DMS_score','Rank_quartile_normalized_DMS_score']
    df_out = df[wanted].copy()
    
    out_path = Path("../data/Abagym_mutation_full_dataset.csv")
    
    df_out.to_csv(out_path,index = False)
    print(f"Ab+Ag AbaGym data summarized on mutation created as new csv file")