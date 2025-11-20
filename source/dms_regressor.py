import os
import argparse
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from models import Model
from align2 import extract_seq, build_key
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split


MODEL_NAMES_DICT = {
        "esm1b": "esm1b_t33_650M_UR50S",
        "esm1v1": "esm1v_t33_650M_UR90S_1",
        "esm1v2": "esm1v_t33_650M_UR90S_2",
        "esm1v3": "esm1v_t33_650M_UR90S_3",
        "esm1v4": "esm1v_t33_650M_UR90S_4",
        "esm1v5": "esm1v_t33_650M_UR90S_5",
        "esm2_650M": "esm2_t33_650M_UR50D",
        "esm2_3B": "esm2_t36_3B_UR50D",
        "esm2_15B": "esm2_t48_15B_UR50D",
}

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
class DMSDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0]
        self.X = torch.from_numpy(X.astype("float32"))
        self.y = torch.from_numpy(y.astype("float32")).view(-1, 1)
        
    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class ESMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),
        )
    
    def forward(self,x):
        out = self.net(x)
        return out

def build_seq_cache(df: pd.DataFrame, pdb_root = "../data/DMS_big_table_PDB_files"):
    seq_cache = {}
    
    for (group_name, group_df) in df.groupby(["PDB_file", "chains"]):
        pdb_file, chain_id = group_name
        if len(chain_id) != 1:
            print("Skipping multi-chain entry:", pdb_file, chain_id)
            continue
        pdb_path = os.path.join(pdb_root, f"{pdb_file}.pdb")
        if not os.path.exists(pdb_path):
            print("PDB file not found:", pdb_path)
            continue
        try:
            seq, numbering = extract_seq(pdb_path, chain_id)
            site2idx = build_key(numbering)
            seq_cache[(pdb_file, chain_id)] = {
                "seq": seq,
                "numbering": numbering,
                "site2idx": site2idx,
            }
        except Exception as e:
            print("Error processing PDB file:", pdb_path, e)
            continue
    return seq_cache

def build_plm_models(model_names):
    models = {}
    for name in model_names:
        if name not in MODEL_NAMES_DICT:
            raise ValueError(f"Unknown model name: {name}")
        models[name] = Model(name = MODEL_NAMES_DICT[name], repr_layer = [-1])
    return models

def build_wt_only_features(df: pd.DataFrame, seq_cache, models, device: torch.device):
    wt_cache = {}
    def get_wt_item(model_name, pdb_file, chain_id):
        key = (model_name, pdb_file, chain_id)
        if key in wt_cache:
            return wt_cache[key]
        if (pdb_file, chain_id) not in seq_cache:
            raise ValueError(f"Sequence not found for PDB {pdb_file} chain {chain_id}")
        info = seq_cache[(pdb_file, chain_id)]
        seq = info["seq"]
        site2idx = info["site2idx"]
        
        model = models[model_name]
        
        print("Encoding WT sequence for model", model_name, "PDB", pdb_file, "chain", chain_id)
        emb_wt = model.encode(seq)
        logits_wt = model.logits(seq)
        
        h_wt_global = emb_wt[1:].mean(axis = 0)
        wt_cache[key] = {
            "emb_wt": emb_wt,
            "logits_wt": logits_wt,
            "h_wt_global": h_wt_global,
            "site2idx": site2idx,
        }
        return wt_cache[key]
    
    pdb_list = []
    x_list = []
    y_list = []
    
    for idx, row in df.iterrows():
        pdb_file = row["PDB_file"]
        chain_id = row["chains"]
        site = str(row["site"])
        wt_aa = row["wildtype"].upper()
        mut_aa = row["mutation"].upper()
        y_val = row["MinMax_normalized_DMS_score"]
        
        feats_per_model = []
        
        try:
            for model_name, model in models.items():
                wt_item = get_wt_item(model_name, pdb_file, chain_id)
                logits_wt = wt_item["logits_wt"]
                emb_wt = wt_item["emb_wt"]
                h_wt_global = wt_item["h_wt_global"]
                site2idx = wt_item["site2idx"]
                alpha = model.alphabet
                
                if site not in site2idx:
                    raise KeyError(f"Site {site} not found in PDB {pdb_file} chain {chain_id}")
                idx1 = site2idx[site]
                pos = idx1
                
                logits_pos = logits_wt[pos]
                idx_wt = alpha.get_idx(wt_aa)
                idx_mut = alpha.get_idx(mut_aa)
                
                logit_wt = logits_pos[idx_wt]
                logit_mut = logits_pos[idx_mut]
                delta_logit = logit_mut - logit_wt
                
                h_wt_pos = emb_wt[pos]

                feat_model = np.concatenate([
                    np.array([delta_logit], dtype = np.float32),
                    h_wt_pos.astype(np.float32),
                    h_wt_global.astype(np.float32),
                ],axis = 0)
                
                feats_per_model.append(feat_model)
            feat_all = np.concatenate(feats_per_model, axis = 0)
            x_list.append(feat_all)
            y_list.append(y_val)
            pdb_list.append(pdb_file)
            
        except Exception as e:
            print("Error processing row idx", idx, e)
            import traceback
            traceback.print_exc()
            print("Problematic row:", row)
            continue
        
    X = np.stack(x_list, axis = 0)
    y = np.array(y_list, dtype = np.float32)
    pdb_ids = np.array(pdb_list)
    
    return X, y, pdb_ids

def train_dms_regressor(X: np.ndarray, y: np.ndarray, pdb_ids: np.ndarray, n_epochs = 20, batch_size = 512, lr = 3e-4, weight_decay = 1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    unique_pdbs = np.unique(pdb_ids)
    train_pdbs, temp_pdbs = train_test_split(unique_pdbs, test_size = 0.3)
    valid_pdbs, test_pdbs = train_test_split(temp_pdbs, test_size = 0.5)
    
    train_mask = np.isin(pdb_ids, train_pdbs)
    valid_mask = np.isin(pdb_ids, valid_pdbs)
    test_mask = np.isin(pdb_ids, test_pdbs)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print("Train set size:", X_train.shape[0])
    print("Valid set size:", X_valid.shape[0])
    print("Test set size:", X_test.shape[0])
    
    train_dataset= DMSDataset(X_train,y_train)
    valid_dataset= DMSDataset(X_valid,y_valid)
    test_dataset= DMSDataset(X_test,y_test)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    
    model = ESMRegressor(input_dim = X.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    criterion = nn.MSELoss()
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            pred = model(xb)
            loss = criterion(pred, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * xb.size(0)
            
        avg_loss = total_loss / len(train_dataset)
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                total_val_loss += loss.item() * xb.size(0)
        avg_val_loss = total_val_loss / len(valid_dataset)
        
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_loss:.4f}, Valid Loss: {avg_val_loss:.4f}")
    model.eval()
    y_true  = []
    y_pred = []
    
    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            pred = model(x)
            y_true.extend(y.numpy().flatten().tolist())
            y_pred.extend(pred.cpu().numpy().flatten().tolist())
        evaluate(np.array(y_true), np.array(y_pred))
    return model

def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_rho, _ = spearmanr(y_true, y_pred)
    
    print("Evaluation Results:")
    print(f"R2: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, Pearson r: {pearson_r:.4f}, Spearman rho: {spearman_rho:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dms_csv", type=str, required=False, help="Path to DMS CSV file", default="../data/Abagym_mutation_full_dataset.csv")
    parser.add_argument("--pdb_root", type=str, required=False, help="Path to PDB files root directory",default="../data/DMS_big_table_PDB_files")
    parser.add_argument("--models", type=str, nargs="+", default=["esm2_650M"], help="List of PLM model names to use")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of training epochs")
    batch_size = parser.add_argument("--batch_size", type=int, default=512, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args() 
    
    seed_everything(args.seed)
    
    
    df = pd.read_csv(args.dms_csv)
    df = df[df["chains"].astype(str).str.len() == 1].copy()
    print("Total DMS data points after filtering single-chain entries:", len(df))
    
    seq_cache = build_seq_cache(df, pdb_root = args.pdb_root)
    plm_models = build_plm_models(args.models)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y,pdb_ids = build_wt_only_features(df, seq_cache, plm_models, device)
    
    model = train_dms_regressor(X, y, pdb_ids, n_epochs = args.n_epochs, batch_size = args.batch_size, lr = args.lr, weight_decay = args.weight_decay)
    print("Training completed.")
    
if __name__ == "__main__":
    main()