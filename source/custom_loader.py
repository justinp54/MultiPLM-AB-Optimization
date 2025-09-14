import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, ab, ag, ab_mut, device):
        self.ab = ab
        self.ag = ag
        self.ab_mut = ab_mut
        self.device = device
		
    def __len__(self):
        return self.ab.shape[0]

    def __getitem__(self, idx):       
        ab = self.ab[idx].to(self.device)
        ag = self.ag[idx].to(self.device)
        ab_mut = self.ab_mut[idx].to(self.device)
        return ab, ag, ab_mut
