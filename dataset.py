import numpy as np
import pickle
import wandb
import pandas as pd
from torch.utils.data import Dataset

PROCESSED_DATA_DIR = "processedData"

class GrooveHVODataset(Dataset):
    def __init__(self, hvo_path, transform=None, target_transform=None):
        self.hvo_pairs = self.get_hvo_pairs(hvo_path)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.hvo_pairs)
    
    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a tuple of (monotonic_hvo, target_hvo), where targer_hvo is the full hvo groove
        """
        monotonic_hvo, target_hvo =  self.hvo_pairs[idx]
        if self.transform:
            monotonic_hvo = self.transform(monotonic_hvo)
        if self.target_transform:
            target_hvo = self.target_transform(target_hvo) 
        if monotonic_hvo.shape[0] != 32:
            raise ValueError(f"Monotonic hvo shape is not 32. monotic_hvo.shape: {monotonic_hvo.shape}, idx: {idx}")
        if target_hvo.shape[0] != 32:
            raise ValueError(f"Target hvo shape is not 32. target_hvo.shape: {target_hvo.shape}, idx: {idx}")
        return monotonic_hvo, target_hvo
    
    def get_hvo_pairs(self, hvo_path) -> pd.DataFrame:
        """
        Returns a dataframe of hvo grooves
        """
        with open(hvo_path, 'rb') as file:
            hvo_pairs = pickle.load(file)
        return hvo_pairs
    

if __name__ == "__main__":
    dataset = GrooveHVODataset(f"{PROCESSED_DATA_DIR}/hvo_pairs.pkl")
    print(f"Total number of HVO pairs in the dataset: {len(dataset)}")

    for i in range(5):
        monotonic_hvo, target_hvo = dataset[i]
        print(f"Sample {i}:")
        print("Monotonic HVO shape:", monotonic_hvo.shape)
        print("Target HVO shape:", target_hvo.shape)