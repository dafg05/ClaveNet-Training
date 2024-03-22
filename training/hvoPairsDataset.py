import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

class HvoPairsDataset(Dataset):
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, dev: str, transform=None, target_transform=None):
        self.dev = dev
        self.hvo_pairs = self.__get_hvo_pairs__(inputs, outputs)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.hvo_pairs)
    
    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a tuple of (monotonic_hvo, target_hvo), where target_hvo is the full hvo groove
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
    
    def __get_hvo_pairs__(self, inputs, outputs) -> pd.DataFrame:
        """
        Returns a dataframe of hvo grooves
        """
        inputs = torch.FloatTensor(inputs).to(self.dev)
        outputs = torch.FloatTensor(outputs).to(self.dev)
        return list(zip(inputs, outputs))