import numpy as np
import pickle
import wandb
import pandas as pd
import torch

from typing import List
from hvo_sequence.hvo_seq import HVO_Sequence
from torch.utils.data import Dataset
from hvo_processing.hvo_sets import HVOSetRetriever
from hvo_processing import converters
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING
from tqdm import tqdm

PROCESSED_DATA_DIR = "processedData"
MAX_LEN = 32

def process_dataset(subset, metadata, max_len, tappify_params, dev):
    """
    Retrieved from: https://github.com/marinaniet0/TransformerGrooveTap2Drum/blob/main/model/dataset.py
    TODO: Some of the unprocessed hvos are of length >>> 32. We're already truncating them to 32, but we should investigate
    why they are so long in the first place.

    Original docstring:
    Process subset of GMD dataset for Tap2Drum

    @:param subset
        Preprocessed subset of GMD, loaded from https://github.com/behzadhaki/preprocessed_dataset
    @:param metadata: DataFrame
        Pandas DF with the subset information
    @:param max_len: int
        Maximum length for the hvo sequences (by default 32)
    @:param tappify_params: dict
        Dictionary containing the parameters for the flatten_voices function that generates the tapped sequences to be
        used as inputs
            - tapped_sequence_voice
            - tapped_sequence_collapsed
            - tapped_sequence_velocity_mode
            - tapped_sequence_offset_mode
    @:return tuple with inputs (tapped sequences), outputs (full-beats) and hvo_sequences (full hvo objects)
    """
    inputs = []
    outputs = []
    hvo_sequences = []
    tapped_voice_idx = list(ROLAND_REDUCED_MAPPING.keys()).index(tappify_params["tapped_sequence_voice"])
    for idx, hvo_seq in enumerate(tqdm(subset)):
        if len(hvo_seq.time_signatures) == 1:
            all_zeros = not np.any(hvo_seq.hvo.flatten())
            if not all_zeros:
                # Add metadata to hvo_sequence
                hvo_seq.drummer = metadata.loc[idx].at["drummer"]
                hvo_seq.session = metadata.loc[idx].at["session"]
                hvo_seq.master_id = metadata.loc[idx].at["master_id"]
                hvo_seq.style_primary = metadata.loc[idx].at["style_primary"]
                hvo_seq.style_secondary = metadata.loc[idx].at["style_secondary"]
                hvo_seq.beat_type = metadata.loc[idx].at["beat_type"]

                pad_count = max(max_len - hvo_seq.hvo.shape[0], 0)
                hvo_seq.hvo = np.pad(hvo_seq.hvo, ((0, pad_count), (0, 0)), "constant")
                hvo_seq.hvo = hvo_seq.hvo[:max_len, :]  # in case seq exceeds max len
                hvo_sequences.append(hvo_seq)
                flat_seq = hvo_seq.flatten_voices(voice_idx=tapped_voice_idx,
                                              reduce_dim=tappify_params["tapped_sequence_collapsed"],
                                              offset_aggregator_modes=tappify_params["tapped_sequence_offset_mode"],
                                              velocity_aggregator_modes=tappify_params["tapped_sequence_velocity_mode"])
                inputs.append(flat_seq)
                outputs.append(hvo_seq.hvo)

    # Load data onto device
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    inputs = torch.FloatTensor(inputs).to(dev)
    outputs = torch.FloatTensor(outputs).to(dev)

    return inputs, outputs, hvo_sequences

class GrooveHVODataset(Dataset):
    def __init__(self, hvo_set: List[HVO_Sequence], metadata: pd.DataFrame, tappify_params: dict, dev: str,transform=None, target_transform=None):
        self.dev = dev
        self.tappify_params = tappify_params

        self.hvo_pairs = self.__get_hvo_pairs__(hvo_set, metadata)
        
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
    
    def __get_hvo_pairs__(self, hvo_set: List[HVO_Sequence], metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a dataframe of hvo grooves
        """
        inputs, outputs, hvo_sequence = process_dataset(hvo_set, metadata, MAX_LEN, self.tappify_params, self.dev)
        return list(zip(inputs, outputs))
    
if __name__ == "__main__":
    tappify_params = {
        "tapped_sequence_voice": "HH_CLOSED",
        "tapped_sequence_collapsed": False,
        "tapped_sequence_velocity_mode": 1,
        "tapped_sequence_offset_mode": 3
    }

    train_set, metadata = HVOSetRetriever("processedDatasets/Processed_On_20_01_2024_at_20_38_hrs").get_testset_and_metadata()
    groove_set = GrooveHVODataset(hvo_set=train_set, metadata=metadata, tappify_params=tappify_params, dev="cpu")
    print(f"Done! Shapes: input: {groove_set[0][0].shape}, output: {groove_set[0][1].shape}")
    