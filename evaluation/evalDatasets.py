import os
import pickle
import numpy as np
import pandas as pd
import torch
from copy import deepcopy

from cnarch.grooveTransformer import GrooveTransformer

from torch.utils.data import Dataset
from hvo_sequence.hvo_seq import HVO_Sequence

from evaluation.constants import *

class ValidationHvoDataset(Dataset):
    """
    Based on: https://github.com/behzadhaki/GrooveEvaluator/blob/main/test/feature_extractor_test.py
    """

    def __init__(self, source_dir, metadata_csv='metadata.csv', hvo_pickle='hvo_sequence_data.obj', max_len=32, subset=None):
        if subset:
            self.hvo_sequences = subset
        else:
            data_file = open(os.path.join(source_dir, hvo_pickle), 'rb')
            dataset = pickle.load(data_file)
            metadata = pd.read_csv(os.path.join(source_dir, metadata_csv))
            self.hvo_sequences = self.__populate_hvo_sequences(dataset, metadata, max_len)

    def __len__(self):
        return len(self.hvo_sequences)
    
    def __getitem__(self, idx) -> HVO_Sequence:
        return self.hvo_sequences[idx]

    def __populate_hvo_sequences(self, dataset, metadata, max_len):
        hvo_sequences = []
        for ix, hvo_seq in enumerate(dataset):
            if len(hvo_seq.time_signatures) == 1:       # ignore if time_signature change happens
                all_zeros = not np.any(hvo_seq.hvo.flatten())
                if not all_zeros:  # Ignore silent patterns
                    # add metadata to hvo_seq scores
                    hvo_seq.drummer = metadata.loc[ix].at["drummer"]
                    hvo_seq.session = metadata.loc[ix].at["session"]
                    hvo_seq.master_id = metadata.loc[ix].at["master_id"]
                    hvo_seq.style_primary = metadata.loc[ix].at["style_primary"]
                    hvo_seq.style_secondary = metadata.loc[ix].at["style_secondary"]
                    hvo_seq.beat_type = metadata.loc[ix].at["beat_type"]
                    # pad with zeros to match max_len
                    pad_count = max(max_len - hvo_seq.hvo.shape[0], 0)
                    hvo_seq.hvo = np.pad(hvo_seq.hvo, ((0, pad_count), (0, 0)), 'constant')
                    hvo_seq.hvo = hvo_seq.hvo [:max_len, :]         # In case, sequence exceeds max_len
                    hvo_sequences.append(hvo_seq)
        return hvo_sequences 

class MonotonicHvoDataset(Dataset):
    def __init__(self, validationHvoDataset: ValidationHvoDataset):
        self.monotonic_sequences = self.__populate_monotonic_sequences(validationHvoDataset)

    def __populate_monotonic_sequences(self, validationHvoDataset: ValidationHvoDataset):
        monotonic_sequences = []
        for hvo_seq in validationHvoDataset:
            monotonic_seq = deepcopy(hvo_seq)
            monotonic_seq.hvo = monotonic_seq.flatten_voices()
            monotonic_sequences.append(monotonic_seq)
        
        return monotonic_sequences
    
    def __len__(self):
        return len(self.monotonic_sequences)
    
    def __getitem__(self, idx) -> HVO_Sequence:
        return self.monotonic_sequences[idx]
        

class GeneratedHvoDataset(Dataset):
    def __init__(self, monotonicHvoDataset: MonotonicHvoDataset, model: GrooveTransformer):
        self.generated_sequences = self.__populate_generated_sequences(monotonicHvoDataset, model)

    def __populate_generated_sequences(self, monotonicHvoDataset: MonotonicHvoDataset, model: GrooveTransformer):
        generated_sequences = []
        for monotonic_seq in monotonicHvoDataset:
            generated_seq = deepcopy(monotonic_seq)
            hvoArray = self.__pad_hvo_timesteps(monotonic_seq.hvo, time_steps=32)
            hvoTensor = torch.from_numpy(hvoArray).float()

            # run inference
            h, v, o = model.inference(hvoTensor)
            generatedTensor = torch.cat((h, v, o), dim=2)

            assert len(generatedTensor) == 1, f"Batch size of generatedTensor should be 1! generatedTensor.shape: {generatedTensor.shape}"
            generatedArray = generatedTensor[0].detach().numpy()
            generated_seq.hvo = generatedArray

            generated_sequences.append(generated_seq)

        return generated_sequences
    
    def __pad_hvo_timesteps(self, hvo_array, time_steps) -> np.ndarray:    
        assert len(hvo_array) <= time_steps, f"can't pad hvo_array to {time_steps} time steps because it is already {len(hvo_array)} time steps long"

        missing_timesteps = time_steps - len(hvo_array)
        return np.pad(hvo_array, ((0, missing_timesteps),(0, 0)), 'constant', constant_values=0.0)
    
    def __len__(self):
        return len(self.generated_sequences)
    
    def __getitem__(self, idx) -> HVO_Sequence:
        return self.generated_sequences[idx]
    


    