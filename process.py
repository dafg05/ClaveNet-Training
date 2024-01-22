import numpy as np
import pickle
import os

from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING
from hvo_processing.hvo_sets import HVOSetRetriever
from tqdm import tqdm
from datetime import datetime

MAX_LEN = 32

PROCESSED_DATASETS_DIR = "processedDatasets"
PREPROCESSED_DATASETS_DIR = "preprocessedDatasets"
SUBSETS_DIR = PREPROCESSED_DATASETS_DIR + "/Processed_On_20_01_2024_at_20_38_hrs"

def process_subset(subset, metadata, max_len, tappify_params):
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

    # Daniel's change: we skip loading data to device, since the training and processing scripts are separate
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    # inputs = torch.FloatTensor(inputs).to(dev)
    # outputs = torch.FloatTensor(outputs).to(dev)

    return inputs, outputs, hvo_sequences

def process_by_partition(subsets_dir, partition, tappify_params):
    hsr = HVOSetRetriever(subsets_dir)

    if partition == "train":
        subset, metadata = hsr.get_trainset_and_metadata()
    elif partition == "test":
        subset, metadata = hsr.get_testset_and_metadata()
    elif partition == "validation":
        subset, metadata = hsr.get_validationset_and_metadata()
    else:
        raise Exception(f"Invalid partition: {partition}")
    
    return process_subset(subset=subset, metadata=metadata, max_len=MAX_LEN, tappify_params=tappify_params)

   
    
if __name__ == "__main__":
    """
    Processed train, test, and validation sets. Write them to processedDatasets
    """
    partitions = ["train", "test", "validation"]

    tappify_params = {
        "tapped_sequence_voice": "HH_CLOSED",
        "tapped_sequence_collapsed": False,
        "tapped_sequence_velocity_mode": 1,
        "tapped_sequence_offset_mode": 3
    }

    # process time meta data goes on out_directory name
    processed_time = int(datetime.now().timestamp())
    out_dir = f"{PROCESSED_DATASETS_DIR}/processed_at_{processed_time}"
    os.mkdir(out_dir)

    # preprocess run name goes on a txt file inside the out directory
    preprocess_run_name = SUBSETS_DIR.split("/")[-1]
    with open(f"{out_dir}/preprocessed_run_name.txt", "x") as f:
        f.write(preprocess_run_name)
    
    for p in partitions:
        inputs, outputs, hvo_sequences = process_by_partition(subsets_dir=SUBSETS_DIR, partition=p, tappify_params=tappify_params)
        content = {
            "preprocess_run_name" : preprocess_run_name,
            "processed_time" : processed_time,
            "inputs" : inputs,
            "outputs" : outputs,
            "hvo_sequences" : hvo_sequences
        }
        filename = f'{out_dir}/{p}.pkl'
        pickle.dump(content, open(filename, 'wb'))

    