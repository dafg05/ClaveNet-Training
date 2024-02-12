import numpy as np
import pickle
import os
import mido
import sys
import shutil

from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING
from hvo_processing.hvo_sets import HVOSetRetriever
from tqdm import tqdm
from datetime import datetime
from constants import *

SUBSETS_DIR = PREPROCESSED_DATASETS_DIR + "/PreProcessed_On_12_02_2024_at_06_30_hrs"

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

    return inputs, outputs, hvo_sequences

def process_by_partition(subsets_dir, partition, tappify_params):
    hsr = HVOSetRetriever(subsets_dir)

    if partition == "train":
        subset = hsr.get_train_hvoset()
        metadata = hsr.get_train_metadata()
    elif partition == "test":
        subset = hsr.get_test_hvoset()
        metadata = hsr.get_test_metadata()
    elif partition == "validation":
        subset = hsr.get_validation_hvoset()
        metadata = hsr.get_validation_metadata()
    else:
        raise Exception(f"Invalid partition: {partition}")
    
    return process_subset(subset=subset, metadata=metadata, max_len=MAX_LEN, tappify_params=tappify_params)

def saveMidiData(midi_data, filename, out_dir):
    midi_path = f'{out_dir}/{filename}.mid'
    with open(midi_path, "wb") as binary_file:
        binary_file.write(midi_data)

def writeMidiSetToDir(partition, preprocessed_dir, out_dir):
    hsr = HVOSetRetriever(preprocessed_dir)
    if partition == "train":
        midi_set = hsr.get_train_midiset()
    elif partition == "test":
        midi_set = hsr.get_test_midiset()
    elif partition == "validation":
        midi_set = hsr.get_validation_midiset()
    else:
        raise Exception(f"Invalid partition: {partition}")

    partition_dir = f'{preprocessed_dir}/GrooveMIDI_processed_{partition}'
    with open (f"{partition_dir}/midi_data.obj", "rb") as pickled_midi:
        midi_set = pickle.load(pickled_midi)
        print(f"Loaded {len(midi_set)} midis from {out_dir}")

        for idx, midi in enumerate(midi_set):
            saveMidiData(midi, f"{partition}{idx}", out_dir)

def processing(preprocessed_dir, processed_dir):
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
    out_dir = f"{processed_dir}/processed_at_{processed_time}"
    os.mkdir(out_dir)

    # preprocess run name goes on a txt file inside the out directory
    preprocess_run_name = preprocessed_dir.split("/")[-1]
    with open(f"{out_dir}/preprocessed_run_name.txt", "x") as f:
        f.write(preprocess_run_name)
    
    for p in partitions:
        inputs, outputs, _ = process_by_partition(subsets_dir=preprocessed_dir, partition=p, tappify_params=tappify_params)
        content = {
            "preprocess_run_name" : preprocess_run_name,
            "processed_time" : processed_time,
            "inputs" : inputs,
            "outputs" : outputs,
        }
        filename = f'{out_dir}/{p}.pkl'
        pickle.dump(content, open(filename, 'wb'))

    # copy dataAugParams.json
    shutil.copy(f'{preprocessed_dir}/{DATA_AUG_PARAMS}', f'{out_dir}/{DATA_AUG_PARAMS}')
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process.py <command>")
        print("command: 'process' or 'writeMidi'")
        sys.exit(1)
    
    if sys.argv[1] == "process":
        processing(SUBSETS_DIR, PROCESSED_DATASETS_DIR)

    elif sys.argv[1] == "writeMidi":
        writeMidiSetToDir('validation', SUBSETS_DIR, SOURCE_DIR)

    