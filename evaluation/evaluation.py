import torch
import sys
import json
import pickle
from datetime import datetime
from pathlib import Path

from training.grooveTransformer import GrooveTransformer as GT
import grooveEvaluator.relativeComparison as rc
from evaluation.evalDatasets import *
from evaluation.constants import *

def evaluateModel(out_dir: Path, model_path: Path, validation_set_path: Path, synthesize_up_to: int=0):
    """
    Evaluate the model on a validation data set. Returns the evaluation time for bookkeeping purposes
    """

    model = loadModel(model_path)
    validation_set = ValidationHvoDataset(validation_set_path)

    # Initiazlize the datasets
    monotonic_set = MonotonicHvoDataset(validation_set)
    generated_set = GeneratedHvoDataset(monotonic_set, model)
    
    assert synthesize_up_to <= len(validation_set)
    
    # Perform relative comparison
    comparison_result_by_feat = rc.relative_comparison(generated_set, validation_set)

    # Create a directory to store the evaluation results
    evaluation_time = int(datetime.now().timestamp())
    evaluation_dir = Path(out_dir) / f'evaluation_{evaluation_time}'
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Synthesize reference audio files
    if synthesize_up_to > 0:
        # Create a directory to store the audio samples
        audio_dir = evaluation_dir / 'audio_samples'
        audio_dir.mkdir(parents=True, exist_ok=True)

        for i in range(synthesize_up_to):
            validation_set[i].save_audio(f'{audio_dir}/sample{i}_validation.wav', sf_path=SF_PATH)
            monotonic_set[i].save_audio(f'{audio_dir}/sample{i}_monotonic.wav', sf_path=SF_PATH)
            generated_set[i].save_audio(f'{audio_dir}/sample{i}_generated.wav', sf_path=SF_PATH)
        print(f"Saved {synthesize_up_to} sets of audio files to {audio_dir}")

    # Save the relative comparison results
    results_path = evaluation_dir / 'results'
    pickle.dump(comparison_result_by_feat, open(results_path, 'wb'))

    return evaluation_time

def loadModel(model_path: Path) -> GT:
    """
    Loads model from its path
    """
    is_smol = model_path.name.split('_')[0] == 'smol'

    hyperparams_setting = model_path.name.split('_')[1]
    hyperparams_filename = f'{hyperparams_setting}.json'
    hypersPath = HYPERS_DIR / hyperparams_filename

    with open(hypersPath) as hp:
        hypersDict = json.load(hp)

    d_model = 8 if is_smol else hypersDict["d_model"]
    dim_forward = hypersDict["dim_forward"]
    n_heads = hypersDict["n_heads"]
    n_layers = hypersDict["n_layers"]
    dropout = hypersDict["dropout"]

    model = GT(d_model=d_model, nhead = n_heads, num_layers=n_layers, dim_feedforward=dim_forward, dropout=dropout, voices=9)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


def usage_and_exit():
    print("Usage: python evaluation.py <modelStartTime> <hypersFile> <synthesize_up_to> <eval_set_size>")
    print("modelStartTime: used to id the model")
    print("hypersFile: json file containing hyperparameters")
    print("synthesize_up_to: number of audiofiles that will be synthesized")
    print("eval_set_size: len of eval set (a subset of validation set)")
    sys.exit(1)