import torch
import sys
import json
import pickle
from datetime import datetime
from pathlib import Path

from training.grooveTransformer import GrooveTransformer as GT
import grooveEvaluator.relativeComparison as rc
from evaluation.evalDatasets import *
from constants import *

VALIDATION_SOURCE_DIR = f'{PREPROCESSED_DATASETS_DIR}/PreProcessed_On_15_02_2024_at_16_49_hrs'

def evaluateModel(model: GT, validation_set: ValidationHvoDataset, eval_set_size: int, synthesize_up_to: int=0):
    """
    Evaluate the model on a validation data set. Returns the evaluation time for bookkeeping purposes
    """
    eval_set = EvaluationHvoDataset(validation_set, eval_set_size)
    print(f"Selected indices of eval_set: {eval_set.selectedIndices}")

    monotonic_set = MonotonicHvoDataset(eval_set)
    generated_set = GeneratedHvoDataset(monotonic_set, model)
    
    assert synthesize_up_to <= len(eval_set)
    
    if synthesize_up_to > 0:
        for i in range(synthesize_up_to):
            eval_set[i].save_audio(f'{AUDIO_DIR}/eval{i}.wav', sf_path=SF_PATH)
            monotonic_set[i].save_audio(f'{AUDIO_DIR}/monotonic{i}.wav', sf_path=SF_PATH)
            generated_set[i].save_audio(f'{AUDIO_DIR}/generated{i}.wav', sf_path=SF_PATH)
        print(f"Saved {synthesize_up_to} sets of audio files to {AUDIO_DIR}")
    
    comparison_result_by_feat = rc.relative_comparison(generated_set, eval_set)

    evaluation_time = int(datetime.now().timestamp())
    results_filename = f'{EVAL_RESULTS_DIR}/comparison_results_{evaluation_time}'
    selected_indices_filename = f'{EVAL_RESULTS_DIR}/indices_{evaluation_time}'
    
    pickle.dump(comparison_result_by_feat, open(results_filename, 'wb'))
    pickle.dump(eval_set.selectedIndices, open(selected_indices_filename, 'wb'))

    print(f"Pickled relative comparison results. Evaluation time: {evaluation_time}")

    return evaluation_time

def oldLoadModel(modelPath: str, hypersDict: dict) -> GT:
    """
    Loads model from path and hypersDict
    """
    if not os.path.exists(modelPath):
        raise FileNotFoundError(f"Model at {modelPath} does not exist!")

    d_model = hypersDict["d_model"]
    dim_forward = hypersDict["dim_forward"]
    n_heads = hypersDict["n_heads"]
    n_layers = hypersDict["n_layers"]
    
    model = GT(d_model=d_model, nhead = n_heads, dim_feedforward=dim_forward, num_layers=n_layers, voices=9)
    model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
    return model

def getModelPath(start_time,full=True, epochs=100):
    """
    Returns the path to the model
    TODO: update me with whatever's needed to load a model
    """
    size_str = "full" if full else "smol"
    return f'{MODELS_DIR}/{size_str}_{epochs}e_{start_time}t.pth'


def loadModel(model_path: Path) -> GT:
    """
    Loads model from its path
    """
    hyperparams = model_path.name.split('_')

    d_model = int(hyperparams[0])
    dim_forward = int(hyperparams[1])
    n_heads = int(hyperparams[2])
    n_layers = int(hyperparams[3])
    pitches = int(hyperparams[4])
    
    model = GT(d_model=d_model, nhead=n_heads, dim_feedforward=dim_forward, num_layers=n_layers, voices=pitches)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    return model

def clearDir(dir):
    """
    Clears a directory
    """
    for filename in os.listdir(dir):
        filePath = os.path.join(dir, filename)
        try:
            os.remove(filePath)
        except Exception as e:
            print(f"Something went wrong when removing {filePath}.")
            continue   

def usage_and_exit():
    print("Usage: python evaluation.py <modelStartTime> <hypersFile> <synthesize_up_to> <eval_set_size>")
    print("modelStartTime: used to id the model")
    print("hypersFile: json file containing hyperparameters")
    print("synthesize_up_to: number of audiofiles that will be synthesized")
    print("eval_set_size: len of eval set (a subset of validation set)")
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        usage_and_exit()
    
    clearDir(AUDIO_DIR)
    
    modelStartTime = int(sys.argv[1])
    hypersFile = sys.argv[2]
    synthesize_up_to = int(sys.argv[3])
    eval_set_size = int(sys.argv[4])
    hypersPath = f'{HYPERS_DIR}/{hypersFile}'

    validation_set = ValidationHvoDataset(source_dir=VALIDATION_SOURCE_DIR)

    with open(hypersPath) as hp:
        hypersDict = json.load(hp)
        modelPath = getModelPath(modelStartTime)
        model = oldLoadModel(modelPath, hypersDict)
        evaluateModel(model, validation_set, eval_set_size, synthesize_up_to)