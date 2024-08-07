import torch
import json
import pickle
import pandas as pd

from datetime import datetime
from pathlib import Path
from typing import Dict, List

from cnarch.grooveTransformer import GrooveTransformer as GT

from evaluation.evalDatasets import *
from evaluation.constants import *

import grooveEvaluator.relativeComparison as rc

NUM_POINTS = 10000

def evaluateModel(out_dir: Path, model_path: Path, validation_set_path: Path, simple=False, num_points: int=NUM_POINTS):
    """
    Evaluate the model on a validation data set. Returns the evaluation time for bookkeeping purposes
    """

    model = loadModel(model_path)
    validation_set = ValidationHvoDataset(validation_set_path)

    # Initiazlize the datasets
    monotonic_set = MonotonicHvoDataset(validation_set)
    generated_set = GeneratedHvoDataset(monotonic_set, model)
    
    # Perform relative comparison
    comparison_result_by_feat = rc.relative_comparison(generated_set, validation_set, simple=simple, num_points=num_points, padding_factor=2)

    # Create a directory to store the evaluation results
    evaluation_time = int(datetime.now().timestamp())
    evaluation_dir = Path(out_dir) / f'{model_path.stem}_evaluation_{evaluation_time}'
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Save the relative comparison results
    csv_path = evaluation_dir / 'results.csv'
    if simple:
        simple_results_dict_to_csv(comparison_result_by_feat, csv_path)
        pickle.dump(comparison_result_by_feat, open(evaluation_dir / 'results.pkl', 'wb'))

    else:
        results_dict_to_csv(comparison_result_by_feat, csv_path)

        results_path = evaluation_dir / 'results.pkl'
        pickle.dump(comparison_result_by_feat, open(results_path, 'wb'))

    return evaluation_dir

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

def audioEval(out_dir: Path, model_path: Path, full_validation_set: ValidationHvoDataset, selected_indices: List[int]):
    subset = [full_validation_set[ix] for ix in selected_indices]
    model = loadModel(model_path)

    validation_subset = ValidationHvoDataset(None, subset=subset)
    monotonic_subset = MonotonicHvoDataset(validation_subset)
    generated_subset = GeneratedHvoDataset(monotonic_subset, model)

    for i in range(len(validation_subset)):
        validation_subset[i].save_audio(f'{out_dir}/sample{i}_validation.wav', sf_path=SF_PATH)
        monotonic_subset[i].save_audio(f'{out_dir}/sample{i}_monotonic.wav', sf_path=SF_PATH)
        generated_subset[i].save_audio(f'{out_dir}/sample{i}_generated.wav', sf_path=SF_PATH)


def simple_results_dict_to_csv(results_dict: Dict[str, rc.SimpleComparisonResult], csv_file_path: Path):
    """
    Although this function is named simple_results_dict_to_csv, it actually saves more information than 'results_dict_to_csv'.
    This is for convenience sake, as it's useful to have the mean and std of the intraset and interset distances directly on the csv file.
    """
    data = {
        'feature' : [],
        'kl_divergence' : [],
        'overlapping_area' : [],
        'gen_intraset_mean' : [],
        'eval_intraset_mean' : [],
        'interset_mean' : [],
        'gen_intraset_std' : [],
        'eval_intraset_std' : [],
        'interset_std' : [],
        'min_point' : [],
        'max_point' : [],
        'num_points' : []
    }
    for feature, scr in results_dict.items():
        data['feature'].append(feature)
        data['kl_divergence'].append(scr.kl_divergence)
        data['overlapping_area'].append(scr.overlapping_area)
        data['min_point'].append(scr.points[0])
        data['max_point'].append(scr.points[-1])
        data['num_points'].append(len(scr.points))

        gen_intraset_dict = scr.stats_dict[rc.GENERATED_INTRASET_KEY]
        eval_intraset_dict = scr.stats_dict[rc.VALIDATION_INTRASET_KEY]
        interset_dict = scr.stats_dict[rc.INTERSET_KEY]

        data['gen_intraset_mean'].append(gen_intraset_dict[rc.MEAN_KEY])
        data['eval_intraset_mean'].append(eval_intraset_dict[rc.MEAN_KEY])
        data['interset_mean'].append(interset_dict[rc.MEAN_KEY])
        data['gen_intraset_std'].append(gen_intraset_dict[rc.STD_KEY])
        data['eval_intraset_std'].append(eval_intraset_dict[rc.STD_KEY])
        data['interset_std'].append(interset_dict[rc.STD_KEY])

    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False)

    print(f"Saved results to {csv_file_path}")
    

def results_dict_to_csv(results_dict: Dict[str, rc.ComparisonResult], csv_file_path: Path):
    data = {
        'feature' : [],
        'kl_divergence' : [],
        'overlapping_area' : [],
        'min_point' : [],
        'max_point' : [],
        'num_points' : []
    }

    for feature, cr in results_dict.items():
        data['feature'].append(feature)
        data['kl_divergence'].append(cr.kl_divergence)
        data['overlapping_area'].append(cr.overlapping_area)
        data['min_point'].append(cr.points[0])
        data['max_point'].append(cr.points[-1])
        data['num_points'].append(len(cr.points))

    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False)

    print(f"Saved results to {csv_file_path}")
