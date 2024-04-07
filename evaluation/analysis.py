from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

TRAIN_RUNS_DIR = Path("train_runs")
ORIGIN_VECTOR = np.array([0, 1]) # [kl_divergence, overlapping_area]

def analysis(eval_runs_paths: List[Path], report_path: Path, n: int):
    best_models_dict = get_best_models(eval_runs_paths, n)
    analysis_data = {
        "model_name": [],
        "validation_loss": [],
        "num_transformations": [],
        "num_replacements": [],
        "out_of_style_prob": [],
    }

    report_df = pd.read_csv(report_path, index_col="training_start_time")
    
    for eval_run_path in best_models_dict.keys():
        model_name = get_model_name_from_eval_run(eval_run_path)
        analysis_data["model_name"].append(model_name)
        analysis_data["validation_loss"].append(best_models_dict[eval_run_path])

        training_start_time = int(model_name.split("_")[-1])
        num_transformations = report_df.loc[training_start_time, "num_transformations"]
        num_replacements = report_df.loc[training_start_time, "num_replacements"]
        out_of_style_prob = report_df.loc[training_start_time, "out_of_style_prob"]

        analysis_data["num_transformations"].append(num_transformations)
        analysis_data["num_replacements"].append(num_replacements)
        analysis_data["out_of_style_prob"].append(out_of_style_prob)
    
    return analysis_data

def get_best_models(eval_runs_paths, n):
    model_validation_losses = {}
    for eval_run_path in eval_runs_paths:
        model_score = get_validation_loss(eval_run_path)
        model_validation_losses[eval_run_path] = model_score
    return get_min_loss_entries(model_validation_losses, n)
    
def get_min_loss_entries(losses_dict, n):
    # Sort the dictionary by its values (ascending order) and select the first n items
    min_loss_entries = sorted(losses_dict.items(), key=lambda item: item[1])[:n]
    # Convert the sorted list of tuples back to a dictionary
    return dict(min_loss_entries)

def get_validation_loss(eval_run_path, positive_kld=True):
    eval_results = pd.read_csv(eval_run_path / "results.csv", index_col="feature")
    validation_loss = 0
    # iterate over the rows of the dataframe
    for _, row in eval_results.iterrows():
        # our metrics vector is going to be [kl_divergence, overlapping_area]
        # retrieve these vales from the dataframe
        metrics_vector = np.array([row["kl_divergence"], row["overlapping_area"]])
        if positive_kld and metrics_vector[0] < 0:
            # if we want to consider only positive kl_divergence values, set negative values to 0
            metrics_vector[0] = 0
        # calculate the euclidean distance between the metrics vector and the origin vector
        distance = np.linalg.norm(metrics_vector - ORIGIN_VECTOR)
        # add the distance to the validation loss
        validation_loss += distance
    return validation_loss

def get_model_name_from_eval_run(eval_run_path):
    # TEST ME FIRST
    run_name = eval_run_path.stem

    return "_".join(run_name.split("_")[2:])