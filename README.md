# ClaveNet-Training

This repo  houses the training module for ClaveNet.

## Requirements

1. `$ pip install -r requirements.txt`

Note that some packages are hosted on github repos.

## Processing module

As a prerequisite for training, preprocessed datasets must be processed. The `training.process` module is used as part of a batch training script (see Batch training section). To write a script that utilizes it, see `tests/testProcess.py` as an example.`

### Arguments

#### Preprocessed dataset

Provide a path to a preprocessed dataset. See the [preprocessing repo](https://github.com/dafg05/ClaveNet-Preprocessing) for more info.

#### Out processed dataset dir

Provide a path where output processed datasets will be written to.

### Output

Three separate pickled files with a `.pkl` extension for each partition (test, train, and validation). The pickled files consist of an `HvoPairsDataset` object (see the [architecture repo](https://github.com/dafg05/ClaveNet-Architecture)).

## Training module

The `training.training` is also a part of the batch training script. To write a script that utilizes it, see `tests/testTraining.py` as an example.

### Arguments

#### Hyperparameter settings

Provide a path to a `.json` hyperparameter setting file. Four different model hyperparameter settings can already be found under the `hypers` directory in the form of json files. To create your hyperparam setting, follow the format of those included.

#### Processed dataset

Provide a path to a processed dataset (not a *pre*-processed dataset). See the Processing module section for more info.

#### Out model directory

Provide a path where output models will be written to.

#### Wandb logging

Specify if wandb will be use to track training, in which case a wandb account must be setup beforehand. Otherwise, training trends will be printed to the console. We recommend the use of wandb, as a `.csv` training report --a wandb built-in feature-- is required to evaluate models (see the [evaluation repo](https://github.com/dafg05/ClaveNet-Evaluation))

#### Is smol (or full)

If `is_smol` is set to true, then the model dimension will be set to 8 regardless of the hyperparameter setting, drastically reducing computation. Useful for testing.

### Output

ClaveNet models. The output filenames are structured as follows:

```
<smol/full>_<hyperparameter-setting>_<training-time-long>.pth
```

## Batch training script

To train multiple models sequentially, use the `train_batch.py` script. 

### Prerequisites

1. Move preprocessed datasets to the `preproc_datasets/` dir.

### Usage

``` $ python train_batch.py```

All models will be trained with the same hyperparameter setting, controlled by the constant `HYPERPARAMS_SETTING`. Additionally, you can set wandb tracking and `is_smol` for all models with `LOG_WANDB`and `IS_SMOL`, respectively.
