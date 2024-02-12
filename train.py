import torch
import wandb
import sys
import os
import json
import pickle

from hvoLoss import HVO_Loss, getHitAccuracy
from dataset import GrooveHVODataset
from grooveTransformer import GrooveTransformer
from torch.utils.data import DataLoader
from datetime import datetime
from constants import *
from utils import is_valid_hvo

PROCESSED_DATASETS_DIR = "processedDatasets"
PROCESSED_TIME = 1707739887
DATA_DIR = f"{PROCESSED_DATASETS_DIR}/processed_at_{PROCESSED_TIME}"

HIT_SIGMOID_IN_FORWARD = False

LOG_EVERY = 256
DEBUG = False

def train_loop(dataloader, model, loss_fn, optimizer, grad_clip, epoch):
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        if not (is_valid_hvo(x) and is_valid_hvo(y)):
            raise Exception("Invalid training data! x or y contains nans or infs!")

        # Compute prediction and loss
        h, v, o = model(x)
        pred = torch.cat((h, v, o), dim=2)
        loss_dict = loss_fn(pred, y)

        if not (is_valid_hvo(pred)):
            raise Exception("Something went wrong in the model! pred contains nans or infs!")

        total_loss = sum(loss_dict.values())
        # Backpropagation
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % LOG_EVERY == 0:
            total_loss, training_sample = total_loss.item(), batch * len(x)
            training_logging(log_wandb, training_sample, loss_dict, grad_norm, epoch)

def test_loop(dataloader, model, loss_fn, epoch):
    """
    From pytorch tutorial: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    
    For metrics, we use loss and hit_accuracy, where:
    hit_accuracy <- correctly_predicted_hits / total_hits

    TODO:
    - make hit accuracy its own function, and test it
    """
    
    model.eval() # Set the model to evaluation mode - important for batch normalization and dropout layers

    num_batches = len(dataloader)
    total_hits = num_batches * TIME_STEPS * PITCHES * dataloader.batch_size

    test_loss, incorrect_hits = 0, 0

    for x, y in dataloader:
        if not (is_valid_hvo(x) and is_valid_hvo(y)):
            raise Exception("Invalid test data! x or y contains nans or infs!")
        h, v, o = model.inference(x)
        pred = torch.cat((h, v, o), dim=2)
        if not (is_valid_hvo(pred)):
            raise Exception("Something went wrong in the model! pred contains nans or infs!")
        # test_loss += loss_fn(pred, y).item()
        loss_dict = loss_fn(pred, y)
        test_loss += sum(loss_dict.values()).item()
        
        h_pred = pred[:, :, :PITCHES]
        h_y = y[:, :, :PITCHES]
        incorrect_hits += torch.sum(torch.abs(h_pred - h_y))

    assert incorrect_hits.item().is_integer(), "incorrect_hits is not an integer!"
    assert incorrect_hits.item() <= total_hits, "incorrect_hits is greater than total_hits!"

    test_loss /= num_batches
    hit_accuracy = 1 - (incorrect_hits / total_hits)

    test_logging(log_wandb, test_loss, hit_accuracy, epoch)

# HELPERS

def get_dataloader(data_dir, partition, batch_size, device):
    path = f'{data_dir}/{partition}.pkl'
    with open(path, 'rb') as file:
        content = pickle.load(file)

        inputs = content["inputs"]
        outputs = content["outputs"]

        dataset = GrooveHVODataset(inputs=inputs, outputs=outputs, dev=device)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    
def get_aug_params(data_dir):
    # open dataAugParams.json file at data_dir, return as dict
    with open(f"{data_dir}/{DATA_AUG_PARAMS}") as file:
        return json.load(file)

def format_aug_params(jsonDict: dict):
    """
    If the params file contains all the necessary keys, return a tuple of the values. Otherwise, return None
    """
    try:
        augSeed = jsonDict["seed"]
        seedExamplesSet = jsonDict["seedExamplesSet"]
        numTransformations = jsonDict["numTransformations"]
        numReplacements = jsonDict["numReplacements"]
        preferredStyle = jsonDict["preferredStyle"]
        outOfStyleProb = jsonDict["outOfStyleProb"]
        return augSeed, seedExamplesSet, numTransformations, numReplacements, preferredStyle, outOfStyleProb
    except KeyError as e:
        print(f"DataAugParam {e} not found in jsonDict!")
        return None, None, None, None, None, None
    
def wandb_init():
    wandb.login() 
    run = wandb.init(
        # Set the project where this run will be logged
        project=project_name,
        # Track hyperparameters and run metadata
        config={
            "start_time" : start_time,
            "torch_seed": torch_seed,
            "epochs": epochs,
            "traning_data_size": len(train_loader.dataset),
            "test_data_size": len(test_loader.dataset),
            "batch_size": batch_size,
            "d_model": d_model,
            "dim_forward": dim_forward,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "loss_penalty": loss_penalty,
            "grad_clip": grad_clip,
            "data_processed_time": processed_time,
            "hit_sigmoid_in_forward": HIT_SIGMOID_IN_FORWARD,
            "augmentation_seed": augSeed,
            "seed_examples_set": seedExamplesSet,
            "num_transformations": numTransformations,
            "num_replacements": numReplacements,
            "preferred_style": preferredStyle,
            "out_of_style_prob": outOfStyleProb,
    })

    # plotting test loss
    wandb.define_metric("epoch")
    wandb.define_metric("test_loss", step_metric="epoch")
    wandb.define_metric("hit_accuracy", step_metric="epoch")

def training_logging(logWandb, training_sample, loss_dict, grad_norm, epoch):
    training_loss = sum(loss_dict.values())
    if logWandb:
        wandb.log({"training_sample": training_sample})
        wandb.log({"training_loss": training_loss})
        wandb.log({"hits_loss": loss_dict["hits_loss"]})
        wandb.log({"velocities_loss": loss_dict["velocities_loss"]})
        wandb.log({"offsets_loss": loss_dict["offsets_loss"]})
        wandb.log({"grad_norm": grad_norm})
        wandb.log({"epoch": epoch})
    else:
        print("-----------------------------------")
        print(f"training_sample: {training_sample:>5d}")
        print(f"training_loss: {training_loss:>7f}")
        print(f"hits_loss: {loss_dict['hits_loss']:>7f}")
        print(f"velocities_loss: {loss_dict['velocities_loss']:>7f}")
        print(f"offsets_loss: {loss_dict['offsets_loss']:>7f}")
        print(f"grad_norm: {grad_norm:>7f}")
        print(f"epoch: {epoch:>7f}")
        print("-----------------------------------")

def test_logging(logWandb, test_loss, hit_accuracy, epoch):
    if logWandb:
        wandb.log({"test_loss": test_loss, "epoch": epoch})
        wandb.log({"hit_accuracy": hit_accuracy, "epoch": epoch})
    else:
        print("-----------------------------------")
        print(f"test_loss: {test_loss:>7f} epoch: {epoch:>7f}")
        print(f"hit_accuracy: {hit_accuracy:>7f} epoch: {epoch:>7f}")
        print("-----------------------------------")

# USAGE

def usage():
    print("Usage: python train.py <size> <log_location> <paramFileName")
    print("smol: 'smol' or 'full'")
    print("log_location: 'wandb' or 'local'")
    print("paramFilename: 'file name that contains hyperparameters")

# MAIN

if __name__ == "__main__":
    if len(sys.argv) < 4:
        usage()
        sys.exit(1)
    if sys.argv[1] == "smol":
        smol = True
    elif sys.argv[1] == "full":
        smol = False
    else:
        usage()
        sys.exit(1)

    if sys.argv[2] == "wandb":
        log_wandb = True
    elif sys.argv[2] == "local":
        log_wandb = False
    else:
        usage()
        sys.exit(1)

    hypersPath = f"{HYPERS_DIR}/{sys.argv[3]}"
    if not os.path.exists(hypersPath):
        print(f"Error: {hypersPath} does not exist!")
        sys.exit(1)

    # some metadata
    torch_seed = TORCH_SEED
    processed_time = PROCESSED_TIME

    # load hyperparameters
    with open(hypersPath) as hp:
        hypersDict = json.load(hp)

    # TODO: make object for hyperparameters
    batch_size = hypersDict["batch_size"]
    d_model = 8 if smol else hypersDict["d_model"]
    dim_forward = hypersDict["dim_forward"]
    n_heads = hypersDict["n_heads"]
    n_layers = hypersDict["n_layers"]
    dropout = hypersDict["dropout"]
    learning_rate = hypersDict["learning_rate"]
    loss_penalty = hypersDict["loss_penalty"] # applied to velocites and offset values that occur in non-hit locations
    grad_clip = hypersDict["grad_clip"]
    epochs = 1 if smol else hypersDict["epochs"]

    # load data augmentation parameters
    augSeed, seedExamplesSet, numTransformations, numReplacements, preferredStyle, outOfStyleProb = format_aug_params(get_aug_params(DATA_DIR))

    # torch options
    torch.manual_seed(torch_seed)
    torch.autograd.set_detect_anomaly(DEBUG)

    # device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # data loading
    print(f"Loading data from {DATA_DIR} using augmentation: {augSeed is not None}")
    try:
        partition = "train"
        train_loader = get_dataloader(data_dir=DATA_DIR, partition=partition, batch_size=batch_size, device=device)
        partition = "test"
        test_loader = get_dataloader(data_dir=DATA_DIR, partition=partition, batch_size=batch_size, device=device)

        print(f"Loaded {len(train_loader.dataset)} train examples and {len(test_loader.dataset)} test examples")
    except Exception as e:
        raise Exception(f"Error loading {partition} data from dir: {DATA_DIR}: {e}")
    
    # training visualization and logging
    torch.set_printoptions(threshold=10000)
    project_name = "MGT-local"

     # record run startTime
    start_time = int(datetime.now().timestamp())

    if log_wandb:
        wandb_init()

    # init
    model = GrooveTransformer(d_model=d_model, nhead = n_heads, num_layers=n_layers, dim_feedforward=dim_forward, dropout=dropout, hit_sigmoid_in_forward=HIT_SIGMOID_IN_FORWARD)
    model.to(device)
    loss_fn = HVO_Loss(penalty=loss_penalty)

    # TODO: can we use Adam?
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # loop
    for e in range(epochs):
        epoch = e + 1
        print(f"Epoch {epoch}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, grad_clip, epoch)
        test_loop(test_loader, model, loss_fn, epoch)
        print("-------------------------------")


    # save model
    print("Saving model...")

    model_prefix = "smol" if smol else "full"
    modelSuffix = f"{epochs}e_{start_time}t"
    modelName = f"{model_prefix}_{modelSuffix}"

    torch.save(model.state_dict(), f"{MODELS_DIR}/{modelName}.pth")
    print(f"Saved model {modelName} in {MODELS_DIR}!")