import torch
import wandb
import sys
import os
import json
import numpy as np

from hvoLoss import HVO_Loss
from dataset import GrooveHVODataset
from hvo_processing.hvo_sets import HVOSetRetriever
from grooveTransformer import GrooveTransformerModel
from torch.utils.data import DataLoader
from datetime import datetime
from utils import ndarray_to_tensor, is_valid_hvo

MODELS_DIR = "models"
HYPERS_DIR = "hypers"
PROCESSED_DATA_DIR = "processedData"

DATASETS_DIR = "preprocessedDatasets"
PREPREOCESSED_DATA_DIR = DATASETS_DIR + "/Processed_On_20_01_2024_at_20_38_hrs"

PITCHES = 9
TIME_STEPS = 32
TORCH_SEED = 0

LOG_EVERY = 256
DEBUG = False

# TRAINING AND TESTING LOOPS

def train_loop(dataloader, model, loss_fn, optimizer, grad_clip, epoch):
    """
    From pytorch tutorial: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    """
    model.train() # Set the model to training mode - important for batch normalization and dropout layers
    # with torch.autograd.detect_anomaly():
    for batch, (x, y) in enumerate(dataloader):
        if not (is_valid_hvo(x) and is_valid_hvo(y)):
            raise Exception("Invalid training data! x or y contains nans or infs!")

        # Compute prediction and loss
        pred = model(x)
        loss_dict = loss_fn(pred, y)

        if not (is_valid_hvo(pred)):
            raise Exception("Something went wrong in the model! pred contains nans or infs!")

        loss = sum(loss_dict.values())
        # Backpropagation
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % LOG_EVERY == 0:
            loss, training_sample = loss.item(), batch * len(x)
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

    with torch.no_grad():
        for x, y in dataloader:
            if not (is_valid_hvo(x) and is_valid_hvo(y)):
                raise Exception("Invalid test data! x or y contains nans or infs!")
            pred = model(x)
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
    
def get_dataloader(preprocessed_datadir: str, partition: str, tappify_params: dict, batch_size: int,dev: str):
    hsr = HVOSetRetriever(preprocessed_datadir)
    if partition == "train":
        dataset, metadata = hsr.get_trainset_and_metadata()
    elif partition == "test":
        dataset, metadata = hsr.get_testset_and_metadata()
    elif partition == "validation":
        dataset, metadata = hsr.get_validationset_and_metadata()
    else:
        raise Exception(f"Invalid partition: {partition}")
    
    processedData = GrooveHVODataset(hvo_set=dataset, metadata=metadata, tappify_params=tappify_params, dev=dev)
    return DataLoader(processedData, batch_size=batch_size, shuffle=False)
    
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
            "data_augmentation": data_augmentation
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

    torch_seed = TORCH_SEED
    # load hyperparameters
    with open(hypersPath) as hp:
        hypersDict = json.load(hp)

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
    data_augmentation = hypersDict["data_augmentation"]
    tappify_params = hypersDict["tappify_params"]

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

    # record run startTime
    start_time = int(datetime.now().timestamp())

    # data loading
    print(f"Loading data from {PREPREOCESSED_DATA_DIR}..., using augmentation: {data_augmentation}")
    try:
        partition = "train"
        train_loader = get_dataloader(PREPREOCESSED_DATA_DIR, partition, tappify_params, batch_size, device)
        partition = "test"
        test_loader = get_dataloader(PREPREOCESSED_DATA_DIR, partition, tappify_params, batch_size, device)

        print(f"Processed {len(train_loader.dataset)} train examples and {len(test_loader.dataset)} test examples")
    except Exception as e:
        raise Exception(f"Error loading {partition} data from dir: {PREPREOCESSED_DATA_DIR}: {e}")
    
    # training visualization and logging
    torch.set_printoptions(threshold=10000)
    project_name = "MGT-local"

    if log_wandb:
        wandb_init()

    # init
    model = GrooveTransformerModel(d_model=d_model, nhead=n_heads, num_layers=n_layers, dim_feedforward=dim_forward, dropout=dropout)
    model.to(device)
    loss_fn = HVO_Loss(penalty=loss_penalty)

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
    augSuffix = "aug" if data_augmentation else "reg"
    modelSuffix = f"{augSuffix}_{torch_seed}s_{epochs}e_{start_time}t"
    modelName = f"{model_prefix}_{modelSuffix}"

    torch.save(model.state_dict(), f"{MODELS_DIR}/{modelName}.pth")
    print(f"Saved model {modelName} in {MODELS_DIR}!")