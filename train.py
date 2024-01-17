import torch
import wandb
import sys
import numpy as np

from hvoLoss import HVO_Loss
from dataset import GrooveHVODataset
from grooveTransformer import GrooveTransformerModel
from torch.utils.data import DataLoader
from datetime import datetime
from utils import ndarray_to_tensor, is_valid_hvo

MODELS_DIR = "models"
SMOL_DATA_DIR = "testData"
PROCESSED_DATA_DIR = "processedData"

MODEL_DIMENSION = 512
PITCHES = 9
TIME_STEPS = 32

LOG_EVERY = 64
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
    
def wandb_init():
    wandb.login() 
    run = wandb.init(
        # Set the project where this run will be logged
        project=project_name,
        # Track hyperparameters and run metadata
        config={
        "torch_seed": torch_seed,
        "learning_rate": learning_rate,
        "grad_clip": grad_clip,
        "batch_size": batch_size,
        "epochs": epochs,
        "d_model": d_model,
        "n_head": n_head,
        "num_layers": num_layers,
        "loss_penalty": loss_penalty,
        "traning_data_size": len(train_loader.dataset),
        "test_data_size": len(test_loader.dataset),
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
    print("Usage: python train.py <size> <log_location> [data_aug]")
    print("smol: 'smol' or 'full'")
    print("log_location: 'wandb' or 'local'")
    print("data_aug: 'aug'")

# MAIN

if __name__ == "__main__":

    if len(sys.argv) < 3:
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

    aug = False
    if len(sys.argv) > 3:
        if sys.argv[3] == "aug":
            aug = True
        else:
            usage()
            sys.exit(1)

    # hyperparameters
    torch_seed = 0
    learning_rate = 1e-3
    grad_clip = 5.0
    batch_size = 4 if smol else 64
    epochs = 2 if smol else 100
    d_model = 8 if smol else 512 
    n_head = 4
    num_layers = 6
    loss_penalty = 0.1 # applied to velocites and offset values that occur in non-hit locations
    data_augmentation = aug

    # torch options
    torch.manual_seed(torch_seed)
    torch.autograd.set_detect_anomaly(DEBUG)

    # data loading
    data_dir = SMOL_DATA_DIR if smol else PROCESSED_DATA_DIR
    augSuffix = "_aug" if data_augmentation else ""
    print(f"Loading data from {data_dir}..., using augmentation: {data_augmentation}")
    try:
        training_name = "training_aug" if data_augmentation else "training" 
        training_data = GrooveHVODataset(f"{data_dir}/{training_name}.pkl", transform=ndarray_to_tensor, target_transform=ndarray_to_tensor)
        train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
    except Exception as e:
        raise Exception(f"Error loading training data from data_dir: {data_dir}. {e}")

    try:
        test_name = "test_aug" if data_augmentation else "test"
        test_data = GrooveHVODataset(f"{data_dir}/{test_name}.pkl", transform=ndarray_to_tensor, target_transform=ndarray_to_tensor)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    except Exception as e:
        raise Exception(f"Error loading test data from data_dir: {data_dir}. {e}")
    
    # training visualization and wandb
    torch.set_printoptions(threshold=10000)
    project_name = "MGT-local"

    if log_wandb:
        wandb_init()

    # init
    model = GrooveTransformerModel(d_model=d_model, nhead=n_head, num_layers=num_layers)
    loss_fn = HVO_Loss(penalty=loss_penalty)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # loop
    for e in range(epochs):
        epoch = e + 1
        print(f"Epoch {epoch}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, grad_clip, epoch)
        test_loop(test_loader, model, loss_fn, epoch)
        print("-------------------------------")

    print("Saving model...")


    model_prefix = "smol" if smol else "full"
    augSuffix = "aug" if data_augmentation else "reg"
    modelSuffix = f"{augSuffix}_{torch_seed}s_{epochs}e_{int(datetime.now().timestamp())}t"
    modelName = f"{model_prefix}_{modelSuffix}"


    torch.save(model.state_dict(), f"{MODELS_DIR}/{modelName}.pth")
    print(f"Saved model {modelName} in {MODELS_DIR}!")