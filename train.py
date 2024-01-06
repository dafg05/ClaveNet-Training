import torch
import wandb
import sys

from hvoLoss import HVO_Loss
from constants import *
from dataset import GrooveHVODataset
from grooveTransformer import GrooveTransformerModel
from torch.utils.data import DataLoader

MODELS_DIR = "models"
SMOL_DATA_DIR = "processedData/smol"
PROCESSED_DATA_DIR = "processedData"

MODEL_DIMENSION = 512
PITCHES = 9
TIME_STEPS = 32

LOG_EVERY = 1
log_wandb = False
smol = True

# TRAINING AND TESTING LOOPS

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    """
    From pytorch tutorial: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

    TODO:
    - is loss working?
    - epochs?
    """
    model.train() # Set the model to training mode - important for batch normalization and dropout layers
    
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % LOG_EVERY == 0:
            loss, current_sample = loss.item(), batch * len(x)
            if log_wandb:
                wandb.log({"training_loss": loss, "current_sample": current_sample})
            else:
                print(f"training_loss: {loss:>7f}, current_sample: {current_sample:>5d}")

def test_loop(dataloader, model, loss_fn, epoch):
    """
    From pytorch tutorial: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    
    For metrics, we use loss and hit_accuracy, where:
    hit_accuracy <- correctly_predicted_hits / total_hits

    TODO:
    - test hit accuracy with different batch sizes and numbers of batches
    """
    
    model.eval() # Set the model to evaluation mode - important for batch normalization and dropout layers

    num_batches = len(dataloader)
    total_hits = num_batches * TIME_STEPS * PITCHES * dataloader.batch_size

    test_loss, incorrect_hits = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            
            h_pred = pred[:, :, :PITCHES]
            h_y = y[:, :, :PITCHES]
            incorrect_hits += torch.sum(torch.abs(h_pred - h_y))

    assert incorrect_hits.item().is_integer(), "incorrect_hits is not an integer!"
    assert incorrect_hits.item() <= total_hits, "incorrect_hits is greater than total_hits!"

    test_loss /= num_batches
    hit_accuracy = 1 - (incorrect_hits / total_hits)

    if log_wandb:
        wandb.log({"test_loss": test_loss, "hit_accuracy": hit_accuracy})
    else:
        print(f"test_loss: {test_loss:>7f} hit_accuracy: {hit_accuracy:>7f}")

# HELPERS

def ndarray_to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array, ).float()

# USAGE

def usage():
    print("Usage: python train.py <SMOL> <LOG_WANDB>")
    print("SMOL: smol or full")
    print("LOG_WANDB: wandb or local")

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

    # training hyperparameters
    learning_rate = 1e-3
    batch_size = 2 if smol else 64
    epochs = 1 

    # model hyperparameters
    d_model = 8 if smol else 512 
    n_head = 4
    num_layers = 6

    # loss hyperparameters
    penalty = 0.1

    # wandb project
    project_name = "MGT-local"

    # data
    data_dir = SMOL_DATA_DIR if smol else PROCESSED_DATA_DIR
    try:
        training_data = GrooveHVODataset(f"{data_dir}/training.pkl", transform=ndarray_to_tensor, target_transform=ndarray_to_tensor)
        train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    except Exception as e:
        raise Exception(f"Error loading training data from data_dir: {data_dir}. {e}")

    try:
        test_data = GrooveHVODataset(f"{data_dir}/test.pkl", transform=ndarray_to_tensor, target_transform=ndarray_to_tensor)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    except Exception as e:
        raise Exception(f"Error loading test data from data_dir: {data_dir}. {e}")
    
    # training visualization
    torch.set_printoptions(threshold=10000)
    if log_wandb:
        wandb.login() 

        run = wandb.init(
            # Set the project where this run will be logged
            project=project_name,
            # Track hyperparameters and run metadata
            config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "d_model": d_model,
            "n_head": n_head,
            "num_layers": num_layers,
            "traning_data_size": len(train_loader.dataset),
            "test_data_size": len(test_loader.dataset)
        })

        wandb.define_metric("current_sample")
        wandb.define_metric("training_loss", step_metric="current_sample") 

    model = GrooveTransformerModel(d_model=d_model, nhead=n_head, num_layers=num_layers)
    loss_fn = HVO_Loss(penalty=penalty)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loop(train_loader, model, loss_fn, optimizer, 1)
    test_loop(test_loader, model, loss_fn, 1)

    print("Saving model...")
    torch.save(model.state_dict(), f"{MODELS_DIR}/smol_model.pth")
    print("Done!")
