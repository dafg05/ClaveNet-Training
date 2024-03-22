import torch
import numpy as np

from torch import nn
from training.hvoLoss import HVO_Loss, getHitAccuracy

SEED = 0
STEPS = 32
PITCHES = 9

PENALTY = 0.5

def _paper_calculate_loss(prediction, y, bce_fn, mse_fn, hit_loss_penalty):
    y_h, y_v, y_o = torch.split(y, int(y.shape[2] / 3), 2)  # split in voices
    pred_h, pred_v, pred_o = prediction

    hit_loss_penalty_mat = torch.where(y_h == 1, float(1), float(hit_loss_penalty))

    bce_h = bce_fn(pred_h, y_h) * hit_loss_penalty_mat  # batch, time steps, voices
    bce_h_sum_voices = torch.sum(bce_h, dim=2)  # batch, time_steps
    bce_hits = bce_h_sum_voices.mean()

    mse_v = mse_fn(pred_v, y_v) * hit_loss_penalty_mat  # batch, time steps, voices
    mse_v_sum_voices = torch.sum(mse_v, dim=2)  # batch, time_steps
    mse_velocities = mse_v_sum_voices.mean()
                    
    mse_o = mse_fn(pred_o, y_o) * hit_loss_penalty_mat
    mse_o_sum_voices = torch.sum(mse_o, dim=2)
    mse_offsets = mse_o_sum_voices.mean()

    total_loss = bce_hits + mse_velocities + mse_offsets

    _h = torch.sigmoid(pred_h)
    h = torch.where(_h > 0.5, 1, 0)  # batch=64, timesteps=32, n_voices=9

    h_flat = torch.reshape(h, (h.shape[0], -1))
    y_h_flat = torch.reshape(y_h, (y_h.shape[0], -1))
    n_hits = h_flat.shape[-1]
    hit_accuracy = (torch.eq(h_flat, y_h_flat).sum(axis=-1) / n_hits).mean()

    hit_perplexity = torch.exp(bce_hits)

    return total_loss, hit_accuracy.item(), hit_perplexity.item(), bce_hits.item(), mse_velocities.item(), mse_offsets.item()

def paper_calculate_loss(pred, target):
    pred = torch.from_numpy(pred)
    pred = torch.chunk(pred, 3, dim=2)

    target = torch.from_numpy(target)

    bce_fn = nn.BCEWithLogitsLoss(reduction='none')
    mse_fn = nn.MSELoss(reduction='none')
    
    x = _paper_calculate_loss(prediction=pred, y=target, bce_fn=bce_fn, mse_fn=mse_fn, hit_loss_penalty=PENALTY)
    total_loss = x[0]
    hits_loss = x[3]
    return total_loss, hits_loss

def paper_calculate_accuracy(pred, target):
    pred = torch.from_numpy(pred)
    pred = torch.chunk(pred, 3, dim=2)

    target = torch.from_numpy(target)

    bce_fn = nn.BCEWithLogitsLoss(reduction='none')
    mse_fn = nn.MSELoss(reduction='none')

    _, hit_accuracy, _, _, _, _ = _paper_calculate_loss(prediction=pred, y=target, bce_fn=bce_fn, mse_fn=mse_fn, hit_loss_penalty=PENALTY)
    return hit_accuracy

def my_calculate_loss(pred, target):
    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)

    loss_fn = HVO_Loss(penalty=PENALTY)
    loss_dict = loss_fn(pred, target)

    total_loss = sum(loss_dict.values())
    hits_loss = loss_dict["hits_loss"]
    return total_loss.item(), hits_loss

def my_calculate_accuracy(pred, target):
    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)

    h_pred = pred[:, :, :PITCHES]
    h_target = target[:, :, :PITCHES]

    return getHitAccuracy(h_pred, h_target)

def generateRandomHVOArray(rng):
    # compute random hits
    hits = rng.integers(low=0, high=2, size=(STEPS, PITCHES))

    # compute random velocities and offsets
    velocities = rng.random((STEPS, PITCHES))
    offsets = rng.random((STEPS, PITCHES))
    
    hvo = np.concatenate((hits, velocities, offsets), axis=1, dtype=np.float32)

    # concatenate as hvo arrays
    assert hvo.shape == (STEPS, PITCHES * 3), f"Hvo shape is invalid: {hvo.shape}"
    return packageArray(hvo)

def packageArray(npArray):
    container = []
    container.append(npArray)
    
    packaged = np.asarray(container)
    assert len(npArray.shape) == len(packaged.shape) - 1, f'Packaging should have added a dimension to array. npArray: {npArray.shape}, packaged: {packaged.shape}'

    return packaged

def compareLosses(pred, target):
    paper_total_loss, paper_hits_loss = paper_calculate_loss(pred, target)
    my_total_loss, my_hits_loss  = my_calculate_loss(pred, target)

    if paper_total_loss == my_total_loss and paper_hits_loss == my_hits_loss:
        print("SUCCESS!!! Losses are equal.")
        print(f"Paper total loss: {paper_total_loss}, My total Loss: {my_total_loss}, Paper hits loss: {paper_hits_loss}, My hits loss: {my_hits_loss}")
    else:
        print("Losses are not equal.")
        print(f"Paper total loss: {paper_total_loss}, My total Loss: {my_total_loss}, Paper hits loss: {paper_hits_loss}, My hits loss: {my_hits_loss}")

def compareAccuracies(pred, target):
    paper_accuracy = paper_calculate_accuracy(pred, target)
    my_accuracy = my_calculate_accuracy(pred, target)

    if paper_accuracy == my_accuracy:
        print("SUCCESS!!! Accuracies are equal.")
        print(f"Paper accuracy: {paper_accuracy}, My accuracy: {my_accuracy}")
    else:
        print("Accuracies are not equal.")
        print(f"Paper accuracy: {paper_accuracy}, My accuracy: {my_accuracy}")

if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth = 1000)
    rng = np.random.default_rng(seed=SEED)

    for i in range(10):
        pred = generateRandomHVOArray(rng)
        target = generateRandomHVOArray(rng)

        # compareLosses(pred, target)
        compareAccuracies(pred, target)
    