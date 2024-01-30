import torch
import numpy as np

from torch import nn

class HVO_Loss(nn.Module):
    """
    Calculate loss between predicted and target HVO grooves.
    Uses bce_loss for hits and mse_loss for velocities and offsets.
    Note the penalty applied to elements that correspond to non-hits in the target HVO groove.
    """
    def __init__(self, penalty=1.0):
        super().__init__()
        # For both loss functions, don't reduce loss
        # since we want to apply penalty to non-hit elements

        # TODO: In paper, they just mention BCE loss, but in code they use BCEWithLogitsLoss
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none') 
        self.mse_loss = nn.MSELoss(reduction='none')
        self.penalty = penalty

    def forward(self, pred, target):
        """
        Loss algorithm: apply bce_loss to hits, mse_loss to velocities and offsets
        Multiply by penalty tensor to apply penalty to non-hit elements
        Sum across the pitches dimension.
        Finally, take the mean of each resulting tensor.
        TODO: hits loss overpowers velocities and offsets loss
        """
        hits, velocities, offsets = torch.chunk(pred, 3, dim=2)
        target_hits, target_velocities, target_offsets = torch.chunk(target, 3, dim=2)

        bce_hits = self.bce_loss(hits, target_hits)
        mse_velocities = self.mse_loss(velocities, target_velocities)
        mse_offsets = self.mse_loss(offsets, target_offsets)

        penalty_tensor = self.getPenaltyTensor(target_hits)
        bce_hits = torch.mul(penalty_tensor, bce_hits)
        mse_velocities = torch.mul(penalty_tensor, mse_velocities)
        mse_offsets = torch.mul(penalty_tensor, mse_offsets)

        # TODO: why are we summing across pitches dimension? Not mentioned in paper
        bce_hits = torch.sum(bce_hits, dim=2)
        mse_velocities = torch.sum(mse_velocities, dim=2)
        mse_offsets = torch.sum(mse_offsets, dim=2)

        bce_hits = torch.mean(bce_hits)
        mse_velocities = torch.mean(mse_velocities)
        mse_offsets = torch.mean(mse_offsets)

        return {
            "hits_loss": bce_hits,
            "velocities_loss": mse_velocities,
            "offsets_loss": mse_offsets
        }
    
    def getPenaltyTensor(self, target_hits):
        return torch.where(target_hits == 1.0, 1.0, self.penalty)
    

def getHitAccuracy(pred_hits, target_hits):
    """
    Calculate hit accuracy between predicted and target HVO grooves.
    TODO: accuracies are slightly different from paper, probably due to floating point precision
    """
    flattened_pred = torch.flatten(pred_hits).int()
    flattened_target = torch.flatten(target_hits).int()
    incorrect_hits = torch.sum(torch.abs(flattened_pred - flattened_target))
    hit_accuracy = 1 - (incorrect_hits / flattened_pred.shape[0])
    return hit_accuracy.item()

if __name__ == "__main__":
    pred = torch.tensor([[[1.0, 0.0, 1.0, 0.0, 1.0 , 0.0 ], [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]]])
    target = torch.tensor([[[0.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]]])

    loss_fn = HVO_Loss(penalty=0.1)
    loss = loss_fn(pred, target)
    print(f"loss: {loss}")