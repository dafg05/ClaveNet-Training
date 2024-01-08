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
        self.bce_loss = nn.BCELoss(reduction='none') 
        self.mse_loss = nn.MSELoss(reduction='none')
        self.penalty = penalty

    def forward(self, pred, target):
        hits, velocities, offsets = torch.chunk(pred, 3, dim=2)
        target_hits, target_velocities, target_offsets = torch.chunk(target, 3, dim=2)

        # calculate loss for each (x,y) pair
        hits_loss = self.bce_loss(hits, target_hits)
        velocities_loss = self.mse_loss(velocities, target_velocities)
        offsets_loss = self.mse_loss(offsets, target_offsets)

        # apply penalty to elements that corresponds to non-hits (value of 0.0) 
        # on the target-hit tensor
        penalty_tensor = self.getPenaltyTensor(target_hits)
        hits_loss = torch.mul(penalty_tensor, hits_loss)
        velocities_loss = torch.mul(penalty_tensor, velocities_loss)
        offsets_loss = torch.mul(penalty_tensor, offsets_loss)

        # return mean of all losses
        return {
            "hits_loss": torch.mean(hits_loss),
            "velocities_loss": torch.mean(velocities_loss),
            "offsets_loss": torch.mean(offsets_loss)
        }
    
    def getPenaltyTensor(self, target_hits):
        return torch.where(target_hits == 1.0, 1.0, self.penalty)
    

if __name__ == "__main__":
    # test HVO_Loss
    pred = torch.tensor([[[1.0, 0.0, 1.0, 0.0, 1.0 , 0.0 ], [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]]])
    target = torch.tensor([[[0.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]]])
    loss_fn = HVO_Loss(penalty=0.1)
    loss = loss_fn(pred, target)
    print(f"loss: {loss}")