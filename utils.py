import torch
import numpy as np

def ndarray_to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array, ).float()

def is_valid_hvo(hvo: np.ndarray) -> bool:
    """
    Check for nans and infs in hvo
    """
    return (torch.isnan(hvo).any() == False) and (torch.isinf(hvo).any() == False)