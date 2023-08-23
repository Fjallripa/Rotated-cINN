# Maximum Mean Discrepancy functions
# ==================================

# Imports
import numpy
import torch


# Main Functions
def mean (batch_P:torch.Tensor, batch_Q:torch.Tensor) -> float :
    """
    computes the maximum mean discrepancy MMD(P, Q) of the distributions P and Q by approximating them with a batch of samples from each.
    This one computes the differences between means for each dimension.

    Arguments
    ---------
    batch_P : torch.Tensor, shape=(N, D)
    batch_Q : torch.Tensor, shape=(N, D)

    N = batch size
    D = number of dimensions of each sample

    Returns
    -------
    mmd : float, >=0
    """

    batch_P, batch_Q = batch_P.detach().cpu(), batch_Q.detach().cpu()
    mean_P = torch.mean(batch_P, dim=0)   # shape=(D)
    mean_Q = torch.mean(batch_Q, dim=0)
    
    return torch.sqrt(torch.mean((mean_P - mean_Q)**2)).numpy()



def var (batch_P:torch.Tensor, batch_Q:torch.Tensor) -> float :
    """
    computes the maximum mean discrepancy MMD(P, Q) of the distributions P and Q by approximating them with a batch of samples from each.
    This one computes the differences between variances for each dimension.

    Arguments
    ---------
    batch_P : torch.Tensor, shape=(N, D)
    batch_Q : torch.Tensor, shape=(N, D)

    N = batch size
    D = number of dimensions of each sample

    Returns
    -------
    mmd : float, >=0
    """

    batch_P, batch_Q = batch_P.detach().cpu(), batch_Q.detach().cpu()
    mean_P = torch.mean(batch_P, dim=0)   # shape=(D)
    mean_Q = torch.mean(batch_Q, dim=0)
    squared_difference = torch.mean(batch_P**2 - batch_Q**2, dim=0)   # shape=(D)

    return torch.sqrt(torch.mean((squared_difference - (mean_P**2 - mean_Q**2))**2)).numpy()
