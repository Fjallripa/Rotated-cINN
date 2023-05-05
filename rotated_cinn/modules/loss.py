# Loss functions
# ==============


# Imports
import numpy
import torch


# Parameters
number_of_dimensions = 28 * 28   # the number of pixels of an MNIST image


def loss (type:str, z:torch.Tensor, log_j:torch.Tensor) -> tuple[torch.Tensor] :
    if type == 'mnist_cinn':
        losses = torch.mean(z**2, dim=1) / 2 - log_j / number_of_dimensions
        std, mean = torch.std_mean(losses)


    elif type == 'max_likelihood':
        losses = (1/2 * torch.sum(z**2, dim=1) - log_j) / number_of_dimensions
        std, mean = torch.std_mean(losses)

    else:
        raise KeyError(f"Only the following loss functions are available: ['mnist_cinn', 'max_likelihood']. You used '{type}'.")

    
    mean_err = std / numpy.sqrt(len(z))   # empirical error of the mean
    return mean, mean_err