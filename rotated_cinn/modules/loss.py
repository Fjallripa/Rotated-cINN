# Loss functions
# ==============


# Imports
import numpy
import torch
import torch.distributions as D


# Parameters
number_of_dimensions = 28 * 28   # the number of pixels of an MNIST image


def loss (type:str, z:torch.Tensor, log_j:torch.Tensor) -> tuple[torch.Tensor] :
    if type == 'mnist_cinn':
        # Original loss by cINN paper
        losses = torch.mean(z**2, dim=1) / 2 - log_j / number_of_dimensions
        std, mean = torch.std_mean(losses)
    
    elif type == 'max_likelihood':
        # Same loss but more readable
        losses = (1/2 * torch.sum(z**2, dim=1) - log_j) / number_of_dimensions
        std, mean = torch.std_mean(losses)

    elif type == 'max_likelihood_distribution':
        # Same loss but via distributions. (Test by Lars)
        distribution = D.Normal(0.0, 1.0).expand(z.shape[1:])
        distribution = D.Independent(distribution, 1)
        log_prob = distribution.log_prob(z)
        log_likelihood = log_prob + log_j
        mean = -log_likelihood.mean(dim=0) / number_of_dimensions
        std = log_likelihood.std(dim=0) / number_of_dimensions

    else:
        raise KeyError(f"Only the following loss functions are available:\n['mnist_cinn', 'max_likelihood', 'max_likelihood_distribution]\nYou used '{type}'.")

    
    mean_err = std / numpy.sqrt(len(z))   # empirical error of the mean
    return mean, mean_err