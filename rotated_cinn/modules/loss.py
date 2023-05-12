# Loss functions
# ==============


# Imports
import numpy
import torch
import torch.distributions as D


# Parameters
number_of_dimensions = 28 * 28   # the number of pixels of an MNIST image


# Support functions
def mean_and_mean_err(mean:torch.Tensor, std:torch.Tensor, number_of_values:int) -> tuple[torch.Tensor] :
    mean_error = std / numpy.sqrt(number_of_dimensions)
    return mean, mean_error



# Main loss functions
def mnist_cinn (z:torch.Tensor, log_j:torch.Tensor) -> tuple[torch.Tensor] :
    # Original loss by cINN paper
    mean = torch.mean(z**2) / 2 - torch.mean(log_j) / number_of_dimensions
    std = torch.zeros_like(mean)
    
    return mean_and_mean_err(mean, std, len(z))


def neg_loglikelihood (z:torch.Tensor, log_j:torch.Tensor) -> tuple[torch.Tensor] :
    # Same loss but more readable
    losses = (1/2 * torch.sum(z**2, dim=1) - log_j) / number_of_dimensions
    std, mean = torch.std_mean(losses)

    return mean_and_mean_err(mean, std, len(z))


def neg_loglikelihood_distribution (z:torch.Tensor, log_j:torch.Tensor) -> tuple[torch.Tensor] :
    # Same loss but via distributions. (Test by Lars)
    distribution = D.Normal(0.0, 1.0).expand(z.shape[1:])
    distribution = D.Independent(distribution, 1)
    log_prob = distribution.log_prob(z)
    log_likelihood = log_prob + log_j
    mean = -log_likelihood.mean(dim=0) / number_of_dimensions
    std = log_likelihood.std(dim=0) / number_of_dimensions

    return mean_and_mean_err(mean, std, len(z))
