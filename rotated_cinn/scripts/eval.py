# Model Evaluation
# ================
# Run this script to see how well the model generalizes to new rotations.


# Imports
import torch
import matplotlib.pyplot as plt
import numpy as np

import path   # adds repo to PATH
from modules.model import Rotated_cINN
from modules.data import RotatedMNIST


# Parameters
save_path = path.package_directory + '/trained_models/rotated_cinn.pt'
device = 'cuda'  if torch.cuda.is_available() else  'cpu'
random_seed = 1
train_domains = [-23, 0, 23, 45, 90, 180]
test_domains = [-135, -90, -45, 10, 30, 60, 75, 135]
samples_per_domain = 100
ndim_total = 28 * 28


# Main functions
## Plotting losses
def show_domain_bar_plot(train_loss:dict[list], test_loss:dict[list]) -> None:

    fig, ax = plt.subplots(layout='constrained')

    # Spots for the bars on the x-axis
    test_spots = np.arange(len(test_loss['angles']))
    train_mask = np.isin(test_loss['angles'], train_loss['angles'])
    train_spots = test_loss['angles'][train_mask]
    width = 1/3  # the width of the bars
    
    # Train and test bars with error
    ax.bar(train_spots - width/2, train_loss['mean'], width=width, label='train')
    ax.errorbar(train_spots - width/2, train_loss['mean'], train_loss['err'], fmt=',', ecolor='black')
    ax.bar(test_spots + width/2, test_loss['mean'], width=width, label='test')
    ax.errorbar(test_spots + width/2, test_loss['mean'], test_loss['err'], fmt=',', ecolor='black')

    # Title, labels, etc.
    ax.set_title('MaxLikelihood loss of Rotated cINN')
    #ax.set_ylim(0, 1.1 * max(max(train_loss['angles']), max(test_loss['angles'])))
    ax.set_xticks(test_spots, labels=test_loss['angles'])
    ax.legend(loc='upper center', ncols=2)

    plt.show()


### Support functions for show_domain_bar_plot()
def compute_loss(model_output: torch.Tensor) -> tuple[float]:
    z, log_j = model_output
    losses = (z**2).mean(dim=1) / 2 - log_j / ndim_total
    std, mean = torch.std_mean(losses)
    mean_err = std / len(z)

    return (float(mean), float(mean_err))


def get_per_domain_loss(domains:list[int], dataset:RotatedMNIST, model:Rotated_cINN, number:int) -> dict[list]:
    loss_info = {'angles':domains, 'mean':[], 'err':[]}
    for domain in domains:
        domain_indices = torch.argwhere(domain == dataset.domain_labels)
        used_indices = domain_indices[:number].squeeze(1)
        
        data = dataset.data[used_indices]
        targets = dataset.targets[used_indices]
        output = model(data, targets)
        
        mean, err = compute_loss(output)
        loss_info['mean'].append(mean)
        loss_info['err'].append(err)

    return loss_info


## Showing generated images
def show_example_images():
    pass



# Main code
if __name__ == "__main__":
    torch.no_grad()
    
    # Load trained model
    cinn = Rotated_cINN().to(device)
    state_dict = {k:v for k,v in torch.load(save_path).items() if 'tmp_var' not in k}
    cinn.load_state_dict(state_dict)
    cinn.eval()

    # Load datasets
    all_domains = sorted(train_domains + test_domains)
    train_set = RotatedMNIST(domains=train_domains, train=True, seed=random_seed, val_set_size=1000)
    test_set = RotatedMNIST(domains=all_domains, train=False, seed=random_seed)
    
    # Calculate loss for each domain
    train_domain_loss = get_per_domain_loss(train_domains, train_set, cinn, samples_per_domain)
    test_domain_loss = get_per_domain_loss(all_domains, test_set, cinn, samples_per_domain)

    # Plot losses
    show_domain_bar_plot(train_domain_loss, test_domain_loss)


    # Generate images 
    """
    #make readable cond tensor
    ##for i in train_domains
    ##..for j in classes
    ##....[i, j] [*] 5
    number_of_copies = 5
    conditions_readable = torch.zeros((len(train_domains), len(train_set.classes), number_of_copies, 2))
    for i, d in enumerate(train_domains):
        conditions_readable[i, :, :, 0] = d
    for j, c in enumerate(train_set.classes):
        conditions_readable[:, j, :, 1] = c
     
    for i, d in enumerate(train_domains):
        for j, c in enumerate(train_set.classes):
            conditions_readable[i, j] = torch.tensor([d, c])
    
    #convert cond tensor

    """
    
    number_of_copies = 5
    grid_shape = (len(train_domains), len(train_set.classes), number_of_copies)

    #create cond tensor    
    conditions = torch.zeros((*grid_shape, 12))
    for i, d in enumerate(train_domains):
        domains_sincos = train_set._deg2sincos(train_domains)
        conditions[i, :, :, :2] = domains_sincos
    for j in range(10):   # "for j in classes"
        classes_onehot = torch.eye(10)
        conditions[:, j, :, 2:] = classes_onehot[j]
        

    #cinn reverse
    latent_tensor = torch.randn((*grid_shape, 28, 28))
    generated_images = cinn.reverse(latent_tensor, conditions)

    #sample from train_set
    ##find enough samples for each domain-class pair
    all_domain_indices = []
    for domain in train_domains:
        domain_indices = torch.argwhere(domain == train_set.domain_labels)
        all_domain_indices.append(domain_indices)
    
    all_class_indices = []
    for class_label in train_set.classes:
        class_indices = torch.argwhere(class_label == train_set.class_labels)
        all_class_indices.append(class_indices)
    
    all_domain_class_indices = torch.zeros(grid_shape, dtype=int)
    for i in grid_shape[0]:   # domains
        for j in grid_shape[1]:   # classes
            class_mask = np.isin(all_domain_indices[i], all_class_indices[j])
            domain_class_indices = all_domain_indices[i][class_mask]
            all_domain_class_indices[i, j] = domain_class_indices[:grid_shape[2]]

    sampled_images = train_set.data[all_domain_class_indices]        
