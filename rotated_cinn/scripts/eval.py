# Model Evaluation
# ================
# Run this script to see how well the model generalizes to new rotations.


# Imports
import torch
import torch.distributions as D
import matplotlib.pyplot as plt
import numpy as np

import path   # adds repo to PATH
from modules.model import Rotated_cINN
from modules.data import RotatedMNIST
from modules import loss



# Parameters
## General Settings
name = "recreation_with_domains"   #! New name for each new training

model_path = path.package_directory + f"/trained_models/{name}.pt"
analysis_path = f"/analysis/{name}x"
save_path = path.package_directory + analysis_path
path.makedir(save_path)
device = 'cuda'  if torch.cuda.is_available() else  'cpu'
random_seed = 1   # For more reproducability

## Eval Settings
train_domains = [-23, 0, 23, 45, 90, 180]
test_domains = [-135, -90, -45, 10, 30, 60, 75, 135]
#train_domains = [0]
#test_domains = [-23, 23, 45, 90, 180]
loss_function = loss.neg_loglikelihood

samples_per_domain = 100
number_of_copies = 5   # number of samples displayed for each domain and class in the visual comparision





# Main functions
## Plotting losses
def show_domain_bar_plot(train_loss:dict[list], test_loss:dict[list]) -> None:

    fig, ax = plt.subplots(layout='constrained')

    # Spots for the bars on the x-axis
    test_spots = np.arange(len(test_loss['angles']))
    train_spots = np.nonzero(np.in1d(test_loss['angles'], train_loss['angles']))[0]
    width = 1/3  # the width of the bars
    
    # Train and test bars with error
    ax.bar(train_spots - width/2, train_loss['mean'], width=width, label='train')
    ax.errorbar(train_spots - width/2, train_loss['mean'], train_loss['err'], fmt=',', ecolor='black')
    ax.bar(test_spots + width/2, test_loss['mean'], width=width, label='test')
    ax.errorbar(test_spots + width/2, test_loss['mean'], test_loss['err'], fmt=',', ecolor='black')

    # Title, labels, etc.
    ax.set_title('MaxLikelihood loss of Rotated cINN')
    ax.set_xticks(test_spots, labels=test_loss['angles'])
    ax.legend(loc='upper center', ncols=2)

    # Saving the plot
    save_name = "/domain_bar_plot.png"
    print(f"    Saving plot as {save_name}")
    plt.savefig(save_path + save_name)
    plt.show()


def get_per_domain_loss(domains:list[int], dataset:RotatedMNIST, model:Rotated_cINN, number:int) -> dict[list]:
    loss_info = {'angles':domains, 'mean':[], 'err':[]}
    for domain in domains:
        domain_indices = torch.argwhere(domain == dataset.domain_labels)
        used_indices = domain_indices[:number].squeeze(1)
        
        data = dataset.data[used_indices].to(device)
        targets = dataset.targets[used_indices].to(device)
        z, log_j = model(data, targets)
        
        mean, err = loss_function(z, log_j)
        loss_info['mean'].append(float(mean))
        loss_info['err'].append(float(err))

    return loss_info



## Compare dataset images to generated ones
def generate_image_grid(domains:list[int], dataset:RotatedMNIST, model:Rotated_cINN) -> torch.Tensor:
    grid_shape = (len(domains), len(dataset.classes), number_of_copies)

    # create conditions tensor    
    conditions = torch.zeros((*grid_shape, 12)).to(device)   # for each image, the external condition for the cINN needs to be created
    domains_sincos = RotatedMNIST._deg2sincos(domains)
    conditions[..., :2] = domains_sincos[:, None, None, :]
    classes_onehot = torch.eye(10)
    conditions[..., 2:] = classes_onehot[None, :, None, :]


    # Generate images from the cINN
    latent_tensor = torch.randn((*grid_shape, 28 * 28)).to(device)
    generated_images = torch.zeros([*grid_shape, 28, 28]).cpu()
    for d in range(grid_shape[0]):   # domains
        for c in range(grid_shape[1]):   # classes
            with torch.no_grad():
                images, _ = model.reverse(latent_tensor[d, c], conditions[d, c])
                images = dataset.unnormalize(images)   # If dataset.normalized=False, this won't do anything.
                generated_images[d, c] = images.squeeze(1).cpu().detach()   # remove batch dimension
    
    return generated_images


def sample_dataset_grid(domains:list[int], dataset:RotatedMNIST, model:Rotated_cINN) -> torch.Tensor:
    grid_shape = (len(domains), len(dataset.classes), number_of_copies)

    all_domain_indices = []
    for domain in domains:
        domain_indices = torch.argwhere(domain == dataset.domain_labels)
        all_domain_indices.append(domain_indices)
    
    all_class_indices = []
    for class_label in dataset.classes:
        class_indices = torch.argwhere(class_label == dataset.class_labels)
        all_class_indices.append(class_indices)
    
    # Finding dataset indices for each domain-class pair
    all_domain_class_indices = torch.zeros(grid_shape, dtype=int)
    for i in range(grid_shape[0]):   # domains
        for j in range(grid_shape[1]):   # classes
            class_mask = np.isin(all_domain_indices[i], all_class_indices[j])
            domain_class_indices = all_domain_indices[i][class_mask]
            all_domain_class_indices[i, j] = domain_class_indices[:grid_shape[2]]

    # Sampling images for a grid of those pairs
    return dataset.data[all_domain_class_indices]


def display_image_grid(sampled_images:torch.Tensor, generated_images:torch.Tensor, train:bool, domains:list[int]) -> None:
    # Reshape images for better display
        # e.g. [6, 10, 5, 28, 28] -> [6, 5, 28, 10, 28] -> [6, 140, 10, 28] -> [6, 140, 280]
    generated_image_grid = generated_images.movedim(1, 3).flatten(1, 2).flatten(2, 3)
    sampled_image_grid = sampled_images.movedim(1, 3).flatten(1, 2).flatten(2, 3)
    generated_image_grid = generated_image_grid.clamp(0, 1)

    # Set plotting parameters
    grid_shape = sampled_images.shape[:-2]
    image_scaling = 0.5
    subfigure_size = np.array([(2*grid_shape[1] + 2),   # 2x classes + middle column
                            (grid_shape[2] + 1)],    # rows + title
                        dtype=float) * image_scaling  
    figure_size = subfigure_size * np.array([1, grid_shape[0]], dtype=float)

    # Plot visualizations, one domain after another
    fig = plt.figure(figsize=figure_size, layout='constrained')
    fig.suptitle("Visual Comparison between train set and generated images", fontsize=15)
    subfigs = fig.subfigures(grid_shape[0], 1)
    
    for d in range(grid_shape[0]):   # "for d in domains"
        figd = subfigs[d]  if grid_shape[0] > 1 else  subfigs
        figd.suptitle(f"{domains[d]}°")
        sub = figd.subplots(1, 2)

        sub[0].set_title("training set samples")
        sub[0].imshow(sampled_image_grid[d], cmap='gray')
        sub[0].axis('off')

        sub[1].set_title("Rotated_cINN samples")
        sub[1].imshow(generated_image_grid[d], cmap='gray')
        sub[1].axis('off')
    
    # Saving the plot
    suffix = 'train'  if train else  'test'
    save_name = f"/visual_comparison_{suffix}.png"
    print(f"    Saving plot as {save_name}")
    plt.savefig(save_path + save_name)
    plt.show()



## Classification
def classify_classes(domains:list[int], dataset:RotatedMNIST, model:Rotated_cINN):
    
    # Encode dataset
    data = dataset.data.to(device)
    targets = dataset.data.to(device)
    
    #all_domain_indices = 
    for domain in domains:
        domain_indices = torch.argwhere(domain == dataset.domain_labels)

    ## Create targets for each class
    one_hot = torch.eye(10).to(device)  # class one-hot vectors
    for c in dataset.classes:
        c_targets = targets[:, 2:] = one_hot[c]
        z, log_j = model(data, targets)


    normal = D.Normal(0.0, 1.0).expand(z.shape[-1])
    normal = D.Independent(normal, 1)
    likelihood = torch.exp(normal.log_prob(z) + log_j)
    
    class_prob = 0.1 * torch.ones(10)   # the naive class prior
    image_prob = torch.sum(likelihood * class_prob, )





# Main code
if __name__ == "__main__":
    print(f"Starting the evaluation of the model '{name}'")
    print(f"    Save location: ...{analysis_path}")
    print("")


    # Preparation
    ## Load trained model
    print("Prep: Loading model")
    cinn = Rotated_cINN().to(device)
    state_dict = {k:v for k,v in torch.load(model_path).items() if 'tmp_var' not in k}
    cinn.load_state_dict(state_dict)
    cinn.eval()

    ## Load datasets
    print("Prep: Loading training and test datasets")
    all_domains = sorted(train_domains + test_domains)
    train_set = RotatedMNIST(domains=train_domains, train=True, seed=random_seed, val_set_size=1000, normalize=True, add_noise=True)
    test_set = RotatedMNIST(domains=all_domains, train=False, seed=random_seed, normalize=True, add_noise=False)
    
    
    # Calculating losses
    print("")
    print("Eval: Creating a domain-wise loss plot")
    
    ## Calculate loss for each domain
    train_domain_loss = get_per_domain_loss(train_domains, train_set, cinn, samples_per_domain)
    test_domain_loss = get_per_domain_loss(all_domains, test_set, cinn, samples_per_domain)

    ## Plot losses
    show_domain_bar_plot(train_domain_loss, test_domain_loss)


    # Compare dataset images to generated ones
    ## Train set
    print("Eval: Displaying training set images next to generated ones")
    generated_images = generate_image_grid(train_domains, train_set, cinn)   # Generate images from cinn
    sampled_images = sample_dataset_grid(train_domains, train_set, cinn)     # Sample from dataset
    display_image_grid(sampled_images, generated_images, True, train_domains)   # Display them side by side
    
    ## Test set
    print("Eval: Displaying test set images next to generated ones")
    generated_images = generate_image_grid(test_domains, test_set, cinn)
    sampled_images = sample_dataset_grid(test_domains, test_set, cinn)
    display_image_grid(sampled_images, generated_images, False, test_domains)