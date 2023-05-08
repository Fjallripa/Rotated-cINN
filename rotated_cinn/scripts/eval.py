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
from modules.loss import loss


# Parameters
model_path = path.package_directory + "/trained_models/normalized_data.pt"
analysis_path = path.package_directory + "/analysis/normalized_data"
path.makedir(analysis_path)
device = 'cuda'  if torch.cuda.is_available() else  'cpu'
random_seed = 1

train_domains = [-23, 0, 23, 45, 90, 180]
test_domains = [-135, -90, -45, 10, 30, 60, 75, 135]
samples_per_domain = 100
number_of_copies = 3   # number of samples displayed for each domain and class in the visual comparision

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

    plt.savefig(analysis_path + "/domain_bar_plot.png")
    plt.show()


def get_per_domain_loss(domains:list[int], dataset:RotatedMNIST, model:Rotated_cINN, number:int) -> dict[list]:
    loss_info = {'angles':domains, 'mean':[], 'err':[]}
    for domain in domains:
        domain_indices = torch.argwhere(domain == dataset.domain_labels)
        used_indices = domain_indices[:number].squeeze(1)
        
        data = dataset.data[used_indices]
        targets = dataset.targets[used_indices]
        z, log_j = model(data, targets)

        mean, err = loss('max_likelihood', z, log_j)
        loss_info['mean'].append(float(mean))
        loss_info['err'].append(float(err))

    return loss_info



# Main code
if __name__ == "__main__":
    torch.no_grad()
    
    # Load trained model
    cinn = Rotated_cINN().to(device)
    state_dict = {k:v for k,v in torch.load(model_path).items() if 'tmp_var' not in k}
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



    # Compare test set images to generated ones

    ## Generate images from cinn
    grid_shape = (len(train_domains), len(train_set.classes), number_of_copies)

    ### create cond tensor    
    conditions = torch.zeros((*grid_shape, 12))   # for each image, the external condition for the cINN needs to be created
    domains_sincos = RotatedMNIST._deg2sincos(train_domains)
    conditions[:, :, :, :2] = domains_sincos[:, None, None, :]
    classes_onehot = torch.eye(10)
    conditions[:, :, :, 2:] = classes_onehot[None, :, None, :]  

    ### cinn reverse
    latent_tensor = torch.randn((*grid_shape, 28 * 28))
    generated_images = torch.zeros([*grid_shape, 28, 28])
    for d in range(grid_shape[0]):   # domains
        for c in range(grid_shape[1]):   # classes
            images, gradients = cinn.reverse(latent_tensor[d, c], conditions[d, c])
            images = train_set.unnormalize(images)   # If train_set.normalized=False, this won't do anything.
            generated_images[d, c] = images.squeeze(1).cpu().detach()   # remove batch dimension


    ## sample from train_set
        # find enough samples for each domain-class pair
    all_domain_indices = []
    for domain in train_domains:
        domain_indices = torch.argwhere(domain == train_set.domain_labels)
        all_domain_indices.append(domain_indices)
    
    all_class_indices = []
    for class_label in train_set.classes:
        class_indices = torch.argwhere(class_label == train_set.class_labels)
        all_class_indices.append(class_indices)
    
    all_domain_class_indices = torch.zeros(grid_shape, dtype=int)
    for i in range(grid_shape[0]):   # domains
        for j in range(grid_shape[1]):   # classes
            class_mask = np.isin(all_domain_indices[i], all_class_indices[j])
            domain_class_indices = all_domain_indices[i][class_mask]
            all_domain_class_indices[i, j] = domain_class_indices[:grid_shape[2]]

    sampled_images = train_set.data[all_domain_class_indices]


    ## Display sampled and generated images side by side
    ### Reshape images for better display
        # [6, 10, 5, 28, 28] -> [6, 5, 28, 10, 28] -> [6, 140, 10, 28] -> [6, 140, 280]
    generated_image_grid = generated_images.movedim(1, 3).flatten(1, 2).flatten(2, 3)
    sampled_image_grid = sampled_images.movedim(1, 3).flatten(1, 2).flatten(2, 3)
    generated_image_grid = generated_image_grid.clamp(0, 1)

    ### Set plotting parameters
    image_scaling = 0.5
    subfigure_size = np.array([(2*grid_shape[1] + 2),   # 2x classes + middle column
                            (grid_shape[2] + 1)],    # rows + title
                        dtype=float) * image_scaling  
    figure_size = subfigure_size * np.array([1, grid_shape[0]], dtype=float)

    ### Plot visualizations, one domain after another
    fig = plt.figure(figsize=figure_size, layout='constrained')
    fig.suptitle("Visual Comparison between train set and generated images", fontsize=15)
    subfigs = fig.subfigures(grid_shape[0], 1)
    
    for d in range(grid_shape[0]):   # "for d in domains"
        figd = subfigs[d]
        figd.suptitle(f"{train_domains[d]}°")
        sub = figd.subplots(1, 2)

        sub[0].set_title("training set samples")
        sub[0].imshow(sampled_image_grid[d], cmap='gray')
        sub[0].axis('off')

        sub[1].set_title("Rotated_cINN samples")
        sub[1].imshow(generated_image_grid[d], cmap='gray')
        sub[1].axis('off')
    fig.savefig(analysis_path + "/visual_comparison.png")
      
    plt.show()
