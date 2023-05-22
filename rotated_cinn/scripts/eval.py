# Model Evaluation
# ================
# Run this script to see how well the model generalizes to new rotations.


# Imports
import torch
import torch.distributions as distributions
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import path   # adds repo to PATH
from modules.model import Rotated_cINN
from modules.data import RotatedMNIST
from modules import loss



# Parameters
## General Settings
device = 'cuda'  if torch.cuda.is_available() else  'cpu'
random_seed = 1   # For more reproducability

## Loading and saving
### Model loading
model_name = "recreation_with_domains"
model_path = path.package_directory + f"/trained_models/{model_name}.pt"

### Dataset loading or saving
load_saved_datasets = False   # if False, they will be created in place
save_datasets = True
dataset_name = "eval_default"
dataset_path = path.package_directory + "/datasets"
if not load_saved_datasets:
    train_domains = [-23, 0, 23, 45, 90, 180]
    test_domains = [-135, -90, -45, 10, 30, 60, 75, 135]
    #train_domains = [0]
    #test_domains = [-23, 23, 45, 90, 180]

### Saving the evaluation results
save_analysis = True
analysis_path = f"/analysis/{model_name}x"
save_path = path.package_directory + analysis_path
path.makedir(save_path)


## Eval Settings
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
                images = model.reverse(latent_tensor[d, c], conditions[d, c], jac=False)
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
def find_indices_for_each_label(labels:list[int], label_tensor:torch.Tensor) -> torch.Tensor:
    """
    creates a tensor with a row of indices for each label in the label_tensor

    Arguments
    ---------
    labels : list, dtype=int, shape=(L)
    label_tensor: torch.Tensor, dtype=int, shape=(N)
    
    L = number of distinct labels
    N = number of labels
    M = number of the least common label in label_tensor

    Returns
    -------
    label_indices : torch.Tensor, dtype=int(0..N-1), shape=(L, M)
    """ 
    

    indices_for_all_labels = [torch.argwhere(label == label_tensor).squeeze()  for label in labels]   # label indeces grouped by label
    min_label_size = min([int(len(label_indices))  for label_indices in indices_for_all_labels])   # M, min of length of each domain list
    label_indices = torch.stack([label_indices[:min_label_size]  for label_indices in indices_for_all_labels], dim=0)   # shape=(L, M)
    
    return label_indices



def classify_classes(dataset:RotatedMNIST, model:Rotated_cINN, log_progress:bool=False) -> torch.Tensor:
    """
    Arguments
    ---------
    dataset : RotatedMNIST
        training or test set,
        preferrably without data augmentation
    model : Rotated_cINN
        preferrably pre-trained
    log_progress : bool=Fals


    D = number of domains
    C = number of classes
    M = size of the smallest domain in dataset ('batch_size')

    Returns
    -------
    accuracies : torch.Tensor, dtype=float(0..1), shape=(D, C)
        classification accuracies of the model for each domain and class
    """


    # Sort the data set into batches of equal size for each domain
    data, targets = dataset.data, dataset.targets
    domains = dataset.domains   # list of domains as degrees of rotation (int)
    domain_index_batches = find_indices_for_each_label(domains, dataset.domain_labels)   # shape=(D, M)
    data_batches = data[domain_index_batches]   # shape=(D, M, 28, 28)
    target_batches = targets[domain_index_batches]   # shape=(D, M, 12)
    
    D = len(domains)
    C = 10
    M = target_batches.shape[1]
    

    # Calculate the loglikelihoods for each class for images of each domain
    one_hot = torch.eye(C)  # class one-hot vectors
    loglikelihood_batches = torch.zeros((D, C, M))
    for d in range(D):
        for c in dataset.classes:
            # Create targets for each class
            cd_targets = target_batches[d]
            cd_targets[:, 2:] = one_hot[c]
            cd_targets = cd_targets.to(device)
            
            cd_data = data_batches[d]
            cd_data = cd_data.to(device)
            
            # Encode the images of a domain with targets for each class
            z, log_j = model(cd_data, cd_targets)
            z, log_j = z.cpu().detach(), log_j.cpu().detach()
            
            # Calculate the likelihood of an image for each class.
            normal = distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
            normal = distributions.Independent(normal, 1)
            loglikelihood = normal.log_prob(z) + log_j   # log p(x | c, d), shape=(M)
            loglikelihood_batches[d, c] = loglikelihood
            
            if log_progress:
                print(f"\rclassification in progress: domain {d+1:2d}/{D}, class {c+1:2d}/{C}", end="")
                if d+1==D and c+1==C:  print("")   # return at end of loop
        

    # Compute the classification accuracy wrt. classes and domains
    ## Dataset labels and classifier label for each domain
    class_batches  = dataset.class_labels[domain_index_batches]   # shape=(D, M)
    choice_batches = torch.argmax(loglikelihood_batches, dim=1)   # classifier choice for each domain, shape=(D, M)
        # choose the class with the largest loglikelihood (assumes a naive prior of 1/10 per class)

    ## Convert the labels into one-hot vectors (enables class-wise accuracy figures)
    one_hot = torch.eye(C, dtype=bool)
    class_mask = one_hot[class_batches]   # shape=(D, M, C)
    choice_mask = one_hot[choice_batches]   # shape=(D, M, C)
    
    ## Calculate the domain-class accuracy matrix
    agreement_counts = (class_mask & choice_mask).sum(dim=1)   # shape=(D, C)
        # counts for each domain and class how often the classifier choice was the same a the dataset label
    class_counts = class_mask.sum(dim=1)   # shape=(D, C)
        # counts for each domain and class how often that class appeared in the dataset labels
    accuracies = agreement_counts / class_counts   # shape=(D, C)


    return accuracies



def plot_classification_accuracy(accuracies:torch.Tensor, domains:list[int]) -> None:
    """
    Arguments
    ---------
    accuracies : torch.Tensor, dtype=float, shape=(D, C)
        classification accuracies of the model for each domain and class
    domains : list, dtype=int, shape=(D)
        list of domains as degrees of rotation
    
    
    D = number of domains
    C = number of classes


    Source: This function was created in collaboration with chatGPT4.
    """
    

    # Calculating the plotted values
    accuracies = 100 * accuracies.T   # conversion to %, shape=(C, D)
    margin_C = accuracies.mean(dim=1)  # average over domains, get class-wise accuracies, shape=(C)
    margin_D = accuracies.mean(dim=0)  # average over classes, get domain-wise accuracies, shape=(D)
    overall = accuracies.mean()  # overall average, get total classification accuracy, float


    # Formatting the plot
    C, D = accuracies.shape
    labels_C = [f"'{i}'" for i in range(C)]   # class labels
    labels_D = [f"{angle}°" for angle in domains]   # domain labels

    fig, axes = plt.subplots(nrows=2, ncols=2,
                            figsize=(D+1, C+1),   # +1 is for the labels to have space as well
                            gridspec_kw={'height_ratios': [1, C], 
                                        'width_ratios': [1, D],
                                        'wspace': 0.05*(C+1)/(D+1), 
                                        'hspace': 0.05})
    fig.suptitle("Classification accuracy of Rotated cINN in %", y=1.0)
    heatmap_kwargs = {'annot':True, 'fmt':" 4.1f", 'cbar':False, 'vmin':0, 'vmax':100}


    # Plotting the accuracies as heatmaps
    ## Domain- and class-wise accuracy heatmaps
    sns.heatmap(accuracies, ax=axes[1, 1], xticklabels=False, yticklabels=False, **heatmap_kwargs)

    ## Marginal heatmaps
    margin_C_map = sns.heatmap(margin_C[:, None], ax=axes[1, 0], xticklabels=False, yticklabels=labels_C, **heatmap_kwargs)
    margin_D_map = sns.heatmap(margin_D[None, :], ax=axes[0, 1], xticklabels=labels_D, yticklabels=False, **heatmap_kwargs)

    ## Overall average
    overall_map = sns.heatmap(torch.tensor([[overall]]), ax=axes[0, 0], xticklabels=False, yticklabels=False, **heatmap_kwargs)
    overall_map.texts[0].set_weight('bold')
    overall_map.texts[0].set_size(11)


    # Format the labels correctly
    ## Move ticks to the right and top
    axes[1, 0].yaxis.tick_left()
    axes[0, 1].xaxis.tick_top()

    ## Set labels
    axes[1, 0].set_ylabel('classes')
    axes[0, 1].set_xlabel('domains')

    ## Adjust the labels' positions
    axes[1, 0].yaxis.set_label_position('left') 
    axes[0, 1].xaxis.set_label_position('top') 


    plt.show()



## MMD Loss
def calculate_mmd_losses(dataset:RotatedMNIST, model:Rotated_cINN) -> tuple[torch.Tensor]:
    """
    Arguments
    ---------
    dataset : RotatedMNIST
        training or test set,
        preferrably without data augmentation
    model : Rotated_cINN
        preferrably pre-trained

    D = number of domains
    
    Returns
    -------
    """

    D = len(dataset.domains)
    domain_labels = dataset.domain_labels

    for d in range(D):
        indices = torch.argwhere(d == domain_labels).squeeze()
        data = dataset.data[indices]
        targets = dataset.targets[indices]
        
        with torch.zero_
        latent_vectors, _ = model()
    


def plot_mmd_losses(mmd_values:tuple[torch.Tensor]) -> None:
    pass



# Main code
if __name__ == "__main__":
    print(f"Starting the evaluation of the model '{model_name}'")
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
    if load_saved_datasets:
        # Load saved datasets
        train_set = torch.load(f"{dataset_path}/train_set.pt")
        test_set = torch.load(f"{dataset_path}/test_set.pt")
        train_domains = train_set.domains
        test_domains = test_set.domains
        all_domains = sorted(train_domains + test_domains)
    else:
        # Load new datasets
        all_domains = sorted(train_domains + test_domains)
        train_set = RotatedMNIST(domains=train_domains, train=True, seed=random_seed, val_set_size=1000, normalize=True, add_noise=False)
        test_set = RotatedMNIST(domains=all_domains, train=False, seed=random_seed, normalize=True, add_noise=False)
        
    """
    ### Shorten datasets
    new_length = 1000

    train_set.data = train_set.data[:new_length]
    train_set.targets = train_set.targets[:new_length]
    train_set.class_labels = train_set.class_labels[:new_length]
    train_set.domain_labels = train_set.domain_labels[:new_length] 
    
    test_set.data = test_set.data[:new_length]
    test_set.targets = test_set.targets[:new_length]
    test_set.class_labels = test_set.class_labels[:new_length]
    test_set.domain_labels = test_set.domain_labels[:new_length] 
    """

    ## Save datasets
    if save_datasets:
        torch.save(train_set, f"{dataset_path}/train_{dataset_name}.pt")
        torch.save(test_set, f"{dataset_path}/test_{dataset_name}.pt")
    

    # Calculating accuracies
    print("Eval: Using the model as a classifier and plotting its accuracies over classes and domains")
    # Train set
    print("      Training set and training domains")
    accuracies = classify_classes(train_set, cinn, log_progress=True)
    plot_classification_accuracy(accuracies, train_set.domains)

    
    ## Test set
    print("      Test set and all domains")
    accuracies = classify_classes(test_set, cinn, log_progress=True)
    plot_classification_accuracy(accuracies, test_set.domains)
    
    
    # Calculating losses
    print("")
    print("Eval: Creating a domain-wise loss plot")
    
    ## Calculate loss for each domain
    train_domain_loss = get_per_domain_loss(train_domains, train_set, cinn, samples_per_domain)
    test_domain_loss = get_per_domain_loss(all_domains, test_set, cinn, samples_per_domain)

    ## Plot losses
    show_domain_bar_plot(train_domain_loss, test_domain_loss)


    # Compare dataset images to generated ones
    print("Eval: Displaying dataset images next to generated ones")
    ## Train set
    print("      Training set and training domains")
    generated_images = generate_image_grid(train_domains, train_set, cinn)   # Generate images from cinn
    sampled_images = sample_dataset_grid(train_domains, train_set, cinn)     # Sample from dataset
    display_image_grid(sampled_images, generated_images, True, train_domains)   # Display them side by side
    
    ## Test set
    print("      Test set and test domains")
    generated_images = generate_image_grid(test_domains, test_set, cinn)
    sampled_images = sample_dataset_grid(test_domains, test_set, cinn)
    display_image_grid(sampled_images, generated_images, False, test_domains)
