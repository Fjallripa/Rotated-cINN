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
device = 'cpu'
#device = 'cuda'  if torch.cuda.is_available() else  'cpu'
random_seed = 1   # For more reproducability

## Loading and saving
### Model loading
model_name = "v125_80_epochs"
model_subnet = 'og_two_deeper'
model_path = path.package_directory + f"/trained_models/{model_name}.pt"

### Dataset loading or saving
load_saved_datasets = True   # if False, they will be created in place
save_datasets = False
dataset_name = "eval_default_biquintic"
dataset_path = path.package_directory + "/datasets"
if not load_saved_datasets:
    train_domains = [-23, 0, 23, 45, 90, 180]
    test_domains = [-135, -90, -45, 10, 30, 60, 75, 135]
    #train_domains = [0]
    #test_domains = [0]
    #test_domains = [-23, 23, 45, 90, 180]

### Saving the evaluation results
save_analysis = True
analysis_path = f"/analysis/{model_name}"
save_path = path.package_directory + analysis_path
path.makedir(save_path)


## Eval Settings
loss_function = loss.neg_loglikelihood

samples_per_domain = 100
number_of_copies = 5   # number of samples displayed for each domain and class in the visual comparision





# Main functions
## Plotting losses
def plot_training_losses() -> None:
    loss_path = save_path + "/training_losses.npz"
    if not path.os.path.exists(loss_path):
        print("    no training losses found.")
        return
    
    # Loading the losses
    losses = dict(np.load(loss_path))
    N_epochs = len(list(losses.items())[0][1])  # length of one of the loss arrays = number of epochs trained
    epochs = np.arange(N_epochs) + 1

    # Plotting the losses
    plt.title("Training losses")
    ymax = -np.inf
    ymin = np.inf
    for loss_name, loss_array in losses.items():
        plt.plot(epochs, loss_array, label=loss_name)
        ymax = max(ymax, np.max(loss_array[1:]))   
            # the loss of the first epoch is an outlier and compresses the plotted range of the subsequent epochs to much. It is therefor ignored for setting the uppe limit of the plot.
        ymin = min(ymin, np.min(loss_array[1:]))
    plt.ylim((ymin-0.1, ymax+0.1))
    plt.xlabel("epoch")
    plt.legend()

    # Saving the plot
    save_name = "/training_losses.png"
    print(f"    Saving plot as {save_name}")
    plt.savefig(save_path + save_name)
    plt.show()


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
    domains_sincos = RotatedMNIST.deg2cossin(domains)
    conditions[..., :2] = domains_sincos[:, None, None, :]
    classes_onehot = torch.eye(10)
    conditions[..., 2:] = classes_onehot[None, :, None, :]


    # Generate images from the cINN
    latent_tensor = torch.randn((*grid_shape, 28 * 28)).to(device)
    generated_images = torch.zeros([*grid_shape, 28, 28]).cpu()
    for d in range(grid_shape[0]):   # domains
        for c in range(grid_shape[1]):   # classes
            with torch.no_grad():
                images, _ = model.reverse(latent_tensor[d, c], conditions[d, c], jac=False)
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
                print(f"\r    classification in progress: domain {d+1:2d}/{D}, class {c+1:2d}/{C}", end="")
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



def plot_classification_accuracy(accuracies:torch.Tensor, domains:list[int], train:bool) -> None:
    """
    Arguments
    ---------
    accuracies : torch.Tensor, dtype=float, shape=(D, C)
        classification accuracies of the model for each domain and class
    domains : list, dtype=int, shape=(D)
        list of domains as degrees of rotation
    train : bool
        if True, save as '...train.png',
        else as '...test.png'
    
    
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

    # Saving the plot
    suffix = 'train'  if train else  'test'
    save_name = f"/classification_{suffix}.png"
    print(f"    Saving plot as {save_name}")
    plt.savefig(save_path + save_name)
    plt.show()




## MMD Loss
def calculate_mmd_losses(dataset:RotatedMNIST, model:Rotated_cINN, mmd_funcs:list) -> dict[tuple[float, np.ndarray]]:
    """
    calculates MMD values for multiple loss.mmd functions, both for the whole dataset and domain-wise.

    Arguments
    ---------
    dataset : RotatedMNIST
        training or test set,
        preferrably without data augmentation
    model : Rotated_cINN
        preferrably pre-trained
    mmd_funcs : list of loss.mmd functions

    D = number of domains
    
    Returns
    -------
    all_mmds : dict 
        * each key is the name of one of the loss.mmd functions
        * each value is a tuple of a float and a numpy array (shape=(D), dtype=float).
        * the first float is the MMD value of the whole dataset while the array contains the MMD values for each domain of the dataset.
    """


    D = len(dataset.domains)
    
    # For each domain, find which dataset elements belong to that domain
    domain_labels = dataset.domain_labels
    indices_of_domain = [torch.argwhere(d == domain_labels).squeeze()  for d in dataset.domains]   


    all_mmds = {}
    # Calculate MMD values for each function
    for func in mmd_funcs:
        # Calculate a reference value
        random_vectors_1 = torch.randn((len(dataset)//14, 28*28))
        random_vectors = torch.randn_like(random_vectors_1)
        mmd_reference = func(random_vectors_1, random_vectors)

        # Calculate for the whole dataset
        with torch.no_grad():
            data = dataset.data.to(device)
            targets = dataset.targets.to(device)
            
            latent_vectors, _ = model.forward(data, targets, jac=False)
            random_vectors = torch.randn_like(latent_vectors)
        
            mmd_global = func(latent_vectors, random_vectors)

        # Calculate for each domain
        mmd_domain = np.zeros(D)
        for d in range(D):
            with torch.no_grad():
                data = dataset.data[indices_of_domain[d]].to(device)
                targets = dataset.targets[indices_of_domain[d]].to(device)
                
                latent_vectors, _ = model.forward(data, targets, jac=False)
                random_vectors = torch.randn_like(latent_vectors)
            
                mmd_domain[d] = func(latent_vectors, random_vectors)

        # Store the results in a dictionary
        all_mmds[func.__name__] = (mmd_global, mmd_domain, mmd_reference)


    return all_mmds

    

def plot_mmd_losses(mmd_values:dict[tuple[float, np.ndarray]], train_domains:list[int], test_domains:list[int]) -> None:
    """
    presents the `mmd_values` for each domain as one bar plot for each function of `mmd_values`. 

    Arguments
    ---------
    mmd_values : dict
        * each key is the name of one of the F loss.mmd functions
        * each value is a tuple of a float and a numpy array (shape=(D), dtype=float).
        * the first float is the MMD value of the whole dataset while the array contains the MMD values for each domain of the dataset.
    train_domains: list of ints (rotation angles)
    test_domains: list (same)

    F = number of functions
    D = length of training and test domains combined
    """
    

    # Create x-axis labels ('global' + each domain) and indices for train and test domains
    domains_sorted = sorted(train_domains + test_domains)
    train_indices = np.array([domains_sorted.index(d)  for d in train_domains])
    test_indices = np.array([domains_sorted.index(d)  for d in test_domains])
    x_labels = ["global"] + [f"{angle}°"  for angle in domains_sorted]
    
    F = len(mmd_values.keys())
    D = len(domains_sorted)


    # Plot a bar plot for the MMD values of each function over all domains
    plt.figure(figsize=(6, 4*F), constrained_layout=True)
    plt.suptitle("Maximum Mean Discrepancy between\nthe cINN latent space and the normal distribution")
    
    for f, func in enumerate(mmd_values.keys()):
        plt.subplot(F, 1, f+1)
        mmd_global = [mmd_values[func][0]]   # [float]
        mmd_train = mmd_values[func][1][train_indices]
        mmd_test = mmd_values[func][1][test_indices]
        mmd_reference = mmd_values[func][2]

        plt.title(f"mmd.{func}()")
        plt.bar([0], mmd_global, color='black', label="whole dataset")
        plt.bar(train_indices+1, mmd_train, label="training domains")
        plt.bar(test_indices+1, mmd_test, label="test_domains")
        plt.hlines([mmd_reference], xmin=0, xmax=D, colors='grey', linestyles='dashed', label="normal distribution (ideal)")
        plt.ylim(0, 1.1*max(mmd_values[func][1]))  # between 0 and largest bar
        plt.xticks(range(D+1), x_labels, rotation=45)
        plt.xlabel("domains")
        plt.ylabel("MMD values")
        plt.legend()


    # Save the plot
    save_name = f"/mmd_losses.png"
    print(f"    Saving plot as {save_name}")
    plt.savefig(save_path + save_name)
    plt.show()
    



## Domain Transfer
def domain_transfer(model:Rotated_cINN, data:torch.Tensor, targets:torch.Tensor, angles:list):
    """
    Arguments
    ---------
    model : Rotated_cINN
    data : torch.Tensor, shape=(N, H, W)
    targets: torch.Tensor, shape=(N, 12)
    angles : list, dtype= int or float, shape=(N)

    N = number of angles
    H, W = height and width of image

    Returns
    -------
    data_rotated : torch.Tensor, shape=(N, H, W)
    data_reconstructed : torch.Tensor, shape=(N, H, W)
    """

    with torch.no_grad():
        data_rotated = RotatedMNIST.rotate(data, angles, interpolation='biquintic').to(device)
        targets_rotated = torch.cat([RotatedMNIST.deg2cossin(angles).to(device), targets[:, 2:]], dim=1)
        latent_vectors = model.forward(RotatedMNIST._normalize(data), targets)[0]
        data_reconstructed = RotatedMNIST._unnormalize(model.reverse(latent_vectors, targets_rotated)[0].squeeze())

    return data_rotated, data_reconstructed



def numpify(tensor:torch.Tensor) -> np.ndarray :
    return tensor.detach().cpu().numpy()



def visualize_domain_tranfer(model, data:torch.Tensor, targets:torch.Tensor, angles:list):
    """
    this function takes in R dataset images and plots a grid of N * R rotated, reconstructed images.

    Arguments
    ---------
    data : torch.Tensor, shape=(R, H, W)
        dataset images to be reconstruct with different rotations
    targets : torch.Tensor, shape=(R, 12)
        their targets
    angles : list, shape=(N)
        rotation angles / domains to be reconstructed

    R = number of example images
    N = number of angles
    H, W = height and width of image
    """


    N = len(angles)
    R = data.shape[0]
    reconstructed_grid = torch.zeros((N, *data.shape))
    rotated_grid = torch.zeros((N, *data.shape))

    # Calculate the rotations and reconstructions
    for n in range(N):
        angle = [angles[n]] * R   # R times the same angle
        rotated, reconstructed = domain_transfer(model, data, targets, angle)   # reconstructions of all R example images for `angle`.
        reconstructed_grid[n] = reconstructed
        rotated_grid[n] = rotated
    
    # Change shape of the grids to make them presentable in matplotlib: (N, R, H, W) -> (R * H, N * W)
    reconstructed_grid = numpify(reconstructed_grid.moveaxis(0, 2).flatten(2, 3).flatten(0, 1)) 
    rotated_grid = numpify(rotated_grid.moveaxis(0, 2).flatten(2, 3).flatten(0, 1))
    difference_grid = reconstructed_grid - rotated_grid

    
    # Plot the two grids
    ticks = np.arange(N) * 28 + 14
    max_value = 2
    angle_labels = [f"{angle}°" for angle in angles]
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    
    plt.figure(figsize=(N, 2*R))
    plt.suptitle("Examples of Domain Transfer")
    plt.tight_layout()
    
    plt.subplot(2, 1, 1)
    plt.title("Results for the cINN")
    plt.imshow(reconstructed_grid, vmin=-max_value, vmax=max_value)
    plt.xticks(ticks, angle_labels)
    plt.yticks([])
    
    plt.subplot(2, 1, 2)
    plt.title("Difference to rotated orignals")
    plt.imshow(difference_grid, vmin=-max_value, vmax=max_value)
    plt.xticks(ticks, angle_labels)
    plt.yticks([])
    
    
    # Save the plot
    save_name = f"/domain_transfer_examples.png"
    print(f"    Saving plot as {save_name}")
    plt.savefig(save_path + save_name)
    plt.show()   



def plot_transfer_loss(model:Rotated_cINN, data:torch.Tensor, targets:torch.Tensor, interval:int, train_domains=None, test_domains=None):
    """
    Arguments
    ---------
    model : Rotated_cINN
    data : torch.Tensor, shape=(N, H, W)
    targets : torch.Tensor, shape=(N, 12)
    interval : int, number of degrees between each bar


    N = number of images to average over
    A = N // interval, number of angles for whicht to to calculate a loss
    H, W = height and width of image
    """
    
    degree_range = np.array(range(-179, 181, interval))
    A = len(degree_range)
    transfer_losses = np.zeros(A)
    
    for i, degree in enumerate(degree_range):    
        print(f"\r    calculating loss for {degree}°", end=" ")
        degrees = degree * torch.ones(len(data))
        
        data_rotated, data_reconstructed = domain_transfer(model, data, targets, degrees)

        l1_loss = torch.mean(torch.abs(data_rotated - data_reconstructed))
        transfer_losses[i] = l1_loss.detach().cpu().numpy()
    print("\r")


    plt.figure(figsize=(15, 5))
    plt.bar(degree_range, transfer_losses, width=interval)
    if train_domains != None:
        i_train = np.argwhere((degree_range[:, None] == np.array(train_domains)[None]).any(1))[:,0]
        plt.bar(degree_range[i_train], transfer_losses[i_train], width=interval, label='training domains')
    if test_domains != None:
        i_test = np.argwhere((degree_range[:, None] == np.array(test_domains)[None]).any(1))[:,0]
        plt.bar(degree_range[i_test], transfer_losses[i_test], width=interval, color="red", label="test domains")

    plt.legend()
    plt.xlabel("rotation angle [°]")
    plt.ylabel("mean(abs(rotated - reconstructed))")
    plt.title("L1 loss of domain transfer")

    # Save the plot
    save_name = f"/domain_transfer_barplot.png"
    print(f"    Saving plot as {save_name}")
    plt.savefig(save_path + save_name)
    plt.show()
    



## Classification across Domains
def classification_accuracy(model, dataset, angle, sample_size = None) -> float:
    """
    Arguments
    ---------
    model : Rotated_cINN
        preferrably pre-trained
    dataset : RotatedMNIST
        should be unrotated
    angle : int
        how many degrees the dataset shoulb be rotated
    sample_size : None or int
        how many samples of the dataset are used for the accuracy computation

    Returns
    -------
    accuracy : float(0..1)
        fraction of dataset images that the model classified correctly
    """


    N = len(dataset)  if sample_size == None else  sample_size
    C = len(dataset.classes)  # number of classes
    


    # Prepare the dataset by rotating it by `angle` degrees. 
    angles = [angle] * N
    data, targets = dataset.data[:N], dataset.targets[:N]
    data_rotated = RotatedMNIST.rotate(data, angles, interpolation='biquintic').to(device)
    data_rotated = RotatedMNIST._normalize(data_rotated)
    targets_rotated = torch.cat([RotatedMNIST.deg2cossin(angles).to(device), targets[:, 2:]], dim=1)


    # Calculate the loglikelihoods for each class
    loglikelihoods = torch.zeros((C, N))
    for c in dataset.classes:
        print(f"\r   angle: {angle: 4d}°, class: {c}", end="")

        # Create targets for each class
        class_targets = targets_rotated.detach().clone()
        class_targets[:, 2:] = torch.eye(C)[c].to(device)
        
        # Encode the images
        z, log_j = model(data_rotated, class_targets)
        z, log_j = z.cpu().detach(), log_j.cpu().detach()
        
        # Calculate the likelihoods
        normal = distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
        normal = distributions.Independent(normal, 1)
        loglikelihood = normal.log_prob(z) + log_j   # log p(x | c), shape=(N)
        loglikelihoods[c] = loglikelihood
        
        
      
    # Compute the classification accuracy
    choices = torch.argmax(loglikelihoods, dim=0)   # classifier label for each domain, shape=(N)
        # choose the class with the largest loglikelihood (assumes a naive prior of 1/10 per class)
    return torch.mean(choices == dataset.class_labels[:N], dtype=float)



def classification_bar_plot(accuracies=list[float], angles=list[int], train_domains=None, test_domains=None) -> None:
    """
    plots and saves a diagram that shows the overall classification accuracy of for every domain (rotation angle) in `angles`.
    
    Arguments
    ---------
    accuracies : list[float], len=len(angles), dtype=float(0..1)
    angles : list[int]
        all the rotation angles thath the classification accuracy should be tested for.
    train_domains : None or list[int]
        angles to be highlighted as training domains in the plot
    test_domains : None or list[int]
        angles to be highlighted as test domains in the plot
    """ 
    

    # Create bar plot
    angles = np.array(sorted(set(angles)))
    interval = max(1, min(angles[1:] - angles[:-1]))  # minimum angle between domains => bar width
    degree_range = np.array(range(min(angles), max(angles)+1))
    
    plt.figure(figsize=(15, 5))
    
    plt.bar(angles, accuracies * 100, width=interval)   # convert from fraction to %
    ## Highlight accuracies of training and test domains
    if train_domains != None:
        i_train = np.argwhere((degree_range[:, None] == np.array(train_domains)[None]).any(1))[:,0]
        plt.vlines(degree_range[i_train], ymin=0, ymax=100, colors='yellow', label='training domains', linestyles='dashed')
    if test_domains != None:
        i_test = np.argwhere((degree_range[:, None] == np.array(test_domains)[None]).any(1))[:,0]
        plt.vlines(degree_range[i_test], ymin=0, ymax=100, colors='red', label='test domains', linestyles='dashed')

    if train_domains != None or test_domains != None:
        plt.legend()
    plt.xlabel("rotation angle [°]")
    plt.ylabel("[%]")
    plt.ylim(0, 100)
    big_ticks = np.arange(0,110, 10)
    small_ticks = np.arange(0, 100, 10) + 5
    plt.yticks(ticks=big_ticks, labels=big_ticks, minor=False)
    plt.yticks(ticks=small_ticks, minor=True)
    plt.title("total classification accuracy across domains")

    # Save the plot
    save_name = f"/class_accuracy_barplot"
    print(f"    Saving plot as {save_name}")
    torch.save(accuracies, save_path + save_name + ".pt")
    plt.savefig(save_path + save_name + ".png")
    plt.show()





# Main code
if __name__ == "__main__":
    print(f"Starting the evaluation of the model '{model_name}'")
    print(f"    Save location: ...{analysis_path}")
    print("")


    # Preparation
    ## Load trained model
    print("Prep: Loading model")
    cinn = Rotated_cINN(subnet=model_subnet).to(device)
    state_dict = {k:v for k,v in torch.load(model_path, map_location=device).items() if 'tmp_var' not in k}
    cinn.load_state_dict(state_dict)
    cinn.eval()

    ## Load datasets
    if load_saved_datasets:   # Load saved datasets
        print(f"Prep: Loading training and test datasets '{dataset_name}'")
        train_set = torch.load(f"{dataset_path}/train_{dataset_name}.pt")
        test_set = torch.load(f"{dataset_path}/test_{dataset_name}.pt")
        train_domains = train_set.domains
        all_domains = test_set.domains
        test_domains = sorted(set(all_domains) - set(train_domains))   # the domains that are not training domains
    else:   # Create new datasets
        print("Prep: Creating training and test datasets")
        all_domains = sorted(set(train_domains) | set(test_domains))   # all unique angles, sorted
        train_set = RotatedMNIST(domains=train_domains, 
                                 train=True, 
                                 seed=random_seed, 
                                 val_set_size=1000, 
                                 normalize=True, 
                                 add_noise=False, 
                                 interpolation='biquintic')
        test_set = RotatedMNIST(domains=all_domains, 
                                train=False, 
                                seed=random_seed, 
                                normalize=True, 
                                add_noise=False, 
                                interpolation='biquintic')

    ## Save datasets
    if save_datasets:
        print(f"Prep: Saving training and test datasets '{dataset_name}'")
        torch.save(train_set, f"{dataset_path}/train_{dataset_name}.pt")
        torch.save(test_set, f"{dataset_path}/test_{dataset_name}.pt")
    
    '''
    # Calculating losses
    
    ## Show the training losses
    print("")
    print("\nEval: Displaying the training losses")
    plot_training_losses()

    
    ## Calculate loss for each domain
    print("\nEval: Creating a domain-wise loss plot")
    train_domain_loss = get_per_domain_loss(train_domains, train_set, cinn, samples_per_domain)
    test_domain_loss = get_per_domain_loss(all_domains, test_set, cinn, samples_per_domain)

    ## Plot losses
    show_domain_bar_plot(train_domain_loss, test_domain_loss)


    # Compare dataset images to generated ones
    print("\nEval: Displaying dataset images next to generated ones")
    ## Train set
    print("  Training set and training domains")
    generated_images = generate_image_grid(train_domains, train_set, cinn)   # Generate images from cinn
    sampled_images = sample_dataset_grid(train_domains, train_set, cinn)     # Sample from dataset
    display_image_grid(sampled_images, generated_images, True, train_domains)   # Display them side by side
    
    ## Test set
    print("  Test set and test domains")
    generated_images = generate_image_grid(test_domains, test_set, cinn)
    sampled_images = sample_dataset_grid(test_domains, test_set, cinn)
    display_image_grid(sampled_images, generated_images, False, test_domains)


    # Calculate MMD losses
    print("\nEval: Compare the model's latent space to a normal distribution via different MMD functions")
    mmd_values = calculate_mmd_losses(test_set, cinn, mmd_funcs=[loss.mmd.mean, loss.mmd.var])
    plot_mmd_losses(mmd_values, train_domains, test_domains)
    

    # Calculating accuracies
    print("\nEval: Using the model as a classifier and plotting its accuracies over classes and domains")
    ## Train set
    print("  Training set and training domains")
    accuracies = classify_classes(train_set, cinn, log_progress=True)
    plot_classification_accuracy(accuracies, train_set.domains, train=True)
    
    ## Test set
    print("  Test set and all domains")
    accuracies = classify_classes(test_set, cinn, log_progress=True)
    plot_classification_accuracy(accuracies, test_set.domains, train=False)
    
    
    # Domain Transfer
    ## Creating domain transfer examples
    print("\nEval: Domain Transfer: Comparing how well model-rotated images agree with directly rotated images")
    test_mnist = torch.load(f"{dataset_path}/test_eval_og_mnist.pt", map_location=device)
    data, targets, *_ = test_mnist[:100]
    data, targets = data.to(device), targets.to(device)
    angles = list(range(0, 181, 15)) # => N = 13
    R = 10   # number of example images
    
    print("  Creating Domain Transfer example images")
    visualize_domain_tranfer(cinn, data[:R], targets[:R], angles)

    ## Measuring domain transfer loss over all angles
    print("  Calculating the L1 loss for all degrees")
    plot_transfer_loss(cinn, data, targets, 1, train_domains, test_domains)
    '''


    # Classification across Domains
    ## Calculating total classification accuracies for a range of rotation angles
    print("\nEval: Classification Domain Transfer: How good is the classifier for different rotation angles?")
    test_mnist = torch.load(f"{dataset_path}/test_eval_og_mnist.pt", map_location=device)
    angles = list(range(-179, 181, 1)) # => N = 360
    accuracies = torch.tensor([classification_accuracy(cinn, test_mnist, angle, sample_size=1000)  for angle in angles])
    classification_bar_plot(accuracies, angles, train_domains)


    print("")
    