# Model Training
# ==============
# Run this script to train the model.
#
# code adapted from https://github.com/vislearn/conditional_INNs



# Imports
from time import time

import numpy as np
import torch
import torch.nn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms

import path   # adds repo to PATH
from modules.model import Rotated_cINN
from modules.data import RotatedMNIST, AddGaussianNoise
from modules import loss





# Parameters
## Model saving
model_name = "v125_80_epochs"   #! New name for each new training
model_subnet = 'og_two_deeper'
model_init_identity = False
model_path = path.package_directory + f"/trained_models/{model_name}.pt"

device = 'cuda'  if torch.cuda.is_available() else  'cpu'
random_seed = 1


## Loss logging
analysis_path = path.package_directory + f"/analysis/{model_name}"
losses_path = analysis_path + f"/training_losses"
path.makedir(analysis_path)


## Dataset loading or saving
load_dataset = True   # if False, it will be created in place
save_dataset = False
dataset_name = "train_default_biquintic"
dataset_path = path.package_directory + "/datasets"
if not load_dataset:
    domains = [-23, 0, 23, 45, 90, 180]
    #domains = [0]


## Training settings
N_epochs = 80
batch_size = 256
learning_rate = 5e-4
loss_function = loss.neg_loglikelihood





# Main code

## Preparation
t_start = time()
print("")
print(f"Training the model '{model_name}'")
print("")

### Create new model
print("Prep: Create new model")
cinn = Rotated_cINN(init_identity=model_init_identity, 
                    subnet=model_subnet,
                   ).to(device)


### Load or create dataset
if load_dataset:   # Load saved datasets
    print(f"Prep: Loading dataset '{dataset_name}'")
    train_set = torch.load(f"{dataset_path}/{dataset_name}.pt")
    domains = train_set.domains

else:   # Create new datasets
    print(f"Prep: Creating new dataset")
    train_set = RotatedMNIST(
        domains=domains, 
        train=True, 
        seed=random_seed, 
        val_set_size=1000, 
        normalize=True, 
        add_noise=True, 
        transform = AddGaussianNoise(0, 0.08), 
        interpolation='biquintic'
    )

val_data = train_set.val.data.to(device)
val_targets = train_set.val.targets.to(device)


### Save dataset
if save_dataset:
    print(f"Prep: Saving dataset '{dataset_name}'")
    torch.save(train_set, f"{dataset_path}/{dataset_name}.pt")


### Create loader, optimizer and scheduler
print(f"Prep: Set up loader, optimizer and scheduler")
train_loader = DataLoader(train_set, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=4, 
                          pin_memory=True,
                          drop_last=True
                         )
optimizer = torch.optim.Adam(cinn.trainable_parameters, lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_epochs, eta_min=5e-6)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=1/np.sqrt(10))
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 55], gamma=0.1)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

print(f"Prep: time = {time() - t_start:.1f}s")
print("")


## Training
print("Train: Start")

### Log losses
all_losses = {'total':np.zeros(N_epochs), 'validation':np.zeros(N_epochs)}
N_batches = len(train_loader)
display_losses = []


### Display training info
print('Epoch\tBatch/Total \tTime \ttraining loss \tval. loss \tlearning rate')
    # Epoch:         number of walks through the whole training set
    # Total:         number of batches in training set
    # Time:          clock time in minutes since start of training
    # training loss: mean loss over last 50 batches
    # val. loss:     loss over validation set
    # learning rate: current learning rate


### Run training loop
t_start = time()
for epoch in range(N_epochs):
    epoch_losses = {'total':np.zeros(N_batches), 'validation':np.zeros(N_batches//50 + 1)}
    for i, batch in enumerate(train_loader):   # for batches in training set
        # run model forward
        data, targets = batch[0].to(device), batch[1].to(device)
        z, log_j = cinn(data, targets)

        # calculate and log loss
        losses, _ = loss_function(z, log_j)
        display_losses.append(losses.item())
        epoch_losses['total'][i] = losses.item()
        
        # do backprop
        losses.backward()
        torch.nn.utils.clip_grad_norm_(cinn.trainable_parameters, 10.)
        optimizer.step()
        optimizer.zero_grad()

        # print training stats for every 50th batch 
        if not i % 50:
            with torch.no_grad():
                z, log_j = cinn(val_data, val_targets)
                losses_val, _ = loss_function(z, log_j)

            print('{:3d}\t{:5d}/{:d}\t{:6.2f}\t{:8.3f}\t{:8.3f}\t{:.2e}'.format(
                    epoch, i, len(train_loader), 
                    (time() - t_start)/60.,
                    np.mean(display_losses),
                    losses_val.item(),
                    optimizer.param_groups[0]['lr'],
                ), 
                flush=True
            )

            display_losses = []   # reset list
            epoch_losses['validation'][i//50] = losses_val.item()

    # take mean of losses from epoch
    all_losses['total'][epoch] = np.mean(epoch_losses['total'])
    all_losses['validation'][epoch] = np.mean(epoch_losses['validation'])

    # update learning rate
    scheduler.step()   

print("")
print("Training: Complete!")
training_time = (time() - t_start)
print(f"Training: time = {training_time/60:.0f}min {training_time%60:.0f}s  ({training_time/N_epochs:.1f}s/epoch)")
print("")

## Save trained model and losses
print(f"Cleanup: Save folder: \n{path.package_directory}")
print(f"Cleanup: Save model under '.{model_path[len(path.package_directory):]}'")
torch.save(cinn.state_dict(), model_path)

print(f"Cleanup: Save losses under '.{losses_path[len(path.package_directory):]}.npz'")
np.savez(losses_path, **all_losses)

print("")
