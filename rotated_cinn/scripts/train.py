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
from modules.data import RotatedMNIST
from modules import loss


# Parameters
## Model saving
model_name = "recreation_bilinear"   #! New name for each new training
save_path = path.package_directory + f"/trained_models/{model_name}.pt"

device = 'cuda'  if torch.cuda.is_available() else  'cpu'
random_seed = 1

## Training settings
N_epochs = 60
batch_size = 256
learning_rate = 5e-4

domains = [-23, 0, 23, 45, 90, 180]
#domains = [0]
loss_function = loss.neg_loglikelihood
noise_augmentation = lambda data: data + 0.08 * torch.randn_like(data)

# Main code
## set up model etc.
t_start = time()
print(f"Training the model '{model_name}'\n")
print("Setting up the model, data loader, etc...")

cinn = Rotated_cINN(init_identity=False).to(device)
train_set = RotatedMNIST(domains=domains, 
                         train=True, 
                         seed=random_seed, 
                         val_set_size=1000, 
                         normalize=True, 
                         add_noise=True, 
                         transform = transforms.Lambda(noise_augmentation), 
                         interpolation='bilinear'
                        )
train_loader = DataLoader(train_set, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=4, 
                          pin_memory=True, 
                          drop_last=True
                         )
optimizer = torch.optim.Adam(cinn.trainable_parameters, lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

val_data = train_set.val.data.to(device)
val_targets = train_set.val.targets.to(device)

print(f"Setup time: {time() - t_start:.1f}s")
print("")
print('Epoch\tBatch/Total \tTime \ttraining loss \tval. loss \tlearning rate')
    # Epoch:         number of walks through the whole training set
    # Total:         number of batches in training set
    # Time:          clock time in minutes since start of training
    # training loss: mean loss over last 50 batches
    # val. loss:     loss over validation set
    # learning rate: current learning rate


## run training loop
losses_mean = []
for epoch in range(N_epochs):
    for i, batch in enumerate(train_loader):   # for batches in training set
        # run model forward
        data, targets = batch[0].to(device), batch[1].to(device)
        z, log_j = cinn(data, targets)

        # do backprop
        losses, _ = loss_function(z, log_j)
        losses.backward()
        torch.nn.utils.clip_grad_norm_(cinn.trainable_parameters, 10.)
        losses_mean.append(losses.item())
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
                    np.mean(losses_mean),
                    losses_val.item(),
                    optimizer.param_groups[0]['lr'],
                ), 
                flush=True
            )
            losses_mean = []   # reset list
    
    scheduler.step()   # update learning rate
print("")
print("Training complete!")
print(f"total training time: {(time() - t_start)/60:.0f}min {(time() - t_start)%60:.0f}s")
print("")

## save trained model
torch.save(cinn.state_dict(), save_path)
print("Saved model under")
print(save_path)
