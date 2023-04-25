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

import path   # adds repo to PATH
from modules.model import Rotated_cINN
from modules.data import RotatedMNIST


# Parameters
save_path = path.package_directory + '/trained_models/rotated_cinn.pt'
print(save_path)
device = 'cuda'  if torch.cuda.is_available() else  'cpu'
random_seed = 1
nll_mean = []

N_epochs = 5
batch_size = 256
learning_rate = 5e-4

domains = [-23, 0, 23, 45, 90, 180]
ndim_total = 28 * 28



# Main code
## set up model etc.
t_start = time()
print("Setting up the model, data loader, etc...")

cinn = Rotated_cINN().to(device)
train_set = RotatedMNIST(domains=domains, train=True, seed=random_seed, val_set_size=1000)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                           num_workers=4, pin_memory=True, drop_last=True)
optimizer = torch.optim.Adam(cinn.trainable_parameters, lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

print(f"Setup time: {time() - t_start:.1f} s")
print("")
print('Epoch\tBatch/Total \tTime \tNLL train\tNLL val\tLR')
    # Epoch:    number of walks through the whole training set
    # Total:    number of batches in training set
    # Time:     clock time in minutes since start of training
    # NLL train:    mean loss over last 50 batches
    # NLL val:      loss over validation set
    # LR:       current learning rate


## run training loop
for epoch in range(N_epochs):
    for i, batch in enumerate(train_loader):   # for batches in training set
        # run model forward
        data, targets = batch[0].to(device), batch[1].to(device)
        z, log_j = cinn(data, targets)

        # do backprop
        nll = torch.mean(z**2) / 2 - torch.mean(log_j) / ndim_total
        nll.backward()
        torch.nn.utils.clip_grad_norm_(cinn.trainable_parameters, 10.)
        nll_mean.append(nll.item())
        optimizer.step()
        optimizer.zero_grad()

        # print training stats for every 50th batch 
        if not i % 50:
            with torch.no_grad():
                z, log_j = cinn(train_set.val.data, train_set.val.targets)
                nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / ndim_total

            print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.2e' % (
                    epoch, i, len(train_loader), 
                    (time() - t_start)/60.,
                    np.mean(nll_mean),
                    nll_val.item(),
                    optimizer.param_groups[0]['lr'],
                ), 
                flush=True
            )
            nll_mean = []   # reset list
    
    scheduler.step()   # update learning rate
print("")
print("Training complete!")
print("")

## save trained model
torch.save(cinn.state_dict(), save_path)
print("Saved model under")
print(save_path)
