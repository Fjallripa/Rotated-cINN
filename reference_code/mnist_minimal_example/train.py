# Model Training
# ==============
# Run this script to train the model.

from time import time

from tqdm import tqdm
import torch
import torch.nn
import torch.optim
import numpy as np

import path
import model
import data


# Parameters
save_path = path.file_directory + '/output/mnist_cinn.pt'
device = 'cuda'  if torch.cuda.is_available() else  'cpu'

cinn = model.Rotated_cINN(lr=5e-4)
cinn.to(device)
scheduler = torch.optim.lr_scheduler.MultiStepLR(cinn.optimizer, milestones=[20, 40], gamma=0.1)

N_epochs = 60
t_start = time()
nll_mean = []


# Main code
print('Epoch\tBatch/Total \tTime \tNLL train\tNLL val\tLR')
    # Epoch:    number of walks through the whole training set
    # Total:    number of batches in training set
    # Time:     clock time in minutes since start of training
    # NLL train:    mean loss over last 50 batches
    # NLL val:      loss over validation set
    # LR:       current learning rate

## run training loop
for epoch in range(N_epochs):
    for i, (x, l) in enumerate(data.train_loader):   # for batches in training set
        # run model forward
        x, l = x.to(device), l.to(device)
        z, log_j = cinn(x, l)

        # do backprop
        nll = torch.mean(z**2) / 2 - torch.mean(log_j) / model.ndim_total
        nll.backward()
        torch.nn.utils.clip_grad_norm_(cinn.trainable_parameters, 10.)
        nll_mean.append(nll.item())
        cinn.optimizer.step()
        cinn.optimizer.zero_grad()

        # print training stats for every 50th batch 
        if not i % 50:
            with torch.no_grad():
                z, log_j = cinn(data.val_x, data.val_l)
                nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / model.ndim_total

            print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.2e' % (
                    epoch, i, len(data.train_loader), 
                    (time() - t_start)/60.,
                    np.mean(nll_mean),
                    nll_val.item(),
                    cinn.optimizer.param_groups[0]['lr'],
                ), 
                flush=True
            )
            nll_mean = []   # reset list
    
    scheduler.step()   # update learning rate

## save trained model
torch.save(cinn.state_dict(), save_path)
