# Data Preparation
# ================
# Downloads MNIST and creates a data loader.

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T
import torchvision.datasets


# Parameters
device = 'cuda'  if torch.cuda.is_available() else  'cpu'

batch_size = 256
data_mean = 0.128
data_std = 0.305

## Amplitude for the noise augmentation
augm_sigma = 0.08
data_dir = 'mnist_data'


# Support functions
def unnormalize(x):
    '''go from normalized data x back to the original range'''
    return x * data_std + data_mean



# Main code
## Download MNIST dataset
train_data = torchvision.datasets.MNIST(data_dir, train=True, download=True,
                                        transform=T.Compose([T.ToTensor(), lambda x: (x - data_mean) / data_std]))# creates an instance of MNIST class
test_data = torchvision.datasets.MNIST(data_dir, train=False, download=True,
                                        transform=T.Compose([T.ToTensor(), lambda x: (x - data_mean) / data_std]))


## Create a validation set
### Sample a fixed batch of 1024 validation examples
val_x, val_l = zip(*list(train_data[i] for i in range(1024)))
    # train_data[i] is a (image, label) tuple: (<torch.tensor>, <int>)
    # val_x: images. a tuple of 1024 images, each image is a (1, 28, 28) tensor of floats (a grayscale MNIST image).
    # val_l> labels. the tuple of the corresponding labels
val_x = torch.stack(val_x, 0).to(device)
    # torch.tensor, shape: (1024, 1, 28, 28)
val_l = torch.LongTensor(val_l).to(device)
    # torch.tensor, shape: (1024)
    # LongTensor means 64bit integers as dtype. Unnecessary here as dataset is small.

### Exclude the validation batch from the training data
train_data.data = train_data.data[1024:]
train_data.targets = train_data.targets[1024:]


## Add the noise-augmentation to the (non-validation) training data:
train_data.transform = T.Compose([train_data.transform, lambda x: x + augm_sigma * torch.randn_like(x)])


## Create the data loaders for train and test data
train_loader  = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                           num_workers=4, pin_memory=True, drop_last=True)
test_loader   = DataLoader(test_data,  batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True, drop_last=True)
