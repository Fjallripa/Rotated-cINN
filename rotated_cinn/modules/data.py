# Dataset preparation
# ================
# defines the MnistRotated dataset class



# Imports
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms



# Main class
class RotatedMNIST(Dataset):
    '''
    Creates a custom RotatedMNIST dataset.
    '''

    def __init__(self, domains: list[int], train: bool, val_set_size: int=0, seed: int=None) -> None:
        super().__init__()

        # Set attributes
        self.domains = torch.tensor(domains)
        self.train = train
        self.val_set_size = val_set_size

        self.root = os.path.dirname(os.path.realpath(__file__))
        if seed != None:
            torch.manual_seed(seed)
        self.classes = torch.tensor(range(10))
    

        # Create dataset
        mnist = datasets.MNIST(root=self.root, train=self.train, download=True, transform=transforms.ToTensor())
        
        self.data, self.targets = self._process_data(mnist)


    def _process_data(self, dataset):
        # Shuffle and normalize the images. normalize: uint8 (0..255) -> float32 (0..1)
        loader = DataLoader(dataset, batch_size=len(dataset.targets), shuffle=True)
        images, class_labels = next(iter(loader))
        images = images.squeeze(1)   # remove the batch dimension
        
        #ToDo: Separate validation set from training data

        # Create domain indices
        domain_count = len(self.domains)
        domain_indices = torch.randint_like(class_labels, domain_count)

        # Create the new domain & class label for the cINN
        domains_sincos = torch.tensor([[np.cos(angle), np.sin(angle)]  for angle in np.deg2rad(self.domains)])
        classes_onehot = torch.eye(10)
        sincos_labels = domains_sincos[domain_indices]
        onehot_labels = classes_onehot[class_labels]
        cinn_labels = torch.cat((sincos_labels, onehot_labels), 1)

        # Rotate the images according to their domain
        images_rotated = torch.zeros_like(images)
        rotations = self.domains[domain_indices]
        for i in range(len(images_rotated)):
            images_rotated[i] = transforms.functional.rotate(images[i], rotations[i])
        
        # Return results
        self.domain_labels = rotations
        self.class_labels = class_labels

        return images_rotated, cinn_labels
        

    def __len__(self):
        return len(self.targets)


    def __getitem__(self, index):
        image = self.data[index]
        label = self.targets[index]
        angle = self.domain_labels[index]
        digit = self.class_labels[index]
        
        return image, label, angle, digit


# Old versions of _process_data(). Temporary
'''
        # Create dataset
        mnist = datasets.MNIST(root=self.root, train=self.train, download=True, transform=transforms.ToTensor())
        
        self.data, self.targets = self._process_data(mnist)


    def _process_data(self, dataset):
        # Shuffle and normalize the images. normalize: uint8 (0..255) -> float32 (0..1)
        loader = DataLoader(dataset, batch_size=len(dataset.targets), shuffle=True)
        images, labels = next(iter(loader))

        #ToDo: Separate validation set from training data

        # Create domain indices
        set_size_raw = len(labels)
        num_domains = len(self.domains)
        domain_size = set_size_raw // num_domains
        domain_indices = torch.tensor([i // domain_size for i in range(set_size_raw)])
        domain_mask = domain_indices < num_domains
        
        # Cut off remainder of dataset from division into domains
        domains_cutoff = self.domains[domain_indices][domain_mask]
        labels_cutoff = labels[domain_mask]
        images_cutoff = images[domain_mask]

        # Create the new domain & class label for the cINN
        d = torch.tensor([(np.cos(angle), np.sin(angle))  for angle in np.deg2rad(self.domains)])
        domains_trig = d[domains_cutoff]
        y = torch.eye(10)
        labels_onehot = y[labels_cutoff]

        labels_cinn = torch.cat((d, y), 1)

        # Rotate the images according to their domain
        images_rotated = torch.zeros_like(images_cutoff)
        for i in range(len(images_rotated)):
            images_rotated[i] = transforms.functional.rotate(images_cutoff[i], domains_cutoff[i])

        # Shuffle all once more
        shuffle = torch.randperm(len(labels_cinn))
        data = images_rotated[shuffle]
        targets = labels_cinn[shuffle]
        
        return data, targets
        

    def __len__(self):
        return len(self.targets)


    def __getitem__(self, index):
        image = self.data[index]
        label = self.targets[index]
        
        return image, label
'''

'''
# Create dataset
        if self.train:
            self.load_training()
        else:
            self.load_test()

        self.process_data()
        


    def load_training(self):
        mnist = datasets.MNIST(root=self.root, train=True, download=True, transform=transforms.ToTensor())
        mnist.data = mnist.data.to(float) / 255  
            # converts images from uint8 (0..255) to float32, normalizes them to 0..1
            # The dataloader does the same. This is just a precaution.
        shuffle = torch.randperm(mnist.data.size()[0])
        mnist.data = mnist.data[shuffle]
        mnist.targets = mnist.targets[shuffle]

        


    def load_test(self):
        pass


    def process_data(self):
        pass


    def __len__(self):
        pass


    def __getitem__(self, index):
        pass
'''
