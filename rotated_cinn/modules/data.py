# Dataset preparation
# ================
# defines the MnistRotated dataset class
#
# code inspired by https://github.com/AMLab-Amsterdam/DIVA



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


    def _process_data(self, dataset:Dataset) -> tuple[torch.tensor]:
        # Shuffle and normalize the images. normalize: uint8 (0..255) -> float32 (0..1)
        loader = DataLoader(dataset, batch_size=len(dataset.targets), shuffle=True)
        images, class_labels = next(iter(loader))
        images = images.squeeze(1)   # remove the batch dimension
        
        #ToDo: Separate validation set from training data

        # Create domain indices
        domain_count = len(self.domains)
        domain_indices = torch.randint_like(class_labels, domain_count)

        # Create the new domain & class label for the cINN
        domains_sincos = self._deg2sincos(self.domains)
        classes_onehot = torch.eye(10)
        sincos_labels = domains_sincos[domain_indices]
        onehot_labels = classes_onehot[class_labels]
        cinn_labels = torch.cat((sincos_labels, onehot_labels), 1)

        # Rotate the images according to their domain
        images_rotated = torch.zeros_like(images)
        rotations = self.domains[domain_indices]
        for i in range(len(images_rotated)):
            images_rotated[i] = self._rotate(images[i], int(rotations[i]))
        
        # Return results
        self.domain_labels = rotations
        self.class_labels = class_labels

        return images_rotated, cinn_labels
        

    @staticmethod
    def _deg2sincos(degrees:torch.tensor) -> torch.tensor:
        return torch.tensor(
            [[np.cos(angle), np.sin(angle)]  for angle in np.deg2rad(degrees)], 
            dtype = torch.float32
        )
    

    @staticmethod
    def _rotate(image:torch.tensor, degrees:int) -> torch.tensor:
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        
        return to_tensor(transforms.functional.rotate(to_pil(image), degrees))
        

    def __len__(self):
        return len(self.targets)


    def __getitem__(self, index):
        image = self.data[index]
        label = self.targets[index]
        angle = self.domain_labels[index]
        digit = self.class_labels[index]
        
        return image, label, angle, digit
