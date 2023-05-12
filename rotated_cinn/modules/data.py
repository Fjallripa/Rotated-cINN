# Dataset preparation
# ===================
# defines the MnistRotated dataset class
#
# code inspired by https://github.com/AMLab-Amsterdam/DIVA



# Imports
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import path   # adds repo to PATH


# Parameters
## Mean and std of MNIST pixel values for normalization
norm_mean = 0.1307   # Values taken from an unnormalized training set.
norm_std  = 0.3082

# Amplitude for the noise augmentation
noise_std = 0.08



# Main classes
class DomainMNIST(Dataset):
    '''

    '''

    def __init__(self, domains: list[int], normalize: bool=False, add_noise: bool=False, transform=None) -> None:
        super().__init__()

        # Set attributes
        self.domains = domains
        self.classes = list(range(10))
        self.transform = transform

        ## Normalization
            # If normalize=False (default), then normalize() and unnormalize() won't have any effect on the data.
        self.normalized = normalize
        self.norm_mean = norm_mean  if self.normalized else  0.0
        self.norm_std  = norm_std   if self.normalized else  1.0

        ## Data augmentation
        self.noise_added = add_noise
        self.noise_std = noise_std  if self.noise_added else  0.0

        ## Predefine attributes for subclasses
        self.data = None
        self.targets = None
        self.domain_labels = None
        self.class_labels = None


    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.norm_mean) / self.norm_std

    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.norm_std + self.norm_mean
    
    def add_noise(self, data: torch.Tensor) -> torch.Tensor:
        return data + self.noise_std * torch.randn_like(data)


    def __len__(self):
        return len(self.targets)


    def __getitem__(self, index):
        image = self.data[index]
        label = self.targets[index]
        angle = self.domain_labels[index]
        digit = self.class_labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label, angle, digit



class RotatedMNIST(DomainMNIST):
    '''

    '''

    def __init__(self, domains: list[int], train: bool, val_set_size: int=0, seed: int=None, normalize: bool=False, add_noise: bool=False, transform=None) -> None:
        super().__init__(domains, normalize=normalize, add_noise=add_noise, transform=transform)

        # Set attributes
        self.train = train
        self.val_set_size = val_set_size
        self.val = None
        self.root = path.package_directory + "/data"
        if seed != None:
            torch.manual_seed(seed)

        # Create dataset
        mnist = datasets.MNIST(root=self.root, train=self.train, download=True, transform=transforms.ToTensor())
        self._process_data(mnist)


    def _process_data(self, dataset:Dataset) -> None:
        # Shuffle and normalize the images. normalize: uint8 (0..255) -> float32 (0..1)
        loader = DataLoader(dataset, batch_size=len(dataset.targets), shuffle=True)
        images, class_labels = next(iter(loader))
        images = images.squeeze(1)   # remove the batch dimension
        

        # Process the labels
        ## Create domain indices
        domain_count = len(self.domains)
        domain_indices = torch.randint_like(class_labels, domain_count)

        ## Create the new domain & class label for the cINN
        domains_sincos = self._deg2sincos(self.domains)
        classes_onehot = torch.eye(10)
        sincos_labels = domains_sincos[domain_indices]
        onehot_labels = classes_onehot[class_labels]
        cinn_labels = torch.cat((sincos_labels, onehot_labels), 1)


        # Process the images
        ## Rotate the images according to their domain
        images_rotated = torch.zeros_like(images)
        rotations = torch.tensor(self.domains)[domain_indices]
        for i in range(len(images_rotated)):
            images_rotated[i] = self._rotate(images[i], int(rotations[i]))

        ## Normalize the images
            # If self.normalized=False, this won't do anything.
        images_normalized = self.normalize(images_rotated)
        
        ## Add noise to the images
            # If self.noise_added=False, this won't do anything.
        images_augmented = self.add_noise(images_normalized)

        cinn_images = images_augmented


        # Return results
        if self.train and self.val_set_size > 0:
            cutoff = len(cinn_images) - self.val_set_size   # size of remaining training set
            cut = lambda tensor: self._split(tensor, at=cutoff)

            # Create train and validation set
            self.val = DomainMNIST(self.domains, normalize=self.normalized, add_noise=self.noise_added)   # create instance for validation set
            self.data,          self.val.data          = cut(cinn_images)
            self.targets,       self.val.targets       = cut(cinn_labels)
            self.domain_labels, self.val.domain_labels = cut(rotations)
            self.class_labels,  self.val.class_labels  = cut(class_labels)
        else:
            # Save train/test set
            self.data          = cinn_images
            self.targets       = cinn_labels
            self.domain_labels = rotations
            self.class_labels  = class_labels


    @staticmethod
    def _deg2sincos(degrees:torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [[np.cos(angle), np.sin(angle)]  for angle in np.deg2rad(degrees)],
            dtype = torch.float32
        )


    @staticmethod
    def _rotate(image:torch.Tensor, degrees:int) -> torch.Tensor:
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        return to_tensor(transforms.functional.rotate(to_pil(image), degrees))


    @staticmethod
    def _split(tensor:torch.Tensor, at:int) -> tuple[torch.Tensor]:
        len_a = at
        len_b = len(tensor) - len_a
        tensor_a, tensor_b = tensor.split([len_a, len_b])
        tensor_a = tensor_a.detach().clone()
        tensor_b = tensor_b.detach().clone()

        return tensor_a, tensor_b
