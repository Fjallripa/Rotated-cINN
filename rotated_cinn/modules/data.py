# Dataset preparation
# ================
# defines the MnistRotated dataset class


# Imports
import torch.utils.data


# Main class
class RotatedMNIST(torch.utils.data.Dataset):
    '''
    Creates a custom RotatedMNIST dataset.
    '''

    def __init__(self, domains: list[int], train: bool, val_set_size=0) -> None:
        super().__init__()
    

    def load_training(self):
        pass


    def load_test(self):
        pass


    def process_data(self):
        pass


    def __len__(self):
        pass


    def __getitem__(self, index):
        pass
