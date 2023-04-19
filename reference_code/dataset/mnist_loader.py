# Data Preparation (from DIVA)
# ============================
# defines the MnistRotated class and provides a script for testing it out

import os
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


# Main class definition
class MnistRotated(data_utils.Dataset):
    """
    Pytorch Dataset object that loads MNIST. It returns x,y,s where s=0 when x,y is taken from MNIST.
    """

    # Create an instance 
    def __init__(self, list_train_domains, list_test_domain, num_supervised, mnist_subset, root, transform=None, train=True, download=True):

        self.list_train_domains = list_train_domains  # List of the training angles.
        self.list_test_domain = list_test_domain      # List of the (one) test angle.
        self.num_supervised = num_supervised   # number of labeled training images
        self.mnist_subset = mnist_subset   # the digit of one of the ten 'supervised_inds...' files
            #? Why are there 10 possible subsets (nothing to do with classes)? Why are they precalculated?
        self.root = os.path.expanduser(root)  # should make a full path out of a given relative one.
            # It's unnecessary: The way it's used, setting it to 
            # self.root = os.path.dirname(os.path.realpath(__file__))   # the directory of this file
            # without user input is safer. 
        self.transform = transform
        self.train = train   # if True, creates training dataset, else test dataset
        self.download = download

        # Create a training or a test dataset
        if self.train:
            self.train_data, self.train_labels, self.train_domain = self._get_data()
        else:
            self.test_data, self.test_labels, self.test_domain = self._get_data()


    # Loads the indices of the preselected mnist_subset
    def load_inds(self):
        return np.load(self.root + 'supervised_inds_' + str(self.mnist_subset) + '.npy')


    def _get_data(self):
        # Creating the training dataset
        if self.train:
            # Load the relevant images
            ## Download the whole MNIST training dataset into a data loader, unshuffled.
            train_loader = torch.utils.data.DataLoader(datasets.MNIST(self.root,
                                                                      train=True,
                                                                      download=self.download,
                                                                      transform=transforms.ToTensor()),
                                                       batch_size=60000,
                                                       shuffle=False)

            ## Separate images and labels
            for i, (x, y) in enumerate(train_loader):
                mnist_imgs = x
                mnist_labels = y

            ## Get num_supervised number of labeled examples
                # The training dataset consists of only a small subsection of the MNIST dataset.
                # There seems to be no script left that generated these indices ("mnist_subset").
                # My analysis indicates they were sampled uniformly from 0 to 59999.
                # With replecament inside a subset, without replacement between subsets.
            sup_inds = self.load_inds()
            mnist_labels = mnist_labels[sup_inds]
            mnist_imgs = mnist_imgs[sup_inds]

            
            # Run transforms
            ## Parameters
            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()

            mnist_0_img = torch.zeros((self.num_supervised, 28, 28))
            mnist_15_img = torch.zeros((self.num_supervised, 28, 28))
            mnist_30_img = torch.zeros((self.num_supervised, 28, 28))
            mnist_45_img = torch.zeros((self.num_supervised, 28, 28))
            mnist_60_img = torch.zeros((self.num_supervised, 28, 28))
            mnist_75_img = torch.zeros((self.num_supervised, 28, 28))

            ## Rotate the selected images
                # All domains have the same num_supervised images, only rotated.
            for i in range(len(mnist_imgs)):
                mnist_0_img[i] = to_tensor(to_pil(mnist_imgs[i]))

            for i in range(len(mnist_imgs)):
                mnist_15_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 15))

            for i in range(len(mnist_imgs)):
                mnist_30_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 30))

            for i in range(len(mnist_imgs)):
                mnist_45_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 45))

            for i in range(len(mnist_imgs)):
                mnist_60_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 60))

            for i in range(len(mnist_imgs)):
                mnist_75_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 75))


            # Build dataset out of individual domains
            ## Choose the domains (angles) that should be included into the training
            training_list_img = []
            training_list_labels = []

            for domain in self.list_train_domains:
                if domain == '0':
                    training_list_img.append(mnist_0_img)
                    training_list_labels.append(mnist_labels)
                if domain == '15':
                    training_list_img.append(mnist_15_img)
                    training_list_labels.append(mnist_labels)
                if domain == '30':
                    training_list_img.append(mnist_30_img)
                    training_list_labels.append(mnist_labels)
                if domain == '45':
                    training_list_img.append(mnist_45_img)
                    training_list_labels.append(mnist_labels)
                if domain == '60':
                    training_list_img.append(mnist_60_img)
                    training_list_labels.append(mnist_labels)
                if domain == '75':
                    training_list_img.append(mnist_75_img)
                    training_list_labels.append(mnist_labels)

            ## Stack all the domains into one tensor
            train_imgs = torch.cat(training_list_img)
            train_labels = torch.cat(training_list_labels)

            ## Create domain labels
                # The five training domains are just labelled 0 to 4
            train_domains = torch.zeros(train_labels.size())
            train_domains[0: self.num_supervised] += 0
            train_domains[self.num_supervised: 2 * self.num_supervised] += 1
            train_domains[2 * self.num_supervised: 3 * self.num_supervised] += 2
            train_domains[3 * self.num_supervised: 4 * self.num_supervised] += 3
            train_domains[4 * self.num_supervised: 5 * self.num_supervised] += 4

            ## Shuffle the dataset so that examples from different domains appear randomly.
            inds = np.arange(train_labels.size()[0])
            np.random.shuffle(inds)
            train_imgs = train_imgs[inds]
            train_labels = train_labels[inds]
            train_domains = train_domains[inds].long()  # also make domain lables to long ints

            ## Convert class and domain labels to onehot format
            y = torch.eye(10)
            train_labels = y[train_labels]
            d = torch.eye(5)
            train_domains = d[train_domains]


            # Return data, class & domain labels
            return train_imgs.unsqueeze(1), train_labels, train_domains


        # Creating the test dataset
        else:
            # Load the relevant images
            ## Download the whole MNIST test dataset into a data loader, unshuffled.
                #Why is train=True? -> because the test here is performance on another domain, not on other data.  
            train_loader = torch.utils.data.DataLoader(datasets.MNIST(self.root,
                                                                      train=True,
                                                                      download=self.download,
                                                                      transform=transforms.ToTensor()),
                                                       batch_size=60000,
                                                       shuffle=False)

            ## Separate images and labels
            for i, (x, y) in enumerate(train_loader):
                mnist_imgs = x
                mnist_labels = y

            ## Get num_supervised number of labeled examples
            sup_inds = self.load_inds()
            mnist_labels = mnist_labels[sup_inds]
            mnist_imgs = mnist_imgs[sup_inds]

            
            # Run transforms
            ## Parameters
            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()

            rot_angle = int(self.list_test_domain[0])   # angle of test domain
            mnist_imgs_rot = torch.zeros((self.num_supervised, 28, 28))

            ## Rotate the selected images
            for i in range(len(mnist_imgs)):
                mnist_imgs_rot[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), rot_angle))

            # Build dataset
            ## Create domain labels
            test_domain = torch.zeros(mnist_labels.size()).long()

            ## Convert class and domain labels to onehot format
            y = torch.eye(10)
            mnist_labels = y[mnist_labels]
            d = torch.eye(5)
            test_domain = d[test_domain]


            # Return data, class & domain labels
            return mnist_imgs_rot.unsqueeze(1), mnist_labels, test_domain


    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)


    def __getitem__(self, index):
        if self.train:
            x = self.train_data[index]
            y = self.train_labels[index]
            d = self.train_domain[index]
        else:
            x = self.test_data[index]
            y = self.test_labels[index]
            d = self.test_domain[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y, d



# Main code
if __name__ == "__main__":
    '''
    This script tries out the MnistRotated class and creates a test and train data loader.
    '''
    
    from torchvision.utils import save_image

    # Parameters
    seed = 1
    root = os.path.dirname(os.path.realpath(__file__)) + '/'   # directory of this script

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    list_train_domains = ['0', '15', '30', '45', '60']
    list_test_domains = ['75']
    num_supervised = 1000


    # Try out training dataset
    train_loader = data_utils.DataLoader(
        MnistRotated(list_train_domains, list_test_domains, num_supervised, seed, root, train=True),
        batch_size=100,
        shuffle=False)

    y_array = np.zeros(10)
    d_array = np.zeros(5)

    for i, (x, y, d) in enumerate(train_loader):
        y_array += y.sum(dim=0).cpu().numpy()
        d_array += d.sum(dim=0).cpu().numpy()

        if i == 0:
            print(y)
            print(d)
            n = min(x.size(0), 8)
            comparison = x[:n].view(-1, 1, 16, 16)
            save_image(comparison.cpu(),
                       'reconstruction_rotation_train.png', nrow=n)

    print(y_array, d_array)


    # Try out test dataset
    test_loader = data_utils.DataLoader(
        MnistRotated(list_train_domains, list_test_domains, num_supervised, seed, root, train=False),
        batch_size=100,
        shuffle=False)

    y_array = np.zeros(10)
    d_array = np.zeros(5)

    for i, (x, y, d) in enumerate(test_loader):
        y_array += y.sum(dim=0).cpu().numpy()
        d_array += d.sum(dim=0).cpu().numpy()

        if i == 0:
            print(y)
            print(d)
            n = min(x.size(0), 8)
            comparison = x[:n].view(-1, 1, 16, 16)
            save_image(comparison.cpu(),
                       'reconstruction_rotation_test.png', nrow=n)

    print(y_array, d_array)