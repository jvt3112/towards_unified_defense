"""
Code used from: https://github.com/shaoxiongji/federated-learning/blob/master/utils/sampling.py
- MNIST iid function
"""
import copy
import torch
import numpy as np
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.utils.data import random_split
from torch.utils.data import random_split, DataLoader, Dataset

def mnist_iid(dataset, num_clients):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_clients:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_clients) # dividing the dataset into num_clients parts
    dict_users, all_idxs = {}, [i for i in range(len(dataset))] # dict_users is a dictionary of client index and image index

    for i in range(num_clients): # for each client
        dict_users[i] = set(np.random.choice(all_idxs, num_items,replace=False)) # randomly choose num_items images from all_idxs
        all_idxs = list(set(all_idxs) - dict_users[i]) # remove the chosen images from all_idxs

    return dict_users # return a dictionary of client index and image index

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.idxs = [int(i) for i in range(len(self.dataset))]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]][0], self.dataset[self.idxs[item]][1][0]
        image = image.reshape(1, 28, 28)
        return torch.tensor(image), torch.tensor(label)
    
# get the dataset
def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    # initialize the train_dataset_watermark and user_groups_watermark 
    train_dataset_watermark = None 
    user_groups_watermark = None

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

    elif args.dataset == 'mnist' or 'fmnist':
        apply_transform = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.ToTensor()])
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        else:
            data_dir = '../data/fmnist/'
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                          transform=apply_transform)

            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)
        if args.dataset == 'mnist':
            if args.wm:
                load_watermark_dataset = torch.load('./centralised/watermarks/watermarks_samples_0.4_40.pt')
                train_dataset_watermark = DatasetSplit(load_watermark_dataset)
        if args.dataset == 'fmnist':
            if args.wm:
                load_watermark_dataset = torch.load('./centralised/watermarks/watermark_samples_fmnist_0.30_40_new.pt')
                train_dataset_watermark = DatasetSplit(load_watermark_dataset)
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_clients) 
    return train_dataset, test_dataset, train_dataset_watermark, user_groups, user_groups_watermark
