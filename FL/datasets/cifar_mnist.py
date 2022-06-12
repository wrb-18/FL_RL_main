"""
download the required dataset, split the data among the clients, and generate DataLoader for training
"""
import os
from tqdm import tqdm
from sklearn import metrics
import numpy as np

import torch
import torch.backends.cudnn as cudnn
cudnn.banchmark = True

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from options import args_parser

class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        super(DatasetSplit, self).__init__()
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, target = self.dataset[self.idxs[item]]
        return image, target

def split_data(dataset, args, kwargs, is_shuffle = True):
    data_loaders = [0] * args.num_clients
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(len(dataset))
    # is_shuffle is used to differentiate between train and test
    if is_shuffle:
        labels = dataset.train_labels
    else:
        labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    # sort the data according to their label
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)
    
    ##################################################hyx添加
    tmp =  [[512, 0, 0,  0, 0, 0, 0, 0, 0, 0],   #0
            [0, 512, 0,  0, 0, 0, 0, 0, 0, 0],   #1
            [0, 0, 64, 0, 0, 0, 64, 64, 0, 0],   #2
            [0, 64, 0, 0, 64, 0, 64, 64, 0, 0],   #3
            [0, 0, 0, 64, 0, 0, 0, 0, 64, 64],   #4
            [64, 0, 0, 0, 0, 64, 0, 0, 0, 64],   #5
            [16, 16, 16, 16, 16, 16, 16, 16, 16, 16],   #6
            [16, 16, 16, 16, 16, 16, 16, 16, 16, 16],   #7
            [16, 16, 16, 16, 16, 16, 16, 16, 16, 16],   #8
            [16, 16, 16, 16, 16, 16, 16, 16, 16, 16],   #9

           ]
    
    for i in range(2):
        alloc_list = tmp[i]
        for digit, num_of_digit in enumerate(alloc_list):
            tmp1 = np.argwhere(idxs_labels[1, :] == digit)
            tmp1 = tmp1.ravel()
            tmp2 = np.random.choice(idxs_labels[0, tmp1[0:1]], num_of_digit, replace = True)
            dict_users[i] = np.concatenate((dict_users[i], tmp2), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                    batch_size = args.batch_size,
                                    shuffle = is_shuffle, **kwargs)      
    for i in range(2, args.num_clients):
        alloc_list = tmp[i]
        for digit, num_of_digit in enumerate(alloc_list):
            tmp1 = np.argwhere(idxs_labels[1, :] == digit)
            tmp1 = tmp1.ravel()
            tmp2 = np.random.choice(idxs_labels[0, tmp1], num_of_digit, replace = True)
            dict_users[i] = np.concatenate((dict_users[i], tmp2), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                    batch_size = args.batch_size,
                                    shuffle = is_shuffle, **kwargs)  
    return data_loaders

def get_dataset(dataset_root, dataset, args):
    trains, train_loaders, tests, test_loaders = {}, {}, {}, {}
    if dataset == 'mnist':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_mnist(dataset_root, args)
    elif dataset == 'cifar10':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_cifar10(dataset_root, args)
    elif dataset == 'femnist':
        raise ValueError('CODING ERROR: FEMNIST dataset should not use this file')
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))
    return train_loaders, test_loaders, v_train_loader, v_test_loader

def get_mnist(dataset_root, args):
    #print(dataset_root)
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
    transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ])
    train = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train = True,
                            download = True, transform = transform)
    test =  datasets.MNIST(os.path.join(dataset_root, 'mnist'), train = False,
                            download = True, transform = transform)
    #note: is_shuffle here also is a flag for differentiating train and test
    train_loaders = split_data(train, args, kwargs, is_shuffle = True)
    test_loaders = split_data(test,  args, kwargs, is_shuffle = False)
    #the actual batch_size may need to change.... Depend on the actual gradient...
    #originally written to get the gradient of the whole dataset
    #but now it seems to be able to improve speed of getting accuracy of virtual sequence
    v_train_loader = DataLoader(train, batch_size = args.batch_size * args.num_clients,
                                shuffle = True, **kwargs)
    v_test_loader = DataLoader(test, batch_size = args.batch_size * args.num_clients,
                                shuffle = False, **kwargs)
    return  train_loaders, test_loaders, v_train_loader, v_test_loader


def get_cifar10(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory':True} if is_cuda else{}
    if args.model == 'cnn_complex':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.model == 'resnet18':
        transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding = 4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


    else:
        raise ValueError("this nn for cifar10 not implemented")
    train = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train = True,
                        download = True, transform = transform_train)
    test = datasets.CIFAR10(os.path.join(dataset_root,'cifar10'), train = False,
                        download = True, transform = transform_test)
    v_train_loader = DataLoader(train, batch_size = args.batch_size,
                                shuffle = True, **kwargs)
    v_test_loader = DataLoader(test, batch_size = args.batch_size,
                                shuffle = False, **kwargs)
    train_loaders = split_data(train, args, kwargs, is_shuffle = True)
    test_loaders = split_data(test,  args, kwargs, is_shuffle = False)
    return  train_loaders, test_loaders, v_train_loader, v_test_loader