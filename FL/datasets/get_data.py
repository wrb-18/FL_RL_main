# Interface between the dataset and client
# For artificially partitioned dataset, params include num_clients, dataset

# from FL.datasets.cifar_mnist import get_dataset, show_distribution
from FL.datasets.cifar_mnist import get_dataset
import torch
from torch.autograd import Variable
def get_dataloaders(args):
    """
    :param args:
    :return: A list of trainloaders, a list of testloaders, a concatenated trainloader and a concatenated testloader
    """
    if args.dataset in ['mnist', 'cifar10']:
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_dataset(dataset_root='data',
                                                                                       dataset=args.dataset,
                                                                                       args = args)        
        train_loaders_ = [[] for i in range(args.num_clients)] 
        test_loaders_ = [[] for i in range(args.num_clients)]  
        v_test_loader_ = []  
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        for i in range(args.num_clients):
            print("loading dataset for client", i)
            for data in train_loaders[i]:
                inputs, labels = data     
                inputs = inputs.to(device)
                labels = labels.to(device)
                data = inputs, labels
                train_loaders_[i].append(data)
        for data in v_test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            data = inputs, labels
            v_test_loader_.append(data)
            
    else:
        raise ValueError("This dataset is not implemented yet")
    return train_loaders_, test_loaders_, v_train_loader, v_test_loader_