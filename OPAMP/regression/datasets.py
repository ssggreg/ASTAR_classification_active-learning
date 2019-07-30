import numpy as np
import torch
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset
DEFAULT_DATA_FOLDER = './data'

class Dataset():
    def __init__(self, data_set, data_folder = DEFAULT_DATA_FOLDER):
        super(type(self), self).__init__()

        if data_set == 'binary':
            X1 = torch.randn(1000, 50)
            X2 = torch.randn(1000, 50) + 1.5
            X = torch.cat([X1, X2], dim=0)
            Y1 = torch.zeros(1000, 1)
            Y2 = torch.ones(1000, 1)
            Y = torch.cat([Y1, Y2], dim=0)
            self.train_set = TensorDataset(X, Y)
            self.test_set = TensorDataset(X, Y)

        if data_set == 'mnist':
            #self.composed_transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
            self.train_set = dset.MNIST(root=data_folder,
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)

            self.test_set = dset.MNIST(root=data_folder,
                                       train=False,
                                       transform=transforms.ToTensor())

        if data_set == 'cifar10':

            #print(type(self.composed_transforms))
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            self.train_set = dset.CIFAR10(root=data_folder,
                                          train=True,
                                          transform= train_transform,
                                          download=True)

            self.test_set = dset.CIFAR10(root=data_folder,
                                         train=False,
                                         transform= test_transform)


    def get_train_size(self):
        return len(self.train_set)

    def get_test_size(self):
        return len(self.test_set)

    def get_train_loader(self, batch_size, shuffle=True):
        train_loader = DataLoader(dataset= self.train_set, batch_size=batch_size, shuffle=shuffle, num_workers=8)
        return train_loader

    def get_test_loader(self, batch_size, shuffle=False):
        test_loader = DataLoader(dataset= self.test_set, batch_size=batch_size, shuffle=shuffle, num_workers=8)
        return test_loader

    def load_full_train_set(self, use_cuda=torch.cuda.is_available()):

        full_train_loader = DataLoader(dataset = self.train_set,
                                       batch_size = len(self.train_set),
                                       shuffle = False)

        x_train, y_train = next(iter(full_train_loader))

        if use_cuda:
            x_train, y_train = x_train.cuda(), y_train.cuda()

        return x_train, y_train

    def load_full_test_set(self, use_cuda=torch.cuda.is_available()):

        full_test_loader = DataLoader(dataset = self.test_set,
                                      batch_size = len(self.test_set),
                                      shuffle = False)

        x_test, y_test = next(iter(full_test_loader))

        if use_cuda:
            x_test, y_test = x_test.cuda(), y_test.cuda()

        return x_test, y_test