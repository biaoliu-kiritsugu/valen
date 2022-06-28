import math
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models
import torch.utils.data as data
from sklearn.preprocessing import OneHotEncoder
import random
from copy import deepcopy
# from datasets.realworld.realworld import KFoldDataLoader, RealWorldData
from partial_models.resnet import resnet
from partial_models.resnext import resnext
from partial_models.linear_mlp_models import mlp_model
from utils.realword_dataset import RealwordDataLoader, My_Subset, RealWorldData


def extract_data(config, **args):
    if config.dt == "benchmark":
        if config.ds == "mnist":
            train_dataset = dsets.MNIST(root="data/" + config.dt, train=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]), download=True)
            test_dataset = dsets.MNIST(root="data/" + config.dt, train=False, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        if config.ds == "kmnist":
            train_dataset = dsets.KMNIST(root="data/" + config.dt, train=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]), download=True)
            test_dataset = dsets.KMNIST(root="data/" + config.dt, train=False, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
        if config.ds == "fmnist":
            train_dataset = dsets.FashionMNIST(root="data/" + config.dt, train=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]), download=True)
            test_dataset = dsets.FashionMNIST(root="data/" + config.dt, train=False, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        if config.ds == "cifar10":
            train_dataset = dsets.CIFAR10(root="data/" + config.dt + '/CIFAR10/', train=True,
                                          transform=transforms.Compose([transforms.ToTensor(),
                                                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (
                                                                        0.2023, 0.1994, 0.2010)), ]), download=True)
            test_dataset = dsets.CIFAR10(root="data/" + config.dt + '/CIFAR10/', train=False,
                                         transform=transforms.Compose([transforms.ToTensor(),
                                                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (
                                                                       0.2023, 0.1994, 0.2010)), ]))
        if config.ds == 'cifar100':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            train_dataset = dsets.CIFAR100(root="data/" + config.dt + '/CIFAR100/',
                                           train=True, transform=transform, download=True)
            test_dataset = dsets.CIFAR100(root="data/" + config.dt + '/CIFAR100/', train=False, transform=transform)

        data_size = len(train_dataset)
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset,
                                                                     [int(data_size * 0.9), data_size - int(data_size * 0.9)])
        train_full_loader = data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True,
                                            num_workers=8)
        valid_full_loader = data.DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset), shuffle=True,
                                           num_workers=8)
        test_full_loader = data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=True,
                                           num_workers=8)
        train_X, train_Y = next(iter(train_full_loader))
        train_Y = binarize_class(train_Y)
        valid_X, valid_Y = next(iter(valid_full_loader))
        valid_Y = binarize_class(valid_Y)
        test_X, test_Y = next(iter(test_full_loader))
        test_Y = binarize_class(test_Y)
        if config.ds not in  ["cifar10", "cifar100"]:
            train_X = train_X.view(train_X.shape[0], -1)
            valid_X = valid_X.view(valid_X.shape[0], -1)
            test_X = test_X.view(test_X.shape[0], -1)
        yield train_X, train_Y, test_X, test_Y, valid_X, valid_Y
    if config.dt == "realworld":
        mat_path = "/home/liubiao/datasets/realworld/" + config.ds + '.mat'
        data_reader = RealwordDataLoader(mat_path)
        full_dataset = RealWorldData(data_reader)
        full_data_size = len(full_dataset)
        test_size, valid_size = int(full_data_size * 0.2), int(full_data_size * 0.2)
        train_size = full_data_size - test_size - valid_size
        train_dataset, valid_dataset, test_dataset = \
            torch.utils.data.random_split(full_dataset, [train_size, valid_size, test_size],
                                          torch.Generator().manual_seed(42))
        train_idx, valid_idx, test_idx = train_dataset.indices, valid_dataset.indices, test_dataset.indices
        train_dataset, valid_dataset, test_dataset = \
            My_Subset(full_dataset, train_idx), My_Subset(full_dataset, valid_idx), My_Subset(full_dataset, test_idx)
        train_full_loader = data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True,
                                            num_workers=8)
        valid_full_loader = data.DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset), shuffle=True,
                                            num_workers=8)
        test_full_loader = data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=True,
                                           num_workers=8)
        train_X, train_Y, _, _, _ = next(iter(train_full_loader))
        valid_X, _, _, valid_Y, _ = next(iter(valid_full_loader))
        test_X, _, _, test_Y, _ = next(iter(test_full_loader))
        yield train_X, train_Y, test_X, test_Y, valid_X, valid_Y


def partialize(config, **args):
    if config.partial_type == 'random':
        if config.ds != 'cifar100':
            train_Y = args['train_Y']
            train_p_Y, avgC = random_partialize(train_Y)
        else:
            train_Y = args['train_Y']
            train_p_Y, avgC = random_partialize(train_Y, p=0.05)
    if config.partial_type == 'feature':
        train_Y = args['train_Y']
        train_X = args['train_X']
        device = args['device']
        # model = deepcopy(args['model'])
        # weight_path = args['weight_path']
        partial_batch_size = 100
        if config.ds == 'cifar10':
            weight_path = 'partial_weights/checkpoint_c10_resnet.pt'
            model = resnet(depth=32, num_classes=10)
            rate = 0.4
        elif config.ds == 'mnist':
            weight_path = 'partial_weights/checkpoint_mnist_mlp.pt'
            model = mlp_model(input_dim=args['dim'], output_dim=10)
            rate = 0.4
        elif config.ds == 'kmnist':
            weight_path = 'partial_weights/checkpoint_kmnist_mlp.pt'
            model = mlp_model(input_dim=args['dim'], output_dim=10)
            rate = 0.4
        elif config.ds == 'fmnist':
            weight_path = 'partial_weights/checkpoint_fashion_mlp.pt'
            model = mlp_model(input_dim=args['dim'], output_dim=10)
            rate = 0.4
        elif config.ds == 'cifar100':
            weight_path = 'partial_weights/c100_resnext.pt'
            model = resnext(cardinality=16, depth=29, num_classes=100)
            rate = 0.04

        train_p_Y, avgC = feature_partialize(train_X, train_Y, model, weight_path, device,
                                             rate=rate, batch_size=partial_batch_size)
    return train_p_Y, avgC


def create_realword_data(train_dataset, test_dataset):
    train_X, train_p_Y, train_Y = train_dataset.train_features, train_dataset.train_targets, train_dataset.train_labels
    test_X, test_Y = test_dataset.test_features, test_dataset.test_labels
    indexes = [i for i in range(0, len(train_X))]
    # random.shuffle(indexes)
    # sampled_indexes = indexes[:min(10000, len(train_X))]
    sampled_indexes = indexes
    train_gcn_X, train_gcn_p_Y, train_gcn_Y = train_X[sampled_indexes], train_p_Y[sampled_indexes], train_Y[
        sampled_indexes]
    return (train_X, train_p_Y, train_Y), (test_X, test_Y), (train_gcn_X, train_gcn_p_Y, train_gcn_Y)


def create_uci_data(train_dataset, test_dataset):
    train_X, train_p_Y, train_Y = train_dataset.train_features, train_dataset.train_p_labels, train_dataset.train_logitlabels
    test_X, test_Y = test_dataset.test_features, test_dataset.test_logitlabels
    indexes = [i for i in range(0, len(train_X))]
    # random.shuffle(indexes)
    # sampled_indexes = indexes[:min(10000, len(train_X))]
    sampled_indexes = indexes
    train_gcn_X, train_gcn_p_Y, train_gcn_Y = train_X[sampled_indexes], train_p_Y[sampled_indexes], train_Y[
        sampled_indexes]
    return (train_X, train_p_Y, train_Y), (test_X, test_Y), (train_gcn_X, train_gcn_p_Y, train_gcn_Y)


def create_train_loader(train_X, train_Y, train_p_Y, batch_size=256):
    class dataset(data.Dataset):
        def __init__(self, train_X, train_Y, train_p_Y):
            self.train_X = train_X
            self.train_p_Y = train_p_Y
            self.train_Y = train_Y

        def __len__(self):
            return len(self.train_X)

        def __getitem__(self, idx):
            return self.train_X[idx], self.train_p_Y[idx], self.train_Y[idx], idx

    ds = dataset(train_X, train_Y, train_p_Y)
    dl = data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    return dl


def create_full_dataloader(dataset, shuffle=True, num_workers=8):
    full_loader = data.DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=shuffle, num_workers=num_workers)
    return full_loader


def binarize_class(y):
    label = y.reshape(len(y), -1)
    enc = OneHotEncoder(categories='auto')
    enc.fit(label)
    label = enc.transform(label).toarray().astype(np.float32)
    label = torch.from_numpy(label)
    return label


def random_partialize(y, p=0.5):
    new_y = y.clone()
    n, c = y.shape[0], y.shape[1]
    avgC = 0

    for i in range(n):
        row = new_y[i, :]
        row[np.where(np.random.binomial(1, p, c) == 1)] = 1
        while torch.sum(row) == 1:
            row[np.random.randint(0, c)] = 1
        avgC += torch.sum(row)

    avgC = avgC / n
    return new_y, avgC


def _feature_partialize(train_X, train_Y, model, weight_path, device, rate=0.4, batch_size=2000):
    with torch.no_grad():
        model = model.to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        avg_C = 0
        train_X, train_Y = train_X.to(device), train_Y.to(device)
        train_p_Y_list = []
        step = train_X.size(0) // batch_size
        for i in range(0, step):
            _, outputs = model(train_X[i * batch_size:(i + 1) * batch_size])
            train_p_Y = train_Y[i * batch_size:(i + 1) * batch_size].clone().detach()
            partial_rate_array = F.softmax(outputs, dim=1).clone().detach()
            partial_rate_array[torch.where(train_Y[i * batch_size:(i + 1) * batch_size] == 1)] = 0
            partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
            partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
            partial_rate_array[partial_rate_array > 1.0] = 1.0
            m = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array)
            z = m.sample()
            train_p_Y[torch.where(z == 1)] = 1.0
            train_p_Y_list.append(train_p_Y)
        train_p_Y = torch.cat(train_p_Y_list, dim=0)
        assert train_p_Y.shape[0] == train_X.shape[0]
    avg_C = torch.sum(train_p_Y) / train_p_Y.size(0)
    return train_p_Y.cpu(), avg_C.item()


def feature_partialize(train_X, train_Y, model, weight_path, device, rate=0.4, batch_size=1000):
    with torch.no_grad():
        model = model.to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        # model.eval()
        avg_C = 0
        train_X, train_Y = train_X.to(device), train_Y.to(device)
        train_p_Y_list = []
        # step = train_X.size(0) // batch_size
        step = math.ceil(train_X.size(0) / batch_size)
        for i in range(step):
            low = i * batch_size
            high = min((i + 1) * batch_size, train_X.size(0))
            # outputs = model(train_X[i * batch_size:(i + 1) * batch_size])
            outputs = model(train_X[low: high])
            # train_p_Y = train_Y[i * batch_size:(i + 1) * batch_size].clone().detach()
            train_p_Y = train_Y[low: high].clone().detach()
            partial_rate_array = F.softmax(outputs, dim=1).clone().detach()
            # partial_rate_array[torch.where(train_Y[i * batch_size:(i + 1) * batch_size] == 1)] = 0
            partial_rate_array[torch.where(train_Y[low: high] == 1)] = 0
            partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
            # partial_rate_array = partial_rate_array / torch.sum(partial_rate_array, dim=1, keepdim=True)
            partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
            partial_rate_array[partial_rate_array > 1.0] = 1.0
            partial_rate_array = torch.nan_to_num(partial_rate_array, nan=0)
            # debug_value = partial_rate_array.cpu().numpy()
            # partial_rate_array[partial_rate_array < 0.0] = 0.0
            m = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array)
            z = m.sample()
            train_p_Y[torch.where(z == 1)] = 1.0
            train_p_Y_list.append(train_p_Y)
        train_p_Y = torch.cat(train_p_Y_list, dim=0)
        assert train_p_Y.shape[0] == train_X.shape[0]
    avg_C = torch.sum(train_p_Y) / train_p_Y.size(0)
    return train_p_Y.cpu(), avg_C.item()