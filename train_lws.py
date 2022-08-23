# import
import argparse
import gc
import traceback
from copy import deepcopy
from operator import index
import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
import os
import pickle
import time

from models.resnet import resnet
from utils.args import extract_args
from utils.data_factory import extract_data, partialize, create_train_loader, create_test_loader
from utils.model_factory import create_model, create_model_for_baseline
from utils.utils_graph import gen_adj_matrix2
from utils.utils_loss import partial_loss, alpha_loss, kl_loss, revised_target, out_d_loss, proden_loss, rc_loss, \
    cc_loss, lws_loss
from utils.utils_algo import dot_product_decode, confidence_update, confidence_update_lw
from utils.metrics import evaluate_benchmark, evaluate_realworld, accuracy_check
from utils.utils_log import Monitor, TimeUse, initLogger
from models.linear import linear
from models.mlp import mlp, mlp_phi
from utils.utils_seed import set_seed

# settings
# run device gpu:x or cpu

parser = argparse.ArgumentParser(
        prog='baseline demo file.',
        usage='Demo with partial labels.',
        epilog='end',
        add_help=True
    )
parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=1e-2)
parser.add_argument('-wd', help='weight decay', type=float, default=1e-4)
parser.add_argument('-bs', help='batch size', type=int, default=256)
parser.add_argument('-ep', help='number of epochs', type=int, default=500)
parser.add_argument('-dt', help='type of the dataset', type=str, choices=['benchmark', 'realworld', 'uci'])
# parser.add_argument('-ds', help='specify a dataset', type=str, choices=['mnist', 'fmnist', 'kmnist', 'cifar10', 'cifar100', 'lost', 'MSRCv2', 'birdac', 'spd', 'LYN'])
parser.add_argument('-ds', help='specify a dataset', type=str, choices=['mnist', 'fmnist', 'kmnist', 'cifar10', 'cifar100', 'cub200',
                                                                        'FG_NET', 'lost', 'MSRCv2', 'Mirflickr', 'BirdSong',
                                                                        'malagasy', 'Soccer_Player', 'Yahoo!_News', 'italian'])
parser.add_argument('-partial_type', help='flipping strategy', type=str, default='random', choices=['random', 'feature'])
parser.add_argument('-loss', type=str, choices=['proden', 'rc', 'cc', 'lws'])
parser.add_argument('-gpu', type=int, default=0)

# args = extract_args()
args = parser.parse_args()
device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else 'cpu')
logger, save_dir = initLogger(args)


def train_benchmark(config):
    # data and model
    with TimeUse("Extract Data", logger):
        set_seed(0)
        train_X, train_Y, test_X, test_Y, valid_X, valid_Y = next(extract_data(config))
        set_seed(int(time.time()) % (2 ** 16))
    num_samples = train_X.shape[0]
    train_X_shape = train_X.shape
    train_X = train_X.view((num_samples, -1))
    num_features = train_X.shape[-1]
    train_X = train_X.view(train_X_shape)
    num_classes = train_Y.shape[-1]
    with TimeUse("Create Model", logger):
        # net = resnet(depth=32, n_outputs = num_classes)
        net = create_model_for_baseline(args, num_features=num_features, num_classes=num_classes)
        net.to(device)
    if config.dt == 'benchmark':
        train_p_Y, avgC = partialize(config, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y,
                                     dim=num_features, device=device)
    if config.dt == 'realworld':
        train_p_Y = train_Y
        avgC = torch.sum(train_Y) / train_Y.size(0)
    # logger.info("Net:\n", net)
    logger.info("The Training Set has {} samples and {} classes".format(num_samples, num_features))
    logger.info("Average Candidate Labels is {:.4f}".format(avgC))
    # train_loader = create_train_loader(train_X, train_Y, train_p_Y)
    train_loader = create_train_loader(train_X, train_Y, train_p_Y, batch_size=config.bs)
    valid_loader = create_test_loader(valid_X, valid_Y, batch_size=config.bs)
    test_loader = create_test_loader(test_X, test_Y, batch_size=config.bs)
    opt = torch.torch.optim.SGD(list(net.parameters()), lr=config.lr, weight_decay=config.wd, momentum=0.9)
    partial_weight = train_loader.dataset.train_p_Y.clone().detach().to(device)
    partial_weight = partial_weight / partial_weight.sum(dim=1, keepdim=True)
    confidence = partial_weight
    best_valid = 0
    best_test = 0
    for epoch in range(args.ep):
        net.train()
        for features, targets, trues, indexes in train_loader:
            features, targets, trues = map(lambda x: x.to(device), (features, targets, trues))
            phi, outputs = net(features)
            L_ce, _, _ = lws_loss(outputs, targets.float(), confidence, indexes, 1, 1, None)
            opt.zero_grad()
            L_ce.backward()
            opt.step()
            confidence = confidence_update_lw(net, confidence, features, targets, indexes)
        net.eval()
        if config.ds not in ['cub200']:
            valid_acc = evaluate_benchmark(net, valid_X, valid_Y, device)
            test_acc = evaluate_benchmark(net, test_X, test_Y, device)
        else:
            valid_acc = accuracy_check(valid_loader, net, device)
            test_acc = accuracy_check(test_loader, net, device)
        # valid_acc = evaluate_benchmark(net, valid_X, valid_Y, device)
        # test_acc = evaluate_benchmark(net, test_X, test_Y, device)
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_test = test_acc
        logger.info("Epoch {}, valid acc: {:.4f}, test acc: {:.4f}".format(epoch, valid_acc, test_acc))
    logger.info('early stopping results: valid acc: {:.4f}, test acc: {:.4f}'.format(best_valid, best_test))


# enter
if __name__ == "__main__":
    try:
        if args.dt in ["benchmark", "realworld"]:
            train_benchmark(args)
        # if args.dt == "realworld":
        #     if args.ds not in ['spd', 'LYN']:
        #         train_realworld(args)
        #     else:
        #         train_realworld2(args)
    except Exception as e:
        logger.error("Error : " + str(e))
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())

