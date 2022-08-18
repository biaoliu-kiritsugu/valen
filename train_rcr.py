# import
import argparse
import traceback
from copy import deepcopy, copy
from operator import index
import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
import os
import pickle
import time
import utils

from torch.optim.lr_scheduler import MultiStepLR

from utils import utils_crc
from utils.args import extract_args
from utils.data_factory import extract_data, partialize, create_train_loader, create_test_loader, \
    create_train_loader_DA, extract_data_DA
from utils.model_factory import create_model, create_model_for_baseline
from utils.utils_graph import gen_adj_matrix2
from utils.utils_loss import partial_loss, alpha_loss, kl_loss, revised_target, out_d_loss, out_d_loss_DA, \
    out_cons_loss_DA
from utils.utils_algo import dot_product_decode
from utils.metrics import evaluate_benchmark, evaluate_realworld, accuracy_check
from utils.utils_log import Monitor, TimeUse, initLogger
from models.linear import linear
from models.mlp import mlp, mlp_phi

# settings
# run device gpu:x or cpu
from utils.utils_seed import set_seed


parser = argparse.ArgumentParser(
    prog='baseline demo file.',
    usage='Demo with partial labels.',
    epilog='end',
    add_help=True
)
# nohup python -u train_rcr.py -lr 5e-2 -wd 1e-3 -ep 250 -dt benchmark -ds cifar10 -lam 1 -partial_type random -gpu 0
parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=5e-2)
parser.add_argument('-wd', help='weight decay', type=float, default=1e-3)
parser.add_argument('-bs', help='batch size', type=int, default=256)
parser.add_argument('-ep', help='number of epochs', type=int, default=500)
parser.add_argument('-dt', help='type of the dataset', type=str, default='realworld',
                    choices=['benchmark', 'realworld', 'uci'])
# parser.add_argument('-ds', help='specify a dataset', type=str, default='cifar10', choices=['mnist', 'fmnist', 'kmnist', 'cifar10', 'cifar100', 'lost', 'MSRCv2', 'birdac', 'spd', 'LYN'])
parser.add_argument('-ds', help='specify a dataset', type=str, default='italian',
                    choices=['mnist', 'fmnist', 'kmnist', 'cifar10', 'cifar100', 'cub200',
                             'FG_NET', 'lost', 'MSRCv2', 'Mirflickr', 'BirdSong',
                             'malagasy', 'Soccer_Player', 'Yahoo!_News', 'italian'])
parser.add_argument('-lam', default=1, type=float)
parser.add_argument('-partial_type', help='flipping strategy', type=str, default='feature',
                    choices=['random', 'feature'])
parser.add_argument('-loss', type=str, default='crc')
parser.add_argument('-gpu', type=int, default=0)

# args = extract_args()
args = parser.parse_args()
# args = extract_args()
device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else 'cpu')
logger, save_dir = initLogger(args)


# train benchmark
def train_benchmark(config):
    # data and model
    with TimeUse("Extract Data", logger):
        set_seed(0)
        train_X_DA, train_X, train_Y, test_X, test_Y, valid_X, valid_Y = next(extract_data_DA(config))
    print(train_X.shape)
    print(train_X_DA.shape)
    num_samples = train_X.shape[0]
    train_X_shape = train_X.shape
    train_X = train_X.view((num_samples, -1))
    num_features = train_X.shape[-1]
    train_X = train_X.view(train_X_shape)
    num_classes = train_Y.shape[-1]
    with TimeUse("Create Model", logger):
        consistency_criterion = torch.nn.KLDivLoss(reduction='batchmean').cuda()
        net = create_model_for_baseline(args, num_features=num_features, num_classes=num_classes)
        net.to(device)
    train_p_Y, avgC = partialize(config, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y,
                                 dim=num_features, device=device)
    set_seed(int(time.time()) % (2 ** 16))
    logger.info("The Training Set has {} samples and {} classes".format(num_samples, num_features))
    logger.info("Average Candidate Labels is {:.4f}".format(avgC))
    # train_loader = create_train_loader(train_X, train_Y, train_p_Y)
    train_loader = create_train_loader_DA(train_X_DA, train_Y, train_p_Y, batch_size=config.bs, ds=config.ds)
    valid_loader = create_test_loader(valid_X, valid_Y, batch_size=config.bs)
    test_loader = create_test_loader(test_X, test_Y, batch_size=config.bs)

    confidence = deepcopy(train_loader.dataset.train_p_Y)
    confidence = confidence / confidence.sum(axis=1)[:, None]
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200], last_epoch=-1)
    best_valid = 0
    best_test = 0
    for epoch in range(0, args.ep):
        net.train()
        for features, features1, features2, targets, trues, indexes in train_loader:
            features, features1, features2, targets, trues = map(lambda x: x.to(device), (features, features1, features2, targets, trues))
            _, y_pred_aug0 = net(features)
            _, y_pred_aug1 = net(features1)
            _, y_pred_aug2 = net(features2)

            y_pred_aug0_probas_log = torch.log_softmax(y_pred_aug0, dim=-1)
            y_pred_aug1_probas_log = torch.log_softmax(y_pred_aug1, dim=-1)
            y_pred_aug2_probas_log = torch.log_softmax(y_pred_aug2, dim=-1)

            y_pred_aug0_probas = torch.softmax(y_pred_aug0, dim=-1)
            y_pred_aug1_probas = torch.softmax(y_pred_aug1, dim=-1)
            y_pred_aug2_probas = torch.softmax(y_pred_aug2, dim=-1)

            # consist loss
            consist_loss0 = consistency_criterion(y_pred_aug0_probas_log,
                                                  torch.tensor(confidence[indexes]).float().cuda())
            consist_loss1 = consistency_criterion(y_pred_aug1_probas_log,
                                                  torch.tensor(confidence[indexes]).float().cuda())
            consist_loss2 = consistency_criterion(y_pred_aug2_probas_log,
                                                  torch.tensor(confidence[indexes]).float().cuda())
            # supervised loss
            super_loss = -torch.mean(
                torch.sum(torch.log(1.0000001 - F.softmax(y_pred_aug0, dim=1)) * (1 - targets), dim=1))
            # dynamic lam
            lam = min((epoch / 100) * args.lam, args.lam)

            # Unified loss
            final_loss = lam * (consist_loss0 + consist_loss1 + consist_loss2) + super_loss

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            # update confidence
            utils_crc.confidence_update(confidence, y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, targets, indexes)
        scheduler.step()
        net.eval()
        if config.ds not in ['cub200']:
            valid_acc = evaluate_benchmark(net, valid_X, valid_Y, device)
            test_acc = evaluate_benchmark(net, test_X, test_Y, device)
        else:
            valid_acc = accuracy_check(valid_loader, net, device)
            test_acc = accuracy_check(test_loader, net, device)
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_test = test_acc
        logger.info("Epoch {}, valid acc: {:.4f}, test acc: {:.4f}".format(epoch, valid_acc, test_acc))
    logger.info('early stopping results: valid acc: {:.4f}, test acc: {:.4f}'.format(best_valid, best_test))

# enter
if __name__ == "__main__":
    try:
        if args.dt == "benchmark":
            train_benchmark(args)
    except Exception as e:
        logger.error("Error : " + str(e))
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
