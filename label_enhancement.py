# import
import argparse
import traceback
from copy import deepcopy
from operator import index

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.io import savemat
from torch.distributions.dirichlet import Dirichlet
import os
import pickle
import time

from torch.optim.lr_scheduler import MultiStepLR

from utils.args import extract_args, extract_args_LE
from utils.data_factory import extract_data, partialize, create_train_loader, create_test_loader, create_loader_LE
from utils.metrics_LE import metrics_LE
from utils.model_factory import create_model, create_model_LE
from utils.utils_graph import gen_adj_matrix2
from utils.utils_loss import partial_loss, alpha_loss, kl_loss, revised_target, out_d_loss, out_d_loss_LE
from utils.utils_algo import dot_product_decode
from utils.metrics import evaluate_benchmark, evaluate_realworld, accuracy_check
from utils.utils_log import Monitor, TimeUse, initLogger, initLogger_LE
from models.linear import linear
from models.mlp import mlp, mlp_phi

# settings
# run device gpu:x or cpu
from utils.utils_seed import set_seed

args = extract_args_LE()
device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
logger, save_dir = initLogger_LE(args)

def label_enhancement(config):
    # data and model
    with TimeUse("Extract Data", logger):
        # set_seed(0)
        # train_X, train_Y, test_X, test_Y, valid_X, valid_Y = next(extract_data(config))
        # dataset = 'SJAFFE'
        # dataset = 'Ar'
        # dataset = 'Yeast_spoem'
        dataset = args.ds
        data = scipy.io.loadmat('my_datasets/' + dataset + '.mat')
        log_data = scipy.io.loadmat('my_datasets/' + dataset + '_binary.mat')
        # data1 = scipy.io.loadmat('my_result_mll/' + dataset + '_mll.mat')

        features = data['features']
        logical_label = log_data['logicalLabel']
        label_distribution = data['labels']
        features = torch.from_numpy(features).float()
        logical_label = torch.from_numpy(logical_label).float()
        label_distribution = torch.from_numpy(label_distribution).float()

    num_samples = features.shape[0]
    train_X_shape = features.shape
    train_X = features.view((num_samples, -1))
    num_features = train_X.shape[-1]
    train_X = train_X.view(train_X_shape)
    num_classes = logical_label.shape[-1]
    with TimeUse("Create Model", logger):
        net, enc_d, enc_z, dec_L, dec_phi = create_model_LE(args, num_features=num_features, num_classes=num_classes)
        net, enc_d, enc_z, dec_L, dec_phi = map(lambda x: x.to(device), (net, enc_d, enc_z, dec_L, dec_phi))

    set_seed(int(time.time()) % (2 ** 16))
    logger.info("The Training Set has {} samples and {} classes".format(num_samples, num_features))
    train_loader = create_loader_LE(features, logical_label, label_distribution, num_samples)

    ## training
    logger.info("Use SGD with 0.9 momentum")
    # opt1 = torch.optim.SGD(list(dec_L.parameters()) + list(enc_d.parameters())
    #                        + list(enc_z.parameters()) + list(dec_phi.parameters()),
    #                        lr=args.lr, weight_decay=args.wd, momentum=0.9)
    opt1 = torch.optim.Adam(list(dec_L.parameters()) + list(enc_d.parameters())
                           + list(enc_z.parameters()) + list(dec_phi.parameters()),
                           lr=args.lr, weight_decay=args.wd)
    scheduler = MultiStepLR(opt1, milestones=[250], gamma=0.1)

    # d_array = deepcopy(o_array)
    d_array = torch.full(logical_label.size(), 1.0 / num_classes).to(device)
    prior_alpha = torch.Tensor(1, num_classes).fill_(1.0).to(device)
    for epoch in range(0, args.ep):
        # net.train()
        ave_loss = 0
        for features, targets, trues, indexes in train_loader:
            features, targets, trues = map(lambda x: x.to(device), (features, targets, trues))
            # _, outputs = net(features)
            # encoder for d
            _, log_alpha = enc_d(features)  # embedding
            # L_d, new_out = partial_loss(outputs, d_array[indexes, :], None)
            # L_d = F.binary_cross_entropy_with_logits(outputs, targets)
            logger.debug("log alpha : " + str(log_alpha))
            log_alpha = F.hardtanh(log_alpha, min_val=-5, max_val=5)
            alpha = torch.exp(log_alpha)
            logger.debug("alpha : " + str(alpha))
            alpha = F.hardtanh(alpha, min_val=1e-8, max_val=30)
            logger.debug("hardtanh alpha : " + str(alpha))
            # KLD loss of D
            L_kld_d = alpha_loss(alpha, prior_alpha)
            # dirichlet_sample_machine = torch.distributions.dirichlet.Dirichlet(s_alpha)
            dirichlet_sample_machine = torch.distributions.dirichlet.Dirichlet(alpha)
            d = dirichlet_sample_machine.rsample()
            # d = F.sigmoid(d)
            # encoder for z
            z_mu, z_log_var = enc_z(features, d)
            z_log_var = F.hardtanh(z_log_var, min_val=-5, max_val=5)
            z_std = torch.sqrt(torch.exp(z_log_var))
            z_std = F.hardtanh(z_std, min_val=1e-8, max_val=30)
            logger.debug("z_mu :\n" + str(z_mu))
            logger.debug("z_log_var :\n" + str(z_log_var))
            normal_sample_machine = torch.distributions.normal.Normal(z_mu, z_std)
            z = normal_sample_machine.rsample()
            # decoder for A and L
            partial_label_hat = dec_L(d)
            # partial_label_hat = torch.sigmoid(partial_label_hat)
            # A_hat = F.softmax(dot_product_decode(d), dim=1)
            # decoder for x (phi)
            x_hat = dec_phi(z, d)
            # x_hat = x_hat.view(features.shape)
            # reconstrcution Loss X, L, A
            L_recx = 1 * F.mse_loss(x_hat, features)
            L_recy = 1 * F.binary_cross_entropy_with_logits(partial_label_hat, targets)
            # d = F.softmax(d, dim=1)
            d = F.sigmoid(d)
            L_d = 1 * F.binary_cross_entropy(d, targets)
            # L_recA = 0.001 * F.mse_loss(A_hat, A[indexes, :][:, indexes].to(device))
            # L_recA = 0 * F.mse_loss(A_hat, A[indexes, :][:, indexes].to(device))
            # KLD loss of z
            L_kld_z = -torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp(), dim=1).mean()
            # L_rec = L_recx + L_recy + L_recA
            # L_o, new_out = out_d_loss_LE(outputs, d)
            # L = config.alpha * L_rec + config.beta * L_alpha + config.gamma * L_d + config.theta * L_o
            # L = config.alpha * L_rec + \
            #     config.beta * L_kld_d + \
            #     config.gamma * L_kld_z + \
            #     config.theta * L_o + \
            #     config.sigma * L_d
            L = config.alpha1 * L_recx + \
                config.alpha2 * L_recy + \
                config.beta * L_kld_d + \
                config.gamma * L_kld_z + \
                config.theta * L_d
                # config.theta * L_o + \
                # config.sigma * L_d
            ave_loss += L
            opt1.zero_grad()
            # opt2.zero_grad()
            L.backward()
            # torch.nn.utils.clip_grad_norm_(list(enc_d.parameters()) + list(enc_z.parameters()), 5)
            opt1.step()
            # opt2.step()
            # new_d = revised_target(d, o_array[indexes, :])
            new_d = F.softmax(d, dim=1)
            # new_d = config.correct * new_d + (1 - config.correct) * new_out
            d_array[indexes, :] = new_d.clone().detach()
            # o_array[indexes, :] = new_o.clone().detach()
        scheduler.step()
        # d_array = torch.full(logical_label.size(), 1.0 / num_classes).to(device)
        # d_array = label_distribution
        results = metrics_LE(d_array.cpu().numpy(), label_distribution.numpy())
        logger.info("Epoch {}, Cheb               Clark                Can                 Kl                 Cos               Intersec".format(epoch))
        logger.info("Epoch {}, results : {}".format(epoch, results))
        logger.info("Epoch {}, loss : {}".format(epoch, ave_loss))
        # logger.info("{}".format(d_array))
    saved_result = np.array(results).reshape((-1, 1))
    savemat('final_results_LE/Results/VALEN_' + dataset + '.mat', {'Result': saved_result})


# enter
if __name__ == "__main__":
    try:
        label_enhancement(args)
    except Exception as e:
        logger.error("Error : " + str(e))
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
