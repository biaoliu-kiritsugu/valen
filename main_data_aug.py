# import
import argparse
import traceback
from copy import deepcopy
from operator import index
import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
import os
import pickle
import time

from torch.optim.lr_scheduler import MultiStepLR

from utils.args import extract_args
from utils.data_factory import extract_data, partialize, create_train_loader, create_test_loader, \
    create_train_loader_DA, extract_data_DA
from utils.model_factory import create_model, create_model_DA
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

args = extract_args()
device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else 'cpu')
logger, save_dir = initLogger(args)


def warm_up_benchmark(config, model, train_loader, test_X, test_Y, test_loader=None):
    opt = torch.torch.optim.SGD(list(model.parameters()), lr=config.lr, weight_decay=config.wd, momentum=0.9)
    partial_weight = train_loader.dataset.train_p_Y.clone().detach().to(device)
    partial_weight = partial_weight / partial_weight.sum(dim=1, keepdim=True)
    logger.info("Begin warm-up, warm up epoch {}".format(config.warm_up))
    for _ in range(0, config.warm_up):
        for features, _, _, targets, trues, indexes in train_loader:
            features, targets, trues = map(lambda x: x.to(device), (features, targets, trues))
            phi, outputs = model(features)
            L_ce, new_labels = partial_loss(outputs, partial_weight[indexes, :].clone().detach(), None)
            partial_weight[indexes, :] = new_labels.clone().detach()
            opt.zero_grad()
            L_ce.backward()
            opt.step()
    if config.ds not in ['cub200']:
        test_acc = evaluate_benchmark(model, test_X, test_Y, device)
    else:
        test_acc = accuracy_check(test_loader, model, device)
    # test_acc = evaluate_benchmark(model, test_X, test_Y, device)
    logger.info("After warm up, test acc: {:.4f}".format(test_acc))
    logger.info("Extract feature.")
    feature_extracted = torch.zeros((train_loader.dataset.train_X.shape[0], phi.shape[-1])).to(device)
    with torch.no_grad():
        for features, _, _, targets, trues, indexes in train_loader:
            features, targets, trues = map(lambda x: x.to(device), (features, targets, trues))
            feature_extracted[indexes, :] = model(features)[0]
    return model, feature_extracted, partial_weight


def warm_up_realworld(config, model, train_loader, test_X, test_Y):
    opt = torch.torch.optim.SGD(list(model.parameters()), lr=config.lr, weight_decay=config.wd, momentum=0.9)
    partial_weight = train_loader.dataset.train_p_Y.clone().detach().to(device)
    partial_weight = partial_weight / partial_weight.sum(dim=1, keepdim=True)
    logger.info("Begin warm-up, warm up epoch {}".format(config.warm_up))
    for _ in range(0, config.warm_up):
        for features, targets, trues, indexes in train_loader:
            features, targets, trues = map(lambda x: x.to(device), (features, targets, trues))
            outputs = model(features)
            L_ce, new_labels = partial_loss(outputs, partial_weight[indexes, :].clone().detach(), None)
            partial_weight[indexes, :] = new_labels.clone().detach()
            opt.zero_grad()
            L_ce.backward()
            opt.step()
    test_acc = evaluate_realworld(model, test_X, test_Y, device)
    logger.info("After warm up, test acc: {:.4f}".format(test_acc))
    return model, partial_weight


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
        net, enc_d, enc_z, dec_L, dec_phi = create_model_DA(args, num_features=num_features, num_classes=num_classes)
        net, enc_d, enc_z, dec_L, dec_phi = map(lambda x: x.to(device), (net, enc_d, enc_z, dec_L, dec_phi))
    train_p_Y, avgC = partialize(config, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y,
                                 dim=num_features, device=device)
    set_seed(int(time.time()) % (2 ** 16))
    # logger.info("Net:\n", net)
    # logger.info("Encoder:\n", enc_d, enc_z)
    # logger.info("Decoder:\n", dec_L, dec_phi)
    logger.info("The Training Set has {} samples and {} classes".format(num_samples, num_features))
    logger.info("Average Candidate Labels is {:.4f}".format(avgC))
    # train_loader = create_train_loader(train_X, train_Y, train_p_Y)
    train_loader = create_train_loader_DA(train_X_DA, train_Y, train_p_Y, batch_size=config.bs, ds=config.ds)
    valid_loader = create_test_loader(valid_X, valid_Y, batch_size=config.bs)
    test_loader = create_test_loader(test_X, test_Y, batch_size=config.bs)
    # warm up
    net, feature_extracted, o_array = warm_up_benchmark(config, net, train_loader, test_X, test_Y, test_loader)
    if config.ds in ['cub200']:
        enc_d = deepcopy(net)
    # if config.partial_type == "feature" and config.ds in ["kmnist", "cifar10"]:
    #     logger.info("Copy Net.")
    #     enc = deepcopy(net)
    # compute adj matrix
    logger.info("Compute adj maxtrix or Read.")
    with TimeUse("Adj Maxtrix", logger):
        adj = gen_adj_matrix2(feature_extracted.cpu().numpy(), k=config.knn,
                              path=os.path.abspath("middle/adjmatrix/" + args.dt + "/" + args.ds + ".npy"))
    with TimeUse("Adj to Dense", logger):
        A = adj.to_dense()
    with TimeUse("Adj to Device", logger):
        adj = adj.to(device)
    # compute gcn embedding
    with TimeUse("Spmm", logger):
        embedding = train_X.to(device)
    prior_alpha = torch.Tensor(1, num_classes).fill_(1.0).to(device)
    ## training
    logger.info("Use SGD with 0.9 momentum")
    opt1 = torch.optim.SGD(list(net.parameters()) + list(dec_L.parameters()) + list(enc_d.parameters()),
                           lr=args.lr, weight_decay=args.wd, momentum=0.9)
    # opt1 = torch.optim.Adam(list(net.parameters()) + list(enc_d.parameters()) + list(dec_L.parameters()),
    #                       lr=0.001, weight_decay=0.001)
    opt2 = torch.optim.Adam(list(enc_z.parameters()) + list(dec_phi.parameters()),
                            lr=0.001, weight_decay=0.001)
    scheduler = MultiStepLR(opt1, milestones=[100, 150], gamma=0.1)

    mit = Monitor(num_samples, num_classes, logger)
    d_array = deepcopy(o_array)
    best_valid = 0
    best_test = 0
    for epoch in range(0, args.ep):
        net.train()
        for features, features1, features2, targets, trues, indexes in train_loader:
            features, features1, features2, targets, trues = map(lambda x: x.to(device), (features, features1, features2, targets, trues))
            _, outputs = net(features)
            _, outputs1 = net(features1)
            _, outputs2 = net(features2)
            # encoder for d
            _, log_alpha = enc_d(features)  # embedding
            L_d, new_out = partial_loss(outputs, d_array[indexes, :], None)
            log_alpha = F.hardtanh(log_alpha, min_val=-5, max_val=5)
            alpha = torch.exp(log_alpha)
            alpha = F.hardtanh(alpha, min_val=1e-8, max_val=30)
            # dirichlet_sample_machine = torch.distributions.dirichlet.Dirichlet(s_alpha)
            dirichlet_sample_machine = torch.distributions.dirichlet.Dirichlet(alpha)
            d = dirichlet_sample_machine.rsample()
            # features 1
            _, log_alpha1 = enc_d(features1)  # embedding
            log_alpha1 = F.hardtanh(log_alpha1, min_val=-5, max_val=5)
            alpha1 = torch.exp(log_alpha1)
            alpha1 = F.hardtanh(alpha1, min_val=1e-8, max_val=30)
            # dirichlet_sample_machine = torch.distributions.dirichlet.Dirichlet(s_alpha)
            dirichlet_sample_machine1 = torch.distributions.dirichlet.Dirichlet(alpha1)
            d1 = dirichlet_sample_machine1.rsample()
            # features 2
            _, log_alpha2 = enc_d(features2)  # embedding
            log_alpha2 = F.hardtanh(log_alpha2, min_val=-5, max_val=5)
            alpha2 = torch.exp(log_alpha2)
            alpha2 = F.hardtanh(alpha2, min_val=1e-8, max_val=30)
            # dirichlet_sample_machine = torch.distributions.dirichlet.Dirichlet(s_alpha)
            dirichlet_sample_machine2 = torch.distributions.dirichlet.Dirichlet(alpha2)
            d2 = dirichlet_sample_machine2.rsample()

            # KLD loss of D
            L_kld_d = alpha_loss(alpha, prior_alpha)
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
            A_hat = F.softmax(dot_product_decode(d), dim=1)
            # decoder for x (phi)
            x_hat = dec_phi(z, d)
            # x_hat = x_hat.view(features.shape)
            # reconstrcution Loss X, L, A
            L_recx = 1 * F.mse_loss(x_hat, features)
            L_recy = 1 * F.binary_cross_entropy_with_logits(partial_label_hat, targets)
            L_recA = 0.001 * F.mse_loss(A_hat, A[indexes, :][:, indexes].to(device))
            # L_recA = 0 * F.mse_loss(A_hat, A[indexes, :][:, indexes].to(device))
            # KLD loss of z
            L_kld_z = -torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp(), dim=1).mean()
            # L_rec = L_recx + L_recy + L_recA
            L_o = out_d_loss(outputs, d, targets)
            L_cons1 = out_d_loss_DA(d_array[indexes, :], d, d1, d2, consistency_criterion)
            L_cons2 = out_cons_loss_DA(d_array[indexes, :], outputs, outputs1, outputs2, consistency_criterion)
            # L = config.alpha * L_rec + config.beta * L_alpha + config.gamma * L_d + config.theta * L_o
            # L = config.alpha * L_rec + \
            #     config.beta * L_kld_d + \
            #     config.gamma * L_kld_z + \
            #     config.theta * L_o + \
            #     config.sigma * L_d
            L = config.alpha1 * L_recx + \
                config.alpha2 * L_recy + \
                config.alpha3 * L_recA + \
                config.beta * L_kld_d + \
                config.gamma * L_kld_z + \
                config.theta * L_o + \
                config.sigma * L_d + \
                config.lam1 * L_cons1 + \
                config.lam2 * L_cons2
            opt1.zero_grad()
            opt2.zero_grad()
            L.backward()
            # torch.nn.utils.clip_grad_norm_(list(enc_d.parameters()) + list(enc_z.parameters()), 5)
            opt1.step()
            opt2.step()
            new_d = revised_target(d, o_array[indexes, :])
            new_d = config.correct * new_d + (1 - config.correct) * new_out
            d_array[indexes, :] = new_d.clone().detach()
            # o_array[indexes, :] = new_o.clone().detach()
        scheduler.step()
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
        if args.dt == "benchmark":
            train_benchmark(args)
    except Exception as e:
        logger.error("Error : " + str(e))
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
