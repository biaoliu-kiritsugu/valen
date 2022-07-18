from copy import deepcopy

from models.convnet import convnet
from models.linear import linear
from models.mlp import mlp_feature, mlp_phi
from models.VGAE import VAE_Bernulli_Decoder, Decoder_L, CONV_Decoder, CONV_Encoder_MNIST, CONV_Decoder_MNIST, \
    Z_Encoder, X_Decoder
from models.resnet import resnet
from models.VGAE import CONV_Encoder
from partial_models.linear_mlp_models import linear_model


def create_model(config, **args):
    if config.dt == "benchmark":
        if config.ds in ['mnist', 'kmnist', 'fmnist']:
            if config.partial_type == "random":
                net = mlp_feature(args['num_features'], args['num_features'], args['num_classes'])
            if config.partial_type == "feature":
                # net = mlp_phi(args['num_features'], args['num_classes'])
                net = mlp_feature(args['num_features'], args['num_features'], args['num_classes'])
        if config.ds in ['cifar10']:
            net = resnet(depth=32, n_outputs=args['num_classes'])
        if config.ds in ['cifar100']:
            net = convnet(input_channel=3, n_outputs=args['num_classes'], dropout_rate=0.25)
        enc_d = deepcopy(net)
        # for cifar
        if config.ds in ['cifar10', 'cifar100']:
            enc_z = CONV_Encoder(in_channels=3,
                                 feature_dim=32,
                                 num_classes=args['num_classes'],
                                 hidden_dims=[32, 64, 128, 256],
                                 z_dim=config.z_dim)
            # dec = VAE_Bernulli_Decoder(args['num_classes'], args['num_features'], args['num_features'])
            dec_phi = CONV_Decoder(num_classes=args['num_classes'],
                                   hidden_dims=[256, 128, 64, 32],
                                   z_dim=config.z_dim)
        if config.ds in ['mnist', 'kmnist', 'fmnist']:
            enc_z = CONV_Encoder_MNIST(in_channels=1,
                                       feature_dim=28,
                                       num_classes=args['num_classes'],
                                       hidden_dims=[32, 64, 128, 256],
                                       z_dim=config.z_dim)
            # dec = VAE_Bernulli_Decoder(args['num_classes'], args['num_features'], args['num_features'])
            dec_phi = CONV_Decoder_MNIST(num_classes=args['num_classes'],
                                         hidden_dims=[256, 128, 64, 32],
                                         z_dim=config.z_dim)
        dec_L = Decoder_L(num_classes=args['num_classes'], hidden_dim=128)
        return net, enc_d, enc_z, dec_L, dec_phi
    if config.dt == "realworld":
        net = linear(args['num_features'], args['num_classes'])
        enc_d = deepcopy(net)
        enc_z = Z_Encoder(feature_dim=args['num_features'],
                          num_classes=args['num_classes'],
                          num_hidden_layers=2,
                          hidden_size=25,
                          # z_dim=int(args['num_features'] / 10)
                          z_dim=int(args['num_classes'] * 1.5)
                          )
        dec_phi = X_Decoder(feature_dim=args['num_features'],
                            num_classes=args['num_classes'],
                            num_hidden_layers=2,
                            hidden_size=25,
                            # z_dim=int(args['num_features'] / 10),
                            z_dim = int(args['num_classes'] * 1.5)
        )
        dec_L = Decoder_L(num_classes=args['num_classes'], hidden_dim=128)
        return net, enc_d, enc_z, dec_L, dec_phi


def create_model_for_baseline(config, **args):
    if config.ds == 'cifar10':
        net = resnet(depth=32, n_outputs=10)
    if config.ds in ['mnist', 'kmnist', 'fmnist']:
        if config.partial_type == 'random':
            net = mlp_feature(args['num_features'], args['num_features'], args['num_classes'])
        if config.partial_type == 'feature':
            # net = mlp_phi(args['num_features'], args['num_classes'])
            net = mlp_feature(args['num_features'], args['num_features'], args['num_classes'])
    if config.ds in ['cifar100']:
        net = convnet(input_channel=3, n_outputs=args['num_classes'], dropout_rate=0.25)
    if config.dt == 'realworld':
        net = linear_model(input_dim=args['num_features'], output_dim=args['num_classes'])
    return net
