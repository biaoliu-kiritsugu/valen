from copy import deepcopy

from models.convnet import convnet
from models.linear import linear
from models.mlp import mlp_feature, mlp_phi
from models.VGAE import VAE_Bernulli_Decoder, Decoder_L, CONV_Decoder
from models.resnet import resnet
from models.VGAE import CONV_Encoder

def create_model(config, **args):
    if config.dt == "benchmark":
        if config.ds in ['mnist', 'kmnist', 'fmnist']:
            if config.partial_type == "random":
                net = mlp_feature(args['num_features'], args['num_features'], args['num_classes'])
            if config.partial_type == "feature":
                net = mlp_phi(args['num_features'], args['num_classes'])
        if config.ds in ['cifar10']:
            net = resnet(depth=32, n_outputs = args['num_classes'])
        enc_d = deepcopy(net)
        # for cifar
        enc_z = CONV_Encoder(in_channels=3,
                             feature_dim=32,
                             num_classes=args['num_classes'],
                             hidden_dims=[32, 64, 128, 256],
                             z_dim=config.z_dim)
        # dec = VAE_Bernulli_Decoder(args['num_classes'], args['num_features'], args['num_features'])
        dec_L = Decoder_L(num_classes=10, hidden_dim=128)
        dec_phi = CONV_Decoder(num_classes=args['num_classes'],
                               hidden_dims=[256, 128, 64, 32],
                               z_dim=config.z_dim)
        return net, enc_d, enc_z, dec_L, dec_phi
    if config.dt == "realworld":
        net = linear(args['num_features'],args['num_classes'])
        enc = deepcopy(net)
        dec = VAE_Bernulli_Decoder(args['num_classes'], args['num_features'], args['num_features'])
        return net, enc, dec
        
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
    return net
