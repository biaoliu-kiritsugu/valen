import torch
from torch import nn


class CONV_Encoder(nn.Module):
    """
    Encoder: D and \phi --> z
    """

    def __init__(self, in_channels=3, feature_dim=32, num_classes=2, hidden_dims=[32, 64, 128, 256], z_dim=128):
        super().__init__()
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        self.embed_class = nn.Linear(num_classes, feature_dim * feature_dim)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.in_channels = in_channels
        in_channels += 1
        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, z_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] * 4, z_dim)

    def forward(self, x, partial_label):
        embedded_class = self.embed_class(partial_label)
        x = x.view(x.size(0), self.in_channels, self.feature_dim, self.feature_dim)
        embedded_class = embedded_class.view(-1, self.feature_dim, self.feature_dim).unsqueeze(1)
        embedded_input = self.embed_data(x)
        x = torch.cat([embedded_input, embedded_class], dim=1)
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

class CONV_Decoder(nn.Module):
    """
    Decoder: z, d --> \phi
    """

    def __init__(self, num_classes=10, hidden_dims=[256, 128, 64, 32], z_dim=128):
        super().__init__()
        self.decoder_input = nn.Linear(z_dim + num_classes, hidden_dims[0] * 4)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())

    def forward(self, z, d):
        out = torch.cat((z, d), dim=1)
        out = self.decoder_input(out)
        out = out.view(-1, 2048, 2, 2)
        out = self.decoder(out)
        out = self.final_layer(out)
        return out

enc_z = CONV_Encoder(in_channels=3,
                     feature_dim=256,
                     num_classes=200,
                     hidden_dims=[32, 64, 128, 256, 512, 1024, 2048],
                     z_dim=64)
X = torch.rand((2, 3, 256, 256))
D = torch.rand((2, 200))
mu, log_var = enc_z(X, D)
# print()
dec_phi = CONV_Decoder(num_classes=200,
                       hidden_dims=[2048, 1024, 512, 256, 128, 64, 32],
                       z_dim=64)
Z = torch.rand((2, 64))
D = torch.rand((2, 200))
out = dec_phi(Z, D)
print(out.size())
