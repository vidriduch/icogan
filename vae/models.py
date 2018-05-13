from torch import nn, optim
from torch.nn import functional as F
from utils import get_data_loader, save_checkpoint
from data import TRAIN_DATASETS
from torchvision.utils import save_image
from models import *
import torch

class CVAE(nn.Module):

    def __init__(self, z_size):
        super(CVAE, self).__init__()
        self.name = 'conv_vae'
        self.z_size = z_size

        self.encoder = nn.Sequential(
            self._conv(3, 16, 4, stride=2, padding=1),
            self._conv(16, 32, 3, stride=2, padding=1),
            self._conv(32, 64, 3),
            self._conv(64, 128, 1),
        )

        self.feature_volume = 128 * 4
        self.mu = self._linear(self.feature_volume, z_size, relu=False)
        self.logvar = self._linear(self.feature_volume, z_size, relu=False)

        self.project = self._linear(z_size, self.feature_volume, relu=False)

        self.decoder = nn.Sequential(
            self._deconv(128, 64, 1),
            self._deconv(64, 32, 3),
            self._deconv(32, 16, 3, stride=2),
            self._deconv(16, 3, 4, stride=2, padding=2),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        projected = self.project(z).view(-1, 128, 2, 2)
        recon = self.decoder(projected)
        return recon

    def encode(self, x):
        encoded = self.encoder(x)
        mean, logvar = self.q(encoded)
        return mean, logvar

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.mu(unrolled), self.logvar(unrolled)

    # ======
    # Layers
    # ======

    def _conv(self, channel_size, kernel_num, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(
                channel_size, kernel_num,
                kernel_size=kernel_size,
                stride=stride, padding=padding,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _deconv(self, channel_num, kernel_num, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(
                channel_num, kernel_num,
                kernel_size=kernel_size,
                stride=stride, padding=padding,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)


class VAE(nn.Module):

    def __init__(self, z_size):
        super(VAE, self).__init__()

        self.name = 'simple_vae'
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 384)
        self.fc3 = nn.Linear(384, 256)
        self.fc4 = nn.Linear(256, 128)

        self.fc41 = nn.Linear(128, z_size)
        self.fc42 = nn.Linear(128, z_size)

        self.fc5 = nn.Linear(z_size, 128)
        self.fc6 = nn.Linear(128, 256)
        self.fc7 = nn.Linear(256, 384)
        self.fc8 = nn.Linear(384, 512)
        self.fc9 = nn.Linear(512, 768)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        return self.fc41(h4), self.fc42(h4)

    def decode(self, z):
        h5 = F.relu(self.fc5(z))
        h6 = F.relu(self.fc6(h5))
        h7 = F.relu(self.fc7(h6))
        h8 = F.relu(self.fc8(h7))
        return F.sigmoid(self.fc9(h8))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 768))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

