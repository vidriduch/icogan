from __future__ import print_function
from collections import defaultdict
from os.path import basename, splitext
from glob import glob
import numpy as np
from time import time
from torch import nn, optim
from torch.nn import functional as F
from utils import get_data_loader, save_checkpoint
from data import TRAIN_DATASETS
from torchvision.utils import save_image
from models import *
import torch


def load_favicons(directory):
    urls = []
    for f in glob('{}/*.png'.format(directory)):
        urls.append(splitext(basename(f))[0])
    return urls


def get_char2id(strings):
    ch = []
    chars = [ch.extend(list(x)) for x in urls]
    char2id = {}
    for i, x in enumerate(set(ch)):
        char2id[x] = i
    char2id = defaultdict(lambda: 'unk', char2id)
    id2char = {v: k for k, v in char2id.items()}
    return char2id, id2char


def encode_urls(urls, char2id):
    onehot_urls = []
    for url in urls:
        onehot_urls.append([char2id[x] for x in list(url)])
    return onehot_urls


def kl_loss(recon_x, x, mu, logvar):
    recon = F.binary_cross_entropy(recon_x, x, size_average=False)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl


def train(model, batch_size, log_interval):
    model.train()
    train_loss = 0
    dataset = TRAIN_DATASETS['ico']
    dataset_loader = get_data_loader(dataset, batch_size=batch_size)
    for batch_idx, (data, _) in enumerate(dataset_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = kl_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataset_loader.dataset),
                100. * batch_idx / len(dataset_loader),
                loss.item() / len(data)))

    avg_loss = train_loss / len(dataset_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))
    save_checkpoint(model, './models', epoch)

z_size = 50

urls = load_favicons('16_16_distinct')
char2id, id2char = get_char2id(urls)
onehot_urls = encode_urls(urls, char2id)
device = torch.device("cuda")
model = VAE(z_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 150
batch_size = 256
log_interval = 50

# image dimensions
ch, w, h = 3, 16, 16


for epoch in range(1, epochs + 1):
    train(model, batch_size, log_interval)
    with torch.no_grad():
        sample = torch.randn(batch_size, z_size).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(batch_size, ch, w, h),
                   'results/sample_' + str(epoch) + '.png')
