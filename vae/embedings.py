from os.path import splitext, basename
from glob import glob
import torch
import sys
import re


class Embeder():
    
    def __init__(self, embedings_file='embedings.txt'):
        self.embedings = self.load_embedings(embedings_file)

    def load_embedings(self, filename):
        embed = {}
        with open(filename, 'r') as f:
            for line in f.readlines():
                l = line.strip().split()
                char = l[0]
                vec = map(float, l[1:])
                embed[char] = torch.FloatTensor(vec)
        return embed

    def get_embeding(self, text):
        embed = torch.stack([self.embedings[c] for c in text])
        return embed.mean(0)
        
def load_favicon_names(directory, embeding_file, stop=None):
    embedings = []
    e = Embeder(embeding_file)
    for i, f in enumerate(glob('{}/*.jpg'.format(directory))):
        if stop is not None and i >= stop:
            break
        url = splitext(basename(f))[0]
        url = re.sub(r'<dot>', '.', url)
        embed = e.get_embeding(url)
        embedings.append(embed)
    return torch.stack(embedings)


