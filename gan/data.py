from torchvision import datasets, transforms
from os.path import basename, splitext
from embedings import Embeder
import re

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImagePathFolder(datasets.ImageFolder):
    def __init__(self, root, embeding_file='../embedings.txt', transform=None):
        super(ImagePathFolder, self).__init__(root, transform=transform)
        self.imgs = self.samples
        self.e = Embeder(embeding_file)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        name = splitext(basename(path))[0]
        name = re.sub(r'<dot>', '.', name)
        embeded = self.e.get_embeding(name)

        return sample, target, embeded

TRAIN_DATASETS = {
    'ico':  datasets.ImageFolder(root='../icons',
                                 transform=transforms.Compose([
                                    transforms.ToTensor(),
                                 ])),

    'ico_path': ImagePathFolder(root='../icons',
                                 transform=transforms.Compose([
                                    transforms.ToTensor(),
                                 ])),
}

DATASET_CONFIGS = {
    'ico': {'size': 16, 'channels': 3, 'classes': 100},
    'ico_path': {'size': 16, 'channels': 3, 'classes': 100},
}
