from torch.utils.data import DataLoader
import torch
import os


def get_data_loader(dataset, batch_size, cuda=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                     **({'num_workers': 1, 'pin_memory': True} if cuda else {})
                     )


def save_checkpoint(model, model_dir, epoch):
    path = os.path.join(model_dir, model.name)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({'state': model.state_dict(), 'epoch': epoch}, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))
