import torch
import pickle

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

class DataSet(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.y is not None:
            return self.X[index], self.y[index]
        return self.X[index]


def load_mnist(datapath, normalize=True, pad=None, return_datasets=False):
    with open(datapath+"train_images.pickle", "rb") as f:
        X_train, y_train = pickle.load(f)
    with open(datapath+"test_images.pickle", "rb") as f:
        X_test, y_test = pickle.load(f)

    if normalize:
        max_number = X_train.max()
        X_train = X_train / max_number
        X_test = X_test / max_number

    if pad:
        X_train = np.pad(X_train, [(0, 0), (pad, pad), (pad, pad)], mode='constant')
        X_test = np.pad(X_test, [(0, 0), (pad, pad), (pad, pad)], mode='constant')

    if return_datasets:
        train = DataSet(X_train, y_train)
        test = DataSet(X_test, y_test)
        return(train, test)
    return X_train, y_train, X_test, y_test


def plot_images(images, labels=None, show=True, n=9):
    nrows = int(np.sqrt(n))
    ncols = n // nrows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6))
    axs = np.ravel(axs)
    for i, ax in enumerate(axs):
        ax.imshow(images[i])
        ax.axis("off")
        if labels is not None:
            ax.set_title("Label: {}".format(labels[i]))
    if show:
        plt.show()
    return fig, axs


def wasserstein_loss(input, target):
    if np.all((target==1).cpu().numpy()):
        return -torch.mean(input)
    elif np.all((target==0).cpu().numpy()):
        return torch.mean(input)


def concatenate(tensor1, tensor2):
    assert tensor1.shape[0] == tensor2.shape[0], (
        "Tensors to concatenate must have same dim 0. Tensor1: {}. Tensor2: {}.".format(tensor1.shape[0], tensor2.shape[0])
    )
    assert len(tensor2.shape) == 2, (
        "tensor2 must have 2 dimensions. Given: {}.".format(len(tensor2.shape))
    )
    if len(tensor1.shape) == 2:
        return torch.cat((tensor1, tensor2), axis=1)
    elif len(tensor1.shape) == 4:
        batch_size = tensor2.shape[0]
        y_dim = tensor2.shape[1]
        tensor2 = torch.reshape(tensor2, shape=(batch_size, y_dim, 1, 1))
        tensor2 = torch.tile(tensor2, dims=(1, 1, *tensor1.shape[2:]))
        return torch.cat((tensor1, tensor2), axis=1)
    else:
        raise NotImplementedError("tensor1 one must have 2 or 4 dimensions. Given: {}.".format(len(tensor1.shape)))