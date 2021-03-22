import torch

import numpy as np
import utils.utils as utils

from torch import nn
from models.GAN import VanillaGAN, WassersteinGAN, WassersteinGANGP
from utils.layers import LayerReshape, LayerDebug

if __name__ == '__main__':

    datapath = "./Data/mnist/"
    X_train, y_train, X_test, y_test = utils.load_mnist(datapath, normalize=True, pad=2, return_datasets=False)
    lr_gen = 0.0001
    lr_adv = 0.00005
    epochs = 1
    batch_size = 64

    X_train = X_train.reshape((-1, 1, 32, 32))
    X_test = X_test.reshape((-1, 1, 32, 32))
    z_dim = 2
    out_size = X_train.shape[1:]

    #########################################################################
    # Fully connected network
    #########################################################################
    generator_architecture = [
        (nn.Linear, {"in_features": z_dim, "out_features": 128}),
        (nn.LeakyReLU, {"negative_slope": 0.2}),
        (nn.Linear, {"out_features": 256}),
        (nn.LeakyReLU, {"negative_slope": 0.2}),
        (nn.BatchNorm1d, {}),
        (nn.Linear, {"out_features": 512}),
        (nn.LeakyReLU, {"negative_slope": 0.2}),
        (nn.BatchNorm1d, {}),
        (nn.Linear, {"out_features": 1024}),
        (nn.LeakyReLU, {"negative_slope": 0.2}),
        (nn.BatchNorm1d, {}),
        (nn.Linear, {"out_features": int(np.prod(out_size))}),
        (LayerReshape, {"shape": out_size}),
        (nn.Sigmoid, {})
    ]
    adversariat_architecture = [
        (nn.Flatten, {}),
        (nn.Linear, {"in_features": int(np.prod(out_size)), "out_features": 512}),
        (nn.LeakyReLU, {"negative_slope": 0.2}),
        (nn.Linear, {"out_features": 256}),
        (nn.LeakyReLU, {"negative_slope": 0.2}),
        (nn.Linear, {"out_features": 1}),
        (nn.Sigmoid, {})
    ]


    #########################################################################
    # Convolutional network
    #########################################################################
    # z_dim = [1, 8, 8]
    # out_size = X_train.shape[1:]
    # generator_architecture = [
    #     (nn.ConvTranspose2d, {"in_channels": 1, "out_channels": 64, "kernel_size": 4, "stride": 2, "padding": 1}),
    #     (nn.LeakyReLU, {"negative_slope": 0.2}),

    #     (nn.ConvTranspose2d, {"out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 1}),
    #     (nn.LeakyReLU, {"negative_slope": 0.2}),
    #     (nn.BatchNorm2d, {}),

    #     (nn.Conv2d, {"out_channels": 16, "kernel_size": 5, "stride": 1, "padding": 2}),
    #     (nn.LeakyReLU, {"negative_slope": 0.2}),
    #     (nn.BatchNorm2d, {}),

    #     (nn.Conv2d, {"out_channels": 8, "kernel_size": 5, "stride": 1, "padding": 2}),
    #     (nn.LeakyReLU, {"negative_slope": 0.2}),
    #     (nn.BatchNorm2d, {}),

    #     (nn.Conv2d, {"out_channels": 1, "kernel_size": 5, "stride": 1, "padding": 2}),
    #     (nn.Sigmoid, {})
    # ]
    # adversariat_architecture = [
    #     (nn.Conv2d, {"in_channels": 1, "out_channels": 1, "kernel_size": 5, "stride": 3, "padding": 0}),
    #     (nn.Flatten, {}),
    #     (nn.Linear, {"in_features": 100, "out_features": 128}),
    #     (nn.ReLU, {}),
    #     (nn.Linear, {"out_features": 1}),
    #     (nn.ReLU, {}),
    #     (nn.Linear, {"out_features": 1}),
    #     (nn.Sigmoid, {})
    # ]


    #########################################################################
    # torch.nn.Module 1
    #########################################################################
    generator_architecture = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128,64),
                      nn.ReLU(),
                      nn.Linear(64,10))


    #########################################################################
    # torch.nn.Module 1
    #########################################################################
    # class SimpleNetwork(nn.Module):
    #     def __init__(self):
    #         super().__init__()

    #         self.hidden_1 = nn.Linear(784,128)
    #         self.hidden_2 = nn.Linear(128,64)
    #         self.output = nn.Linear(64,10)

    #     def forward(self,x):
    #         x = F.relu(self.hidden_1(x))
    #         x = F.relu(self.hidden_2(x))
    #         y_pred = self.output(x)
    #         return y_pred
    # model = SimpleNetwork()

    vanilla_gan = VanillaGAN(
        generator_architecture=generator_architecture, adversariat_architecture=adversariat_architecture,
        z_dim=z_dim, in_dim=out_size, folder="TrainedModels/GAN", optim=torch.optim.RMSprop,
        generator_kwargs={"lr": lr_gen}, adversariat_kwargs={"lr": lr_adv}, enable_tensorboard=True
    )
    vanilla_gan.summary(save=True)
    vanilla_gan.save_as_json()
    vanilla_gan.fit(
        X_train=X_train,
        X_test=X_test,
        batch_size=batch_size,
        epochs=epochs,
        adv_steps=5,
        print_every=100
    )
    vanilla_gan.save()