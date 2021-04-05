import torch

import numpy as np
import utils.utils as utils

from torch import nn
from models.GAN import VanillaGAN, WassersteinGAN, WassersteinGANGP
from utils.layers import LayerReshape, LayerDebug

if __name__ == '__main__':

    datapath = "./data/mnist/"
    X_train, y_train, X_test, y_test = utils.load_mnist(datapath, normalize=True, pad=2, return_datasets=False)
    lr_gen = 0.0001
    lr_adv = 0.00005
    epochs = 2
    batch_size = 64

    X_train = X_train.reshape((-1, 1, 32, 32))[:1000]
    X_test = X_test.reshape((-1, 1, 32, 32))

    #########################################################################
    # Convolutional network
    #########################################################################
    z_dim = [1, 8, 8]
    out_size = X_train.shape[1:]

    generator = nn.Sequential(
        nn.ConvTranspose2d(in_channels=z_dim[0], out_channels=64, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(16),
        nn.Conv2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid()
    )

    adversariat = nn.Sequential(
        nn.Conv2d(in_channels=out_size[0], out_channels=32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=8, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=512, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=16),
        nn.ReLU(),
        nn.Linear(in_features=16, out_features=1),
        nn.Sigmoid()
    )

    #########################################################################
    # torch.nn.Module 1
    #########################################################################
    # z_dim = 2
    # out_size = X_train.shape[1:]
#
    # generator = nn.Sequential(
    #     nn.Linear(z_dim, 128),
    #     nn.LeakyReLU(0.2),
    #     nn.Linear(128, 256),
    #     nn.LeakyReLU(0.2),
    #     nn.BatchNorm1d(256),
    #     nn.Linear(256, 512),
    #     nn.LeakyReLU(0.2),
    #     nn.BatchNorm1d(512),
    #     nn.Linear(512, 1024),
    #     nn.LeakyReLU(0.2),
    #     nn.BatchNorm1d(1024),
    #     nn.Linear(1024, int(np.prod(out_size))),
    #     LayerReshape(out_size),
    #     nn.Sigmoid()
    # )
    # adversariat = nn.Sequential(
    #     nn.Flatten(),
    #     nn.Linear(int(np.prod(out_size)), 512),
    #     nn.LeakyReLU(0.2),
    #     nn.Linear(512, 256),
    #     nn.LeakyReLU(0.2),
    #     nn.Linear(256, 1),
    #     nn.Sigmoid()
    # )

    #########################################################################
    # Training
    #########################################################################

    vanilla_gan = VanillaGAN(
        generator=generator, adversariat=adversariat,
        z_dim=z_dim, in_dim=out_size, folder="TrainedModels/GAN", optim=torch.optim.RMSprop,
        generator_kwargs={"lr": lr_gen}, adversariat_kwargs={"lr": lr_adv}
    )
    vanilla_gan.summary(save=True)
    vanilla_gan.fit(
        X_train=X_train,
        X_test=None,
        batch_size=batch_size,
        epochs=epochs,
        adv_steps=5,
        log_every=100,
        save_model_every="0.5e",
        save_images_every="0.25e",
        enable_tensorboard=True
    )
    samples, losses = vanilla_gan.get_training_results(by_epoch=True)
    print(losses)
    vanilla_gan.save()