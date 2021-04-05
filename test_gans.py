import torch

import numpy as np
import utils.utils as utils

from torch import nn
from models.GAN import VanillaGAN, WassersteinGAN, WassersteinGANGP, ConditionalVanillaGAN
from utils.layers import LayerReshape, LayerDebug

if __name__ == '__main__':

    datapath = "./Data/mnist/"
    X_train, y_train, X_test, y_test = utils.load_mnist(datapath, normalize=True, pad=2, return_datasets=False)
    lr_gen = 0.0001
    lr_adv = 0.00005
    epochs = 2
    batch_size = 64

    X_train = X_train.reshape((-1, 1, 32, 32))[:1000]
    X_test = X_test.reshape((-1, 1, 32, 32))
    z_dim = 2
    out_size = X_train.shape[1:]

    #########################################################################
    # Convolutional network
    #########################################################################
    # z_dim = [1, 8, 8]
    # out_size = X_train.shape[1:]
    # generator = [
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
    # adversariat = [
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
    generator = nn.Sequential(
        nn.Linear(z_dim, 128),
        nn.LeakyReLU(0.2),
        nn.Linear(128, 256),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(256),
        nn.Linear(256, 512),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(512),
        nn.Linear(512, 1024),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, int(np.prod(out_size))),
        LayerReshape(out_size),
        nn.Sigmoid()
    )
    adversariat = nn.Sequential(
        nn.Flatten(),
        nn.Linear(int(np.prod(out_size)), 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

    #########################################################################
    # torch.nn.Module 2
    #########################################################################
    class MyGenerator(nn.Module):
        def __init__(self, z_dim):
            super().__init__()
            self.hidden_part = nn.Sequential(
                nn.Linear(z_dim, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 256),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(256),
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(512),
                nn.Linear(512, 1024),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, int(np.prod(out_size))),
                LayerReshape(out_size)
            )
            self.output = nn.Sigmoid()

        def forward(self, x):
            x = self.hidden_part(x)
            y_pred = self.output(x)
            return y_pred

    class MyAdversariat(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.hidden_part = nn.Sequential(
                nn.Flatten(),
                nn.Linear(int(np.prod(out_size)), 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 1)
            )
            self.output = nn.Sigmoid()

        def forward(self, x):
            x = self.hidden_part(x)
            y_pred = self.output(x)
            return y_pred
    generator = MyGenerator(z_dim=z_dim)
    adversariat = MyAdversariat(in_dim=out_size)

    vanilla_gan = ConditionalVanillaGAN(
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