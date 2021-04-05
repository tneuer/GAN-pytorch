import torch

import numpy as np
import utils.utils as utils

from torch import nn
from utils.layers import LayerReshape, LayerDebug, LayerPrintSize
from models.GAN import ConditionalVanillaGAN, ConditionalWassersteinGAN, ConditionalWassersteinGANGP

if __name__ == '__main__':

    datapath = "./data/mnist/"
    X_train, y_train, X_test, y_test = utils.load_mnist(datapath, normalize=True, pad=2, return_datasets=False)
    lr_gen = 0.0001
    lr_adv = 0.00005
    epochs = 10
    batch_size = 32

    X_train = X_train.reshape((-1, 1, 32, 32))
    X_test = X_test.reshape((-1, 1, 32, 32))
    out_size = X_train.shape[1:]
    label_dim = len(set(y_train))


    #########################################################################
    # Convolutional network
    #########################################################################
    # z_dim = [1, 8, 8]

    # class MyGenerator(nn.Module):
    #     def __init__(self, z_dim):
    #         super().__init__()
    #         self.hidden_part = nn.Sequential(
    #             nn.ConvTranspose2d(in_channels=z_dim[0]+label_dim, out_channels=64, kernel_size=4, stride=2, padding=1),
    #             nn.LeakyReLU(0.2),
    #             nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
    #             nn.LeakyReLU(0.2),
    #             nn.BatchNorm2d(32),
    #             nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
    #             nn.LeakyReLU(0.2),
    #             nn.BatchNorm2d(16),
    #             nn.Conv2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1),
    #         )
    #         self.output = nn.Sigmoid()

    #     def forward(self, x):
    #         x = self.hidden_part(x)
    #         y_pred = self.output(x)
    #         return y_pred

    # class MyAdversariat(nn.Module):
    #     def __init__(self, in_dim):
    #         super().__init__()
    #         self.hidden_part = nn.Sequential(
    #             nn.Conv2d(in_channels=in_dim[0]+label_dim, out_channels=32, kernel_size=4, stride=2, padding=1),
    #             nn.ReLU(),
    #             nn.Conv2d(in_channels=32, out_channels=8, kernel_size=4, stride=2, padding=1),
    #             nn.ReLU(),
    #             nn.Flatten(),
    #             nn.Linear(in_features=512, out_features=64),
    #             nn.ReLU(),
    #             nn.Linear(in_features=64, out_features=16),
    #             nn.ReLU(),
    #             nn.Linear(in_features=16, out_features=1)
    #         )
    #         self.output = nn.Linear(1, 1)

    #     def forward(self, x):
    #         x = self.hidden_part(x)
    #         y_pred = self.output(x)
    #         return y_pred

    #########################################################################
    # Flat network
    #########################################################################
    z_dim = 32

    class MyGenerator(nn.Module):
        def __init__(self, z_dim):
            super().__init__()
            self.hidden_part = nn.Sequential(
                nn.Linear(z_dim + label_dim, 128),
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
                nn.Conv2d(in_channels=in_dim[0]+label_dim, out_channels=1, kernel_size=4, stride=4, padding=0),
                nn.Flatten(),
                nn.Linear(64, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 1)
            )
            self.output = nn.Linear(1, 1)

        def forward(self, x):
            x = self.hidden_part(x)
            y_pred = self.output(x)
            return y_pred

    #########################################################################
    # Training
    #########################################################################

    generator = MyGenerator(z_dim=z_dim)
    adversariat = MyAdversariat(in_dim=out_size)
    vanilla_gan = ConditionalWassersteinGAN(
        generator=generator, adversariat=adversariat,
        in_dim=out_size, z_dim=z_dim, y_dim=label_dim, folder="TrainedModels/ConditionalGAN", optim=torch.optim.RMSprop,
        generator_kwargs={"lr": lr_gen}, adversariat_kwargs={"lr": lr_adv}, fixed_noise_size=16
    )
    vanilla_gan.summary(save=True)
    vanilla_gan.fit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        batch_size=batch_size,
        epochs=epochs,
        adv_steps=5,
        log_every="0.1e",
        save_model_every="0.5e",
        save_images_every="0.1e",
        enable_tensorboard=True,
    )
    samples, losses = vanilla_gan.get_training_results(by_epoch=True)
    vanilla_gan.save()