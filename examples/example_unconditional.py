import torch

import numpy as np
import vegans.utils.utils as utils

from torch import nn
from vegans.GAN import VanillaGAN, WassersteinGAN, WassersteinGANGP
from vegans.utils.layers import LayerReshape, LayerDebug, LayerPrintSize

if __name__ == '__main__':

    datapath = "./data/mnist/"
    X_train, y_train, X_test, y_test = utils.load_mnist(datapath, normalize=True, pad=2, return_datasets=False)
    lr_gen = 0.0001
    lr_adv = 0.00005
    epochs = 2
    batch_size = 64

    X_train = X_train.reshape((-1, 1, 32, 32))
    X_test = X_test.reshape((-1, 1, 32, 32))
    im_dim = X_train.shape[1:]


    #########################################################################
    # Architecture
    #########################################################################
    z_dim = 128

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
                nn.Linear(1024, int(np.prod(im_dim))),
                LayerReshape(im_dim)
            )
            self.output = nn.Sigmoid()

        def forward(self, x):
            x = self.hidden_part(x)
            y_pred = self.output(x)
            return y_pred

    class MyAdversariat(nn.Module):
        def __init__(self, x_dim):
            super().__init__()
            self.hidden_part = nn.Sequential(
                nn.Flatten(),
                nn.Linear(int(np.prod(im_dim)), 512),
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
    adversariat = MyAdversariat(x_dim=im_dim)
    gan_model = WassersteinGAN(
        generator=generator, adversariat=adversariat,
        z_dim=z_dim, x_dim=im_dim, folder="TrainedModels/GAN", optim={"Generator": torch.optim.Adam},
        optim_kwargs={"Generator": {"lr": lr_gen}, "Adversariat": {"lr": lr_adv}}
    )
    gan_model.summary(save=True)
    gan_model.fit(
        X_train=X_train,
        X_test=X_test,
        batch_size=batch_size,
        epochs=epochs,
        steps=None,
        log_every=200,
        save_model_every="3e",
        save_images_every="0.25e",
        save_losses_every=10,
        enable_tensorboard=True
    )
    samples, losses = gan_model.get_training_results(by_epoch=True)
    print(losses)
    gan_model.save()