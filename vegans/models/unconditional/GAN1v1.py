import os
import torch

import numpy as np

from vegans.utils.networks import Generator, Adversariat
from vegans.models.unconditional.GenerativeModel import GenerativeModel


class GAN1v1(GenerativeModel):
    """ Special half abstract class for GAN with structure of one generator and
    one discriminator / critic. Examples are the original `VanillaGAN`, `WassersteinGAN`
    and `WassersteinGANGP`.
    """

    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(
            self,
            generator,
            adversariat,
            x_dim,
            z_dim,
            adv_type,
            optim=None,
            optim_kwargs=None,
            fixed_noise_size=32,
            device=None,
            folder="./GAN1v1",
            ngpu=0):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = Generator(generator, input_size=z_dim, device=device, ngpu=ngpu)
        self.adversariat = Adversariat(adversariat, input_size=x_dim, adv_type=adv_type, device=device, ngpu=ngpu)
        self.neural_nets = {"Generator": self.generator, "Adversariat": self.adversariat}

        GenerativeModel.__init__(
            self, x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs,
            fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=ngpu
        )
        assert hasattr(self, "generator"), "Model must have attribute 'generator'."
        assert hasattr(self, "adversariat"), "Model must have attribute 'adversariat'."


    #########################################################################
    # Actions during training
    #########################################################################
    def calculate_losses(self, X_batch, Z_batch, who=None):
        if who == "Generator":
            self._calculate_generator_loss(Z_batch=Z_batch)
        elif who == "Adversariat":
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch)
        else:
            self._calculate_generator_loss(Z_batch=Z_batch)
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch)
            self._losses["RealFakeRatio"] = self._losses["Adversariat_real"]/self._losses["Adversariat_fake"]

    def _calculate_generator_loss(self, Z_batch):
        fake_images = self.generate(z=Z_batch)
        fake_predictions = self.predict(x=fake_images)
        gen_loss = self.loss_functions["Generator"](
            fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
        )
        self._losses.update({"Generator": gen_loss})

    def _calculate_adversariat_loss(self, X_batch, Z_batch):
        fake_images = self.generate(z=Z_batch).detach()
        fake_predictions = self.predict(x=fake_images)
        real_predictions = self.predict(x=X_batch.float())

        adv_loss_fake = self.loss_functions["Adversariat"](
            fake_predictions, torch.zeros_like(fake_predictions, requires_grad=False)
        )
        adv_loss_real = self.loss_functions["Adversariat"](
            real_predictions, torch.ones_like(real_predictions, requires_grad=False)
        )
        adv_loss = 0.5*(adv_loss_fake + adv_loss_real)
        self._losses.update({
            "Adversariat": adv_loss,
            "Adversariat_fake": adv_loss_fake,
            "Adversariat_real": adv_loss_real,
        })