import os
import torch

import numpy as np

from utils.utils import get_input_dim
from models.unconditional.DualGAN import DualGAN
from utils.networks import Generator, Adversariat
from models.conditional.ConditionalGenerativeModel import ConditionalGenerativeModel


class ConditionalDualGAN(ConditionalGenerativeModel, DualGAN):
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(
            self,
            generator,
            adversariat,
            x_dim,
            z_dim,
            y_dim,
            adv_type,
            optim=None,
            optim_kwargs=None,
            generator_optim=None,
            generator_kwargs=None,
            adversariat_optim=None,
            adversariat_kwargs=None,
            fixed_noise_size=32,
            device=None,
            folder="./DualGAN",
            ngpu=0):

        adv_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
        gen_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim)
        DualGAN.__init__(
            self, generator=generator, adversariat=adversariat, x_dim=adv_in_dim, z_dim=gen_in_dim,
            adv_type=adv_type, optim=optim, optim_kwargs=optim_kwargs, generator_optim=generator_optim,
            generator_kwargs=generator_kwargs, adversariat_optim=adversariat_optim, adversariat_kwargs=adversariat_kwargs,
            fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=0
        )
        ConditionalGenerativeModel.__init__(
            self, x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, folder=None, ngpu=ngpu, fixed_noise_size=fixed_noise_size
        )


    #########################################################################
    # Actions during training
    #########################################################################
    def _train(self, X_batch, Z_batch, y_batch, who):
        if who == "Generator":
            gen_steps = self.steps["Generator"]
            self._train_generator(Z_batch=Z_batch, y_batch=y_batch, gen_steps=gen_steps)
        elif who == "Adversariat":
            adv_steps = self.steps["Adversariat"]
            self._train_adversariat(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch, adv_steps=adv_steps)
        else:
            raise NotImplementedError("Passed wrong network name. Called: {}.".format(who))

    def _train_generator(self, Z_batch, y_batch, gen_steps):
        for _ in range(gen_steps):
            self.calculate_losses(X_batch=None, Z_batch=Z_batch, y_batch=y_batch, who="Generator")
            self._zero_grad()
            self._backward(who="Generator")
            self._step(who="Generator")

    def _train_adversariat(self, X_batch, Z_batch, y_batch, adv_steps):
        for _ in range(adv_steps):
            self.calculate_losses(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch, who="Adversariat")
            self._zero_grad()
            self._backward(who="Adversariat")
            self._step(who="Adversariat")

    def calculate_losses(self, X_batch, Z_batch, y_batch, who=None):
        self._losses = {}
        if who == "Generator":
            self._calculate_generator_loss(Z_batch=Z_batch, y_batch=y_batch)
        elif who == "Adversariat":
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        else:
            self._calculate_generator_loss(Z_batch=Z_batch, y_batch=y_batch)
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)

    def _calculate_generator_loss(self, Z_batch, y_batch):
        fake_images = self.generate(y=y_batch, z=Z_batch)
        fake_predictions = self.predict(x=fake_images, y=y_batch)
        gen_loss = self.generator_loss_fn(
            fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
        )
        self._losses.update({"Generator": gen_loss})

    def _calculate_adversariat_loss(self, X_batch, Z_batch, y_batch):
        fake_images = self.generate(y=y_batch, z=Z_batch).detach()
        fake_predictions = self.predict(x=fake_images, y=y_batch)
        real_predictions = self.predict(x=X_batch, y=y_batch)

        adv_loss_fake = self.adversariat_loss_fn(
            fake_predictions, torch.zeros_like(fake_predictions, requires_grad=False)
        )
        adv_loss_real = self.adversariat_loss_fn(
            real_predictions, torch.ones_like(real_predictions, requires_grad=False)
        )
        adv_loss = 0.5*(adv_loss_fake + adv_loss_real)
        self._losses.update({
            "Adversariat": adv_loss,
            "Adversariat_fake": adv_loss_fake,
            "Adversariat_real": adv_loss_real,
        })