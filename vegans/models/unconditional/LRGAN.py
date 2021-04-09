"""
LRGAN
-----
Implements the latent regressor GAN well described in the BicycleGAN paper[1].

It introduces an encoder network which maps the generator output back to the latent
input space. This should help to prevent mode collapse and improve image variety.

Losses:
    - Generator: Binary cross-entropy + L1-latent-loss (Mean Absolute Error)
    - Discriminator: Binary cross-entropy
    - Encoder: L1-latent-loss (Mean Absolute Error)
Default optimizer:
    - torch.optim.Adam

References
----------
.. [1] https://arxiv.org/pdf/1711.11586.pdf
"""

import torch

from torch.nn import BCELoss, L1Loss
from torch.nn import MSELoss as L2Loss

from vegans.utils.networks import Generator, Adversariat, Encoder
from vegans.models.unconditional.AbstractGenerativeModel import AbstractGenerativeModel

class LRGAN(AbstractGenerativeModel):
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(
            self,
            generator,
            adversariat,
            encoder,
            x_dim,
            z_dim,
            optim=None,
            optim_kwargs=None,
            lambda_L1=10,
            fixed_noise_size=32,
            device=None,
            folder="./AbstractGAN1v1",
            ngpu=0):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = Generator(generator, input_size=z_dim, device=device, ngpu=ngpu)
        self.adversariat = Adversariat(adversariat, input_size=x_dim, adv_type="Discriminator", device=device, ngpu=ngpu)
        self.encoder = Encoder(encoder, input_size=x_dim, device=device, ngpu=ngpu)
        self.neural_nets = {
            "Generator": self.generator, "Adversariat": self.adversariat, "Encoder": self.encoder
        }

        AbstractGenerativeModel.__init__(
            self, x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs,
            fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=ngpu
        )
        self.lambda_L1 = lambda_L1
        self.hyperparameters["lambda_L1"] = lambda_L1
        assert (self.generator.output_size == self.x_dim), (
            "Generator output shape must be equal to x_dim. {} vs. {}.".format(self.generator.output_size, self.x_dim)
        )
        assert (self.encoder.output_size == self.z_dim), (
            "Encoder output shape must be equal to z_dim. {} vs. {}.".format(self.encoder.output_size, self.z_dim)
        )

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        self.loss_functions = {"Generator": BCELoss(), "Adversariat": BCELoss(), "L1": L1Loss()}


    #########################################################################
    # Actions during training
    #########################################################################
    def encode(self, x):
        return self.encoder(x)

    def calculate_losses(self, X_batch, Z_batch, who=None):
        if who == "Generator":
            self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch)
        elif who == "Adversariat":
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch)
        elif who == "Encoder":
            self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch)
        else:
            self._calculate_generator_loss(Z_batch=Z_batch)
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch)
            self._losses["RealFakeRatio"] = self._losses["Adversariat_real"]/self._losses["Adversariat_fake"]

    def _calculate_generator_loss(self, Z_batch):
        fake_images = self.generate(z=Z_batch)
        fake_predictions = self.predict(x=fake_images)
        encoded_space = self.encode(x=fake_images)

        gen_loss_original = self.loss_functions["Generator"](
            fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
        )
        latent_space_regression = self.loss_functions["L1"](
            fake_images, encoded_space
        )
        gen_loss = gen_loss_original + self.lambda_L1*latent_space_regression
        self._losses.update({
            "Generator": gen_loss,
            "Generator_Original": gen_loss_original,
            "Generator_L1": latent_space_regression
        })

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

    def _calculate_encoder_loss(self, X_batch, Z_batch):
        fake_images = self.generate(z=Z_batch).detach()
        encoded_space = self.encoder(fake_images)
        latent_space_regression = self.loss_functions["L1"](
            fake_images, encoded_space
        )
        self._losses.update({
            "Encoder": latent_space_regression
        })