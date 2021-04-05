import torch

from torch.nn import BCELoss
from models.conditional.ConditionalDualGAN import ConditionalDualGAN


class ConditionalVanillaGAN(ConditionalDualGAN):
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(
            self,
            generator,
            adversariat,
            in_dim,
            z_dim,
            y_dim,
            optim=None,
            optim_kwargs=None,
            generator_optim=None,
            generator_kwargs=None,
            adversariat_optim=None,
            adversariat_kwargs=None,
            fixed_noise_size=32,
            device=None,
            folder="./ConditionalVanillaGAN",
            ngpu=None):

        ConditionalDualGAN.__init__(
            self,
            generator=generator, adversariat=adversariat,
            in_dim=in_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Discriminator",
            optim=optim, optim_kwargs=optim_kwargs,
            generator_optim=generator_optim, generator_kwargs=generator_kwargs,
            adversariat_optim=adversariat_optim, adversariat_kwargs=adversariat_kwargs,
            fixed_noise_size=fixed_noise_size,
            device=device, folder=folder, ngpu=ngpu
        )

    def _define_loss(self):
        self.generator_loss_fn = BCELoss()
        self.adversariat_loss_fn = BCELoss()
