import torch

from torch.nn import BCELoss
from vegans.models.conditional.ConditionalDualGAN import ConditionalDualGAN


class ConditionalVanillaGAN(ConditionalDualGAN):
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
            optim=None,
            optim_kwargs=None,
            fixed_noise_size=32,
            device=None,
            folder="./ConditionalVanillaGAN",
            ngpu=None):

        ConditionalDualGAN.__init__(
            self,
            generator=generator, adversariat=adversariat,
            x_dim=x_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Discriminator",
            optim=optim, optim_kwargs=optim_kwargs,
            fixed_noise_size=fixed_noise_size,
            device=device, folder=folder, ngpu=ngpu
        )

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        self.generator_loss_fn = BCELoss()
        self.adversariat_loss_fn = BCELoss()
