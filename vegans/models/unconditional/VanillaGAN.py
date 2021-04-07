import torch

from torch.nn import BCELoss
from vegans.models.unconditional.DualGAN import DualGAN


class VanillaGAN(DualGAN):
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(
            self,
            generator,
            adversariat,
            x_dim,
            z_dim,
            optim=None,
            optim_kwargs=None,
            fixed_noise_size=32,
            device=None,
            folder="./VanillaGAN",
            ngpu=None):

        DualGAN.__init__(
            self,
            generator=generator, adversariat=adversariat,
            z_dim=z_dim, x_dim=x_dim, adv_type="Discriminator",
            optim=optim, optim_kwargs=optim_kwargs,
            fixed_noise_size=fixed_noise_size,
            device=device, folder=folder, ngpu=ngpu
        )

    def _default_optimizer(self):
        return torch.optim.Adam

    def _define_loss(self):
        self.generator_loss_fn = BCELoss()
        self.adversariat_loss_fn = BCELoss()