import torch

import numpy as np

from utils.utils import wasserstein_loss
from models.conditional.ConditionalDualGAN import ConditionalDualGAN


class ConditionalWassersteinGAN(ConditionalDualGAN):
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
            folder="./WassersteinGAN"):

        ConditionalDualGAN.__init__(
            self,
            generator=generator, adversariat=adversariat,
            in_dim=in_dim, z_dim=z_dim, y_dim=y_dim, adv_type="Critic",
            optim=optim, optim_kwargs=optim_kwargs,
            generator_optim=generator_optim, generator_kwargs=generator_kwargs,
            adversariat_optim=adversariat_optim, adversariat_kwargs=adversariat_kwargs,
            fixed_noise_size=fixed_noise_size,
            device=device,
            folder=folder,
        )

    def _define_loss(self):
        self.generator_loss_fn = wasserstein_loss
        self.adversariat_loss_fn = wasserstein_loss


    #########################################################################
    # Actions during training
    #########################################################################
    def _step(self, who=None):
        if who is not None:
            self.optimizers[who].step()
            if who == "Adversariat":
                for p in self.adversariat.parameters():
                    p.data.clamp_(-0.01, 0.01)
        else:
            [optimizer.step() for _, optimizer in self.optimizers.items()]