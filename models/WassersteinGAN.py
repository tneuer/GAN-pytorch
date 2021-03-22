import torch

import numpy as np

from models.DualGAN import DualGAN
from utils.utils import wasserstein_loss


class WassersteinGAN(DualGAN):
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(
            self,
            generator_architecture,
            adversariat_architecture,
            z_dim,
            in_dim,
            optim=None,
            optim_kwargs=None,
            generator_optim=None,
            generator_kwargs=None,
            adversariat_optim=None,
            adversariat_kwargs=None,
            device=None,
            folder="./WassersteinGAN",
            enable_tensorboard=True):

        assert adversariat_architecture[-1][0] == torch.nn.Linear, (
            "Last layer activation function of adversariat needs to be 'torch.nn.Linear'."
        )
        super(WassersteinGAN, self).__init__(
            generator_architecture=generator_architecture, adversariat_architecture=adversariat_architecture,
            z_dim=z_dim, in_dim=in_dim,
            optim=optim, optim_kwargs=optim_kwargs,
            generator_optim=generator_optim, generator_kwargs=generator_kwargs,
            adversariat_optim=adversariat_optim, adversariat_kwargs=adversariat_kwargs,
            device=device,
            folder=folder,
            enable_tensorboard=enable_tensorboard
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