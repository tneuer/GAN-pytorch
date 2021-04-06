import torch

import numpy as np

from vegans.models.unconditional.DualGAN import DualGAN
from vegans.utils.utils import wasserstein_loss


class WassersteinGAN(DualGAN):
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
            clip_val=0.01,
            device=None,
            folder="./WassersteinGAN",
            ngpu=None):

        DualGAN.__init__(
            self,
            generator=generator, adversariat=adversariat,
            z_dim=z_dim, x_dim=x_dim, adv_type="Critic",
            optim=optim, optim_kwargs=optim_kwargs,
            fixed_noise_size=fixed_noise_size,
            device=device,
            folder=folder,
            ngpu=ngpu
        )
        self._clip_val = clip_val

    def _default_optimizer(self):
        return torch.optim.RMSprop

    def _default_optimizer(self):
        return torch.optim.RMSprop

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
                    p.data.clamp_(-self._clip_val, self._clip_val)
        else:
            [optimizer.step() for _, optimizer in self.optimizers.items()]