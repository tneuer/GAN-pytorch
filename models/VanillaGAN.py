import torch

from torch.nn import BCELoss
from models.DualGAN import DualGAN


class VanillaGAN(DualGAN):
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
            folder="./VanillaGAN",
            enable_tensorboard=True):

        assert adversariat_architecture[-1][0] == torch.nn.Sigmoid, (
            "Last layer activation function of adversariat needs to be 'torch.nn.sigmoid'."
        )
        super(VanillaGAN, self).__init__(
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
        self.generator_loss_fn = BCELoss()
        self.adversariat_loss_fn = BCELoss()
