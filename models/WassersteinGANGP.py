import torch

import numpy as np

from models.DualGAN import DualGAN
from utils.utils import wasserstein_loss


class WassersteinGANGP(DualGAN):
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
            lmbda_grad=10,
            device=None,
            folder="./WassersteinGANGP",
            enable_tensorboard=True):

        assert adversariat_architecture[-1][0] == torch.nn.Linear, (
            "Last layer activation function of adversariat needs to be 'torch.nn.Linear'."
        )
        super(WassersteinGANGP, self).__init__(
            generator_architecture=generator_architecture, adversariat_architecture=adversariat_architecture,
            z_dim=z_dim, in_dim=in_dim,
            optim=optim, optim_kwargs=optim_kwargs,
            generator_optim=generator_optim, generator_kwargs=generator_kwargs,
            adversariat_optim=adversariat_optim, adversariat_kwargs=adversariat_kwargs,
            device=device,
            folder=folder,
            enable_tensorboard=enable_tensorboard
        )
        self.lmbda_grad = lmbda_grad

    def _define_loss(self):
        self.generator_loss_fn = wasserstein_loss
        self.adversariat_loss_fn = wasserstein_loss
        self.gradient_penalty_fn = gradient_penalty

    def gradient_penalty(self, real_images, fake_images):
        alpha = torch.Tensor(np.random.random((real_images.size(0), 1, 1, 1))).to(self.device)
        interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True).float()
        d_interpolates = self.adversariat(interpolates).to(self.device)
        dummy = torch.ones_like(d_interpolates, requires_grad=False).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=dummy,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


    #########################################################################
    # Actions during training
    #########################################################################
    def _calculate_adversariat_loss(self, X_batch):
        fake_images = self.generate(n=len(X_batch)).detach()
        fake_predictions = self.adversariat(fake_images)
        real_predictions = self.adversariat(X_batch.float())

        adv_loss_fake = self.adversariat_loss_fn(
            fake_predictions, torch.zeros_like(fake_predictions, requires_grad=False)
        )
        adv_loss_real = self.adversariat_loss_fn(
            real_predictions, torch.ones_like(real_predictions, requires_grad=False)
        )
        adv_loss_grad = self.gradient_penalty_fn(X_batch, fake_images)
        adv_loss = 0.5*(adv_loss_fake + adv_loss_real) + self.lmbda_grad*adv_loss_grad
        self.losses.update({
            "Adversariat": adv_loss,
            "Adversariat_fake": adv_loss_fake,
            "Adversariat_real": adv_loss_real,
            "Adversariat_grad": adv_loss_grad
        })