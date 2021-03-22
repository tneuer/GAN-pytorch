import torch

import numpy as np

from models.GenerativeModel import GenerativeModel
from utils.networks import Generator, Adversariat


class DualGAN(GenerativeModel):
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
            folder="./DualGAN",
            enable_tensorboard=True,
            ngpu=0):
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.generator = Generator(generator_architecture, input_size=z_dim)
        self.adversariat = Adversariat(adversariat_architecture, input_size=in_dim)

        self.neural_nets = (self.generator, self.adversariat)
        self._define_optimizers(
            optim=optim, optim_kwargs=optim_kwargs,
            generator_optim=generator_optim, generator_kwargs=generator_kwargs,
            adversariat_optim=adversariat_optim, adversariat_kwargs=adversariat_kwargs
        )
        super(DualGAN, self).__init__(folder=folder, enable_tensorboard=enable_tensorboard, ngpu=ngpu)

    def _define_optimizers(
        self, optim, optim_kwargs,
        generator_optim, generator_kwargs,
        adversariat_optim, adversariat_kwargs):
        assert optim is not None or generator_optim is not None, (
            "Either 'optim' or 'generator_optim' must be not None."
        )
        assert optim is not None or adversariat_optim is not None, (
            "Either 'optim' or 'generator_optim' must be not None."
        )
        generator_optim = optim if generator_optim is None else generator_optim
        generator_kwargs = optim_kwargs if generator_kwargs is None else generator_kwargs
        adversariat_optim = optim if adversariat_optim is None else adversariat_optim
        adversariat_kwargs = optim_kwargs if adversariat_kwargs is None else adversariat_kwargs
        optimizer_generator = generator_optim(params=self.generator.parameters(), **generator_kwargs)
        optimizer_adversariat = adversariat_optim(params=self.adversariat.parameters(), **adversariat_kwargs)
        self.optimizers = {"Generator": optimizer_generator, "Adversariat": optimizer_adversariat}


    #########################################################################
    # Actions during training
    #########################################################################
    def fit(self, X_train, X_test=None, epochs=5, batch_size=32, gen_steps=1, adv_steps=1, print_every=100):
        train_dataloader, X_test = self.set_up_training(X_train, X_test=X_test)
        max_batches = len(train_dataloader)

        self.print_every = print_every
        self.batch_size = batch_size
        self.gen_steps = gen_steps
        self.adv_steps = adv_steps
        self.log(
            X_batch=X_test, batch=0, max_batches=max_batches, epoch=0, max_epochs=epochs,
            print_every=print_every, is_train=False, log_images=False
        )
        for epoch in range(epochs):
            print("---"*20)
            print("EPOCH:", epoch+1)
            print("---"*20)
            for batch, X in enumerate(train_dataloader):
                batch += 1
                step = epoch*max_batches + batch
                X = X.to(self.device)
                for _ in range(adv_steps):
                    self.calculate_losses(X_batch=X, who="Adversariat")
                    self._zero_grad()
                    self._backward(who="Adversariat")
                    self._step(who="Adversariat")

                for _ in range(gen_steps):
                    self.calculate_losses(X_batch=X, who="Generator")
                    self._zero_grad()
                    self._backward(who="Generator")
                    self._step(who="Generator")

                if step % print_every == 0:
                    self.log(
                        X_batch=X, batch=batch, max_batches=max_batches, epoch=epoch, max_epochs=epochs,
                        print_every=print_every, is_train=True, log_images=False
                    )
                    self.log(
                        X_batch=X_test, batch=batch, max_batches=max_batches, epoch=epoch, max_epochs=epochs,
                        print_every=print_every, is_train=False, log_images=False
                    )

            self.log(
                X_batch=X_test, batch=batch, max_batches=max_batches, epoch=epoch, max_epochs=epochs,
                print_every=print_every, is_train=False, log_images=True
            )

        self._clean_up()

    def calculate_losses(self, X_batch, who=None):
        self.losses = {}
        if who == "Generator":
            self._calculate_generator_loss(X_batch=X_batch)
        elif who == "Adversariat":
            self._calculate_adversariat_loss(X_batch=X_batch)
        else:
            self._calculate_generator_loss(X_batch=X_batch)
            self._calculate_adversariat_loss(X_batch=X_batch)

    def _calculate_generator_loss(self, X_batch):
        fake_images = self.generate(n=len(X_batch))
        fake_predictions = self.adversariat(fake_images)
        gen_loss = self.generator_loss_fn(
            fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
        )
        self.losses.update({"Generator": gen_loss})

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
        adv_loss = 0.5*(adv_loss_fake + adv_loss_real)
        self.losses.update({
            "Adversariat": adv_loss,
            "Adversariat_fake": adv_loss_fake,
            "Adversariat_real": adv_loss_real,
        })