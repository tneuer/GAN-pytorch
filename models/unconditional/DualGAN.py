import os
import torch

import numpy as np

from utils.networks import Generator, Adversariat
from models.unconditional.GenerativeModel import GenerativeModel


class DualGAN(GenerativeModel):
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(
            self,
            generator,
            adversariat,
            in_dim,
            z_dim,
            adv_type,
            optim=None,
            optim_kwargs=None,
            generator_optim=None,
            generator_kwargs=None,
            adversariat_optim=None,
            adversariat_kwargs=None,
            fixed_noise_size=32,
            device=None,
            folder="./DualGAN",
            ngpu=0):
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.generator = Generator(generator, input_size=z_dim, ngpu=ngpu)
        self.adversariat = Adversariat(adversariat, input_size=in_dim, adv_type=adv_type, ngpu=ngpu)

        self.neural_nets = (self.generator, self.adversariat)
        self._define_optimizers(
            optim=optim, optim_kwargs=optim_kwargs,
            generator_optim=generator_optim, generator_kwargs=generator_kwargs,
            adversariat_optim=adversariat_optim, adversariat_kwargs=adversariat_kwargs
        )
        GenerativeModel.__init__(self, in_dim=in_dim, z_dim=z_dim, folder=folder, ngpu=ngpu, fixed_noise_size=fixed_noise_size)

    def _define_optimizers(
        self, optim, optim_kwargs, generator_optim, generator_kwargs,
        adversariat_optim, adversariat_kwargs):
        assert optim is not None or generator_optim is not None, (
            "Either 'optim' or 'generator_optim' must be not None."
        )
        assert optim is not None or adversariat_optim is not None, (
            "Either 'optim' or 'adversariat_optim' must be not None."
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
    def fit(self, X_train, X_test=None, epochs=5, batch_size=32, gen_steps=1, adv_steps=1,
        log_every=100, save_model_every=None, save_images_every=None, enable_tensorboard=True):
        train_dataloader, test_dataloader, writer_train, writer_test, save_periods = self._set_up_training(
            X_train, X_test=X_test, epochs=epochs, batch_size=batch_size, gen_steps=gen_steps, adv_steps=adv_steps,
            log_every=log_every, save_model_every=save_model_every, save_images_every=save_images_every,
            enable_tensorboard=enable_tensorboard
        )
        max_batches = len(train_dataloader)
        test_x_batch = next(iter(test_dataloader))[0].to(self.device) if X_test is not None else None
        log_every, save_model_every, save_images_every = save_periods
        if test_x_batch is not None:
            self.log(
                X_batch=test_x_batch, Z_batch=self.sample(len(test_x_batch)), batch=0, max_batches=max_batches, epoch=0, max_epochs=epochs,
                log_every=log_every, is_train=False, log_images=False
            )

        for epoch in range(epochs):
            print("---"*20)
            print("EPOCH:", epoch+1)
            print("---"*20)
            for batch, X in enumerate(train_dataloader):
                batch += 1
                step = epoch*max_batches + batch
                X = X.to(self.device)
                Z = self.sample(n=len(X))
                self._train_adversariat(X_batch=X, Z_batch=Z, adv_steps=adv_steps)

                Z = self.sample(n=len(X))
                self._train_generator(Z_batch=Z, gen_steps=gen_steps)

                if log_every is not None and step % log_every == 0:
                    log_kwargs = {
                        "batch": batch, "max_batches": max_batches, "epoch": epoch, "max_epochs": epochs,
                        "log_every": log_every, "log_images": False
                    }
                    self.log(X_batch=X, Z_batch=Z, is_train=True, writer=writer_train, **log_kwargs)
                    if test_x_batch is not None:
                        self.log(
                            X_batch=test_x_batch, Z_batch=self.sample(len(test_x_batch)), batch=0, max_batches=max_batches,
                            epoch=0, max_epochs=epochs, log_every=log_every, is_train=False, log_images=False
                        )

                if save_model_every is not None and step % save_model_every == 0:
                    self.save(name="models/model_{}.torch".format(step))

                if save_images_every is not None and step % save_images_every == 0:
                    self._log_images(images=self.generator(self.fixed_noise), step=step, writer=writer_train)

        self._clean_up(writers=[writer_train, writer_test])

    def _train_adversariat(self, X_batch, Z_batch, adv_steps):
        for _ in range(adv_steps):
            self.calculate_losses(X_batch=X_batch, Z_batch=Z_batch, who="Adversariat")
            self._zero_grad()
            self._backward(who="Adversariat")
            self._step(who="Adversariat")
        self.logged_losses["Adversariat"].append(self._losses["Adversariat"].item())
        self.logged_losses["Adversariat_fake"].append(self._losses["Adversariat_fake"].item())
        self.logged_losses["Adversariat_real"].append(self._losses["Adversariat_real"].item())

    def _train_generator(self, Z_batch, gen_steps):
        for _ in range(gen_steps):
            self.calculate_losses(X_batch=None, Z_batch=Z_batch, who="Generator")
            self._zero_grad()
            self._backward(who="Generator")
            self._step(who="Generator")
        self.logged_losses["Generator"].append(self._losses["Generator"].item())

    def calculate_losses(self, X_batch, Z_batch, who=None):
        self._losses = {}
        if who == "Generator":
            self._calculate_generator_loss(Z_batch=Z_batch)
        elif who == "Adversariat":
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch)
        else:
            self._calculate_generator_loss(Z_batch=Z_batch)
            self._calculate_adversariat_loss(X_batch=X_batch, Z_batch=Z_batch)

    def _calculate_generator_loss(self, Z_batch):
        fake_images = self.generate(z=Z_batch)
        fake_predictions = self.predict(x=fake_images)
        gen_loss = self.generator_loss_fn(
            fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
        )
        self._losses.update({"Generator": gen_loss})

    def _calculate_adversariat_loss(self, X_batch, Z_batch):
        fake_images = self.generate(z=Z_batch).detach()
        fake_predictions = self.predict(x=fake_images)
        real_predictions = self.predict(x=X_batch.float())

        adv_loss_fake = self.adversariat_loss_fn(
            fake_predictions, torch.zeros_like(fake_predictions, requires_grad=False)
        )
        adv_loss_real = self.adversariat_loss_fn(
            real_predictions, torch.ones_like(real_predictions, requires_grad=False)
        )
        adv_loss = 0.5*(adv_loss_fake + adv_loss_real)
        self._losses.update({
            "Adversariat": adv_loss,
            "Adversariat_fake": adv_loss_fake,
            "Adversariat_real": adv_loss_real,
        })