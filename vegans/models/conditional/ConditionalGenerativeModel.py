import os
import sys
import json
import torch

import numpy as np
import vegans.utils.utils as utils
import matplotlib.pyplot as plt
import time

from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from vegans.models.unconditional.GenerativeModel import GenerativeModel


class ConditionalGenerativeModel(GenerativeModel):
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(self, x_dim, z_dim, y_dim, folder, ngpu, fixed_noise_size, device):
        GenerativeModel.__init__(
            self, x_dim=x_dim, z_dim=z_dim, folder=folder, fixed_noise_size=fixed_noise_size, ngpu=ngpu, device=device
        )
        self.y_dim = [y_dim] if isinstance(y_dim, int) else y_dim
        self.hyperparameters["y_dim"] = y_dim

    def _set_up_training(self, X_train, y_train, X_test, y_test, epochs, batch_size, steps,
        print_every, save_model_every, save_images_every, save_losses_every, enable_tensorboard):
        train_dataloader, test_dataloader, writer_train, writer_test, save_periods = super()._set_up_training(
            X_train, y_train, X_test, y_test, epochs, batch_size, steps,
            print_every, save_model_every, save_images_every, save_losses_every, enable_tensorboard
        )

        if len(self.y_dim) == 1:
            assert self.y_dim[0] == y_train.shape[1], (
                "y_dim differs from number of unique elements in y_train. {} vs. {}.".format(self.y_dim, y_train.shape)
            )
        else:
            assert self.y_dim[0] == y_train.shape[1] and self.y_dim[1] == y_train.shape[2] and self.y_dim[2] == y_train.shape[3], (
                "y_dim differs from number of unique elements in y_train. {} vs. {}.".format(self.y_dim, y_train.shape)
            )

        self.fixed_labels = y_train[:self.fixed_noise_size]
        self.fixed_labels = torch.from_numpy(self.fixed_labels).to(self.device)
        return train_dataloader, test_dataloader, writer_train, writer_test, save_periods


    #########################################################################
    # Actions during training
    #########################################################################
    def fit(self, X_train, y_train, X_test=None, y_test=None, epochs=5, batch_size=32, steps=None,
            print_every="1e", save_model_every=None, save_images_every=None, save_losses_every="1e", enable_tensorboard=True):
        train_dataloader, test_dataloader, writer_train, writer_test, save_periods = self._set_up_training(
            X_train, y_train, X_test=X_test, y_test=y_test, epochs=epochs, batch_size=batch_size, steps=steps,
            print_every=print_every, save_model_every=save_model_every, save_images_every=save_images_every,
            save_losses_every=save_losses_every, enable_tensorboard=enable_tensorboard
        )
        nr_batches = len(train_dataloader)
        test_x_batch = next(iter(test_dataloader))[0].to(self.device) if X_test is not None else None
        test_y_batch = next(iter(test_dataloader))[1].to(self.device) if X_test is not None else None
        print_every, save_model_every, save_images_every, save_losses_every = save_periods
        if test_x_batch is not None:
            self.log(
                X_batch=test_x_batch, Z_batch=self.sample(len(test_x_batch)), y_batch=test_y_batch, batch=0, max_batches=nr_batches,
                epoch=0, max_epochs=epochs, print_every=print_every, is_train=False, log_images=False
            )

        for epoch in range(epochs):
            print("---"*20)
            print("EPOCH:", epoch+1)
            print("---"*20)
            for batch, (X, y) in enumerate(train_dataloader):
                batch += 1
                step = epoch*nr_batches + batch
                X = X.to(self.device)
                y = y.to(self.device)
                Z = self.sample(n=len(X))
                for name, _ in self.neural_nets.items():
                    self._train(X_batch=X, Z_batch=Z, y_batch=y, who=name)

                if print_every is not None and step % print_every == 0:
                    log_kwargs = {
                        "batch": batch, "max_batches": nr_batches, "epoch": epoch, "max_epochs": epochs,
                        "print_every": print_every, "log_images": False
                    }
                    self.log(X_batch=X, Z_batch=Z, y_batch=y, is_train=True, writer=writer_train, **log_kwargs)
                    if test_x_batch is not None:
                        self.log(
                            X_batch=test_x_batch, Z_batch=self.sample(len(test_x_batch)), y_batch=test_y_batch, batch=0,
                            max_batches=nr_batches, epoch=0, max_epochs=epochs, print_every=print_every, is_train=False, log_images=False
                        )

                if save_model_every is not None and step % save_model_every == 0:
                    self.save(name="models/model_{}.torch".format(step))

                if save_images_every is not None and step % save_images_every == 0:
                    self._log_images(images=self.generate(y=self.fixed_labels, z=self.fixed_noise), step=step, writer=writer_train)
                    self._save_losses_plot()

                if save_losses_every is not None and step % save_losses_every == 0:
                    self._log_losses(X_batch=X, Z_batch=Z, y_batch=y, is_train=True)
                    if test_x_batch is not None:
                        self._log_losses(X_batch=test_x_batch, Z_batch=self.sample(len(test_x_batch)), y_batch=test_y_batch, is_train=False)

        self._clean_up(writers=[writer_train, writer_test])

    def calculate_losses(self, X_batch, Z_batch, y_batch, who=None):
        raise NotImplementedError("'calculate_losses' must be implemented by subclass.")


    #########################################################################
    # Logging during training
    #########################################################################
    def log(self, X_batch, Z_batch, y_batch, batch, max_batches, epoch, max_epochs, print_every,
            is_train=True, log_images=False, writer=None):
        step = epoch*max_batches + batch
        if X_batch is not None:
            self.calculate_losses(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
            self._log_scalars(step=step, writer=writer)
        if log_images and self.images_produced:
            self._log_images(images=self.generate(y=self.fixed_labels, z=self.fixed_noise), step=step, writer=writer)

        if is_train:
            self._summarise_batch(batch, max_batches, epoch, max_epochs, print_every)

    def _log_images(self, images, step, writer):
        assert len(self.adversariat.input_size) > 1, (
            "Called _log_images in GenerativeModel for adversariat.input_size = {}.".format(self.adversariat.input_size)
        )
        if writer is not None:
            grid = make_grid(images)
            writer.add_image('images', grid, step)

        fig, axs = self._build_images(images)
        for i, ax in enumerate(np.ravel(axs)):
            try:
                lbl = torch.argmax(self.fixed_labels[i], axis=0).item()
                ax.set_title("Label: {}".format(lbl))
            except ValueError:
                pass
        plt.savefig(self.folder+"images/image_{}.png".format(step))
        plt.close()
        print("Images logged.")

    def _log_losses(self, X_batch, Z_batch, y_batch, is_train):
        mode = "Train" if is_train else "Test"
        self.calculate_losses(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        self._append_losses(mode=mode)


    #########################################################################
    # After training
    #########################################################################
    def get_training_results(self, by_epoch=False, agg=None):
        samples = self.generate(y=self.fixed_labels, z=self.fixed_noise).detach().cpu().numpy()
        losses = self.get_losses(by_epoch=by_epoch, agg=agg)
        return samples, losses


    #########################################################################
    # Utility functions
    #########################################################################
    def generate(self, y, z=None):
        return self(y=y, z=z)

    def predict(self, x, y):
        inpt = utils.concatenate(x, y).float().to(self.device)
        return self.adversariat(inpt)

    def __call__(self, y, z=None):
        if len(y.shape) == 1:
            y = self._one_hot_encoder.transform(y)
        if z is None:
            z = self.sample(n=len(y))

        inpt = utils.concatenate(z.to(self.device), y.to(self.device)).float()
        return self.generator(inpt)
