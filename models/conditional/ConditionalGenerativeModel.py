import os
import sys
import json
import torch

import numpy as np
import utils.utils as utils
import matplotlib.pyplot as plt
import time

from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn.preprocessing import OneHotEncoder
from torch.utils.tensorboard import SummaryWriter
from models.unconditional.GenerativeModel import GenerativeModel


class ConditionalGenerativeModel(GenerativeModel):
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(self, in_dim, z_dim, y_dim, folder, ngpu, fixed_noise_size):
        GenerativeModel.__init__(
            self, in_dim=in_dim, z_dim=z_dim, folder=folder, fixed_noise_size=fixed_noise_size, ngpu=ngpu
        )
        self.y_dim = y_dim
        self.hyperparameters["y_dim"] = y_dim
        self.fixed_labels = np.zeros(shape=(fixed_noise_size, y_dim))
        for i in range(self.fixed_labels.shape[0]):
            self.fixed_labels[i, i % y_dim] = 1
        self.fixed_labels = torch.from_numpy(self.fixed_labels).to(self.device)

    def _set_up_training(self, X_train, y_train, X_test, y_test, epochs, batch_size, gen_steps, adv_steps,
        log_every, save_model_every, save_images_every, enable_tensorboard):
        assert X_train.shape[2:] == self.adversariat.input_size[1:], (
            "Wrong input shape for adversariat. Given: {}. Needed: {}.".format(X_train.shape, self.adversariat.input_size)
        )
        assert len(set(y_train)) == self.y_dim, (
            "y_dim differs from number of unique elements in y_train. {} vs. {}.".format(self.y_dim, len(set(y_train)))
        )
        assert len(y_train.shape) == 1, (
            "y_train must have 1 dimension. Given: {}.".format(len(y_train.shape))
        )
        nr_test = 0 if X_test is None else len(X_test)
        self.hyperparameters.update({
            "epochs": epochs, "batch_size": batch_size, "gen_steps": gen_steps, "adv_steps": adv_steps,
            "log_every": log_every, "save_model_every": save_model_every, "save_images_every": save_images_every,
            "enable_tensorboard": enable_tensorboard, "nr_train": len(X_train), "nr_test": nr_test
        })

        writer_train = writer_test = None
        self._one_hot_encoder = OneHotEncoder(sparse=False)
        y_train = self._one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
        if enable_tensorboard:
            writer_train = SummaryWriter(self.folder+"tensorboard/train/")

        if not isinstance(X_train, DataLoader):
            train_data = utils.DataSet(X=X_train, y=y_train)
            train_dataloader = DataLoader(train_data, batch_size=batch_size)

        test_dataloader = None
        if (X_test is not None):
            y_test = self._one_hot_encoder.transform(y_test.reshape(-1, 1))
            test_data = utils.DataSet(X=X_test, y=y_test)
            test_dataloader = DataLoader(test_data, batch_size=batch_size)
            if enable_tensorboard:
                writer_test = SummaryWriter(self.folder+"tensorboard/test/")

        save_periods = self._set_up_saver(log_every, save_model_every, save_images_every, len(train_dataloader))
        return train_dataloader, test_dataloader, writer_train, writer_test, save_periods


    #########################################################################
    # Actions during training
    #########################################################################
    def fit(self, X_train, y_train, X_test, y_test, epochs, batch_size, gen_steps, adv_steps, log_every, enable_tensorboard):
        raise NotImplementedError("'fit' must be implemented by subclass.")

    def calculate_losses(self, X_batch, Z_batch, y_batch, who=None):
        raise NotImplementedError("'calculate_losses' must be implemented by subclass.")


    #########################################################################
    # After training
    #########################################################################
    def get_training_results(self, by_epoch=False, agg=None):
        samples = self.generate(y=self.fixed_labels, z=self.fixed_noise).detach().cpu().numpy()
        losses = self.get_losses(by_epoch=by_epoch, agg=agg)
        return samples, losses


    #########################################################################
    # Logging during training
    #########################################################################
    def log(self, X_batch, Z_batch, y_batch, batch, max_batches, epoch, max_epochs, log_every,
            is_train=True, log_images=False, writer=None):
        step = epoch*max_batches + batch
        if X_batch is not None:
            self.calculate_losses(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
            self._log_scalars(step=step, is_train=is_train, writer=writer)
        if log_images and self.images_produced:
            self._log_images(images=self.generate(y=self.fixed_labels, z=self.fixed_noise), step=step, writer=writer)

        if is_train:
            self._summarise_batch(batch, max_batches, epoch, max_epochs, log_every)

    def _log_images(self, images, step, writer):
        assert len(self.adversariat.input_size) > 1, (
            "Called _log_images in GenerativeModel for adversariat.input_size = {}.".format(self.adversariat.input_size)
        )
        if writer is not None:
            grid = make_grid(images)
            writer.add_image('images', grid, step)

        fig, axs = self._build_images(images)
        for i, ax in enumerate(np.ravel(axs)):
            lbl = torch.argmax(self.fixed_labels[i], axis=0).item()
            ax.set_title("Label: {}".format(lbl))
        plt.savefig(self.folder+"images/image_{}.png".format(step))
        print("Images logged.")


    #########################################################################
    # Utility functions
    #########################################################################
    def generate(self, y, z=None):
        return self(y=y, z=z)

    def predict(self, x, y):
        return self.adversariat(utils.concatenate(x, y).float())

    def __call__(self, y, z=None):
        if len(y.shape) == 1:
            y = self._one_hot_encoder.transform(y)
        if z is None:
            z = self.sample(n=len(y))

        inpt = utils.concatenate(z, y).float()
        return self.generator(inpt)
