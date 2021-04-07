import os
import sys
import json
import time
import torch

import numpy as np
import matplotlib.pyplot as plt
import vegans.utils.utils as utils

from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from vegans.utils.utils import get_input_dim
from torch.utils.tensorboard import SummaryWriter
from vegans.models.unconditional.GenerativeModel import GenerativeModel


class ConditionalGenerativeModel(GenerativeModel):
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(self, x_dim, z_dim, y_dim, optim, optim_kwargs, fixed_noise_size, device, folder, ngpu):
        """ The ConditionalGenerativeModel is the most basic building block of VeGAN for conditional models. All conditional GAN
        implementation should at least inherit from this class.

        Parameters
        ----------
        x_dim : list, tuple
            Number of the output dimension of the generator and input dimension of the discriminator / critic.
            In the case of images this will be [nr_channels, nr_height_pixels, nr_width_pixels].
        z_dim : int, list, tuple
            Number of the latent dimension for the generator input. Might have dimensions of an image.
        y_dim : int, list, tuple
            Number of dimensions for the target label. Might have dimensions of image for image to image translation, i.e.
            [nr_channels, nr_height_pixels, nr_width_pixels] or an integer
        optim : dict or torch.optim
            Optimizer used for each network. Could be either an optimizer from torch.optim or a dictionary with network
            name keys and torch.optim as value, i.e. {"Generator": torch.optim.Adam}.
        optim_kwargs : dict
            Optimizer keyword arguments used for each network. Must be a dictionary with network
            name keys and dictionary with keyword arguments as value, i.e. {"Generator": {"lr": 0.0001}}.
        fixed_noise_size : int
            Number of images shown when logging. The fixed noise is used to produce the images in the folder/images
            subdirectory, the tensorboard images tab and the samples in get_training_results().
        device : string
            Device used while training the model. Either "cpu" or "cuda".
        folder : string
            Creates a folder in the current working directory with this name. All relevant files like summary, images, models and
            tensorboard output are written there. Existing folders are never overwritten or deleted. If a folder with the same name
            already exists a time stamp is appended to make it unique.
        ngpu : int
            Number of gpus used during training if device == "cuda".
        """
        adv_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
        gen_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim)
        GenerativeModel.__init__(
            self, x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs,
            fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=ngpu
        )
        self.y_dim = tuple([y_dim]) if isinstance(y_dim, int) else y_dim
        self.hyperparameters["y_dim"] = self.y_dim

    def _set_up_training(self, X_train, y_train, X_test, y_test, epochs, batch_size, steps,
        print_every, save_model_every, save_images_every, save_losses_every, enable_tensorboard):
        train_dataloader, test_dataloader, writer_train, writer_test, save_periods = super()._set_up_training(
            X_train, y_train, X_test, y_test, epochs, batch_size, steps,
            print_every, save_model_every, save_images_every, save_losses_every, enable_tensorboard
        )
        self.fixed_labels = y_train[:self.fixed_noise_size]
        self.fixed_labels = torch.from_numpy(self.fixed_labels).to(self.device)
        return train_dataloader, test_dataloader, writer_train, writer_test, save_periods


    #########################################################################
    # Actions during training
    #########################################################################
    def fit(self, X_train, y_train, X_test=None, y_test=None, epochs=5, batch_size=32, steps=None,
            print_every="1e", save_model_every=None, save_images_every=None, save_losses_every="1e", enable_tensorboard=True):
        """ Method to call when the conditional generative adversarial network should be trained.

        Parameters
        ----------
        X_train : np.array
            Training data for the generative adversarial network. Usually images.
        y_train: np.array
            Training labels for the generative adversarial network. Might be images or one-hot encoded vector.
        X_test : np.array, optional
            Testing data for the generative adversarial network. Must have same shape as X_train.
        y_train: np.array
            Testing labels for the generative adversarial network. Might be images or one-hot encoded vector.
        epochs : int, optional
            Number of epochs (passes over the training data set) performed during training.
        batch_size : int, optional
            Batch size used when creating the data loader from X_train. Ignored if torch.utils.data.DataLoader is passed
            for X_train.
        steps : dict, optional
            Dictionary with names of the networks to indicate how often they should be trained, i.e. {"Generator": 5} indicates
            that the generator is trained 5 times while all other networks are trained once.
        print_every : int, string, optional
            Indicates after how many batches the losses for the train data should be printed to the console. Can also be a string
            of the form "0.25e" (4 times per epoch), "1e" (once per epoch) or "3e" (every third epoch).
        save_model_every : int, string, optional
            Indicates after how many batches the model should be saved. Can also be a string
            of the form "0.25e" (4 times per epoch), "1e" (once per epoch) or "3e" (every third epoch).
        save_images_every : int, string, optional
            Indicates after how many batches the images for the losses and fixed_noise should be saved. Can also be a string
            of the form "0.25e" (4 times per epoch), "1e" (once per epoch) or "3e" (every third epoch).
        save_losses_every : int, string, optional
            Indicates after how many batches the losses for the train and test data should be calculated. Can also be a string
            of the form "0.25e" (4 times per epoch), "1e" (once per epoch) or "3e" (every third epoch).
        enable_tensorboard : bool, optional
            Flag to indicate whether subdirectory folder/tensorboard should be created to log losses and images.
        """
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
            self._log(
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
                    self._log(X_batch=X, Z_batch=Z, y_batch=y, is_train=True, writer=writer_train, **log_kwargs)
                    if test_x_batch is not None:
                        self._log(
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


    #########################################################################
    # Logging during training
    #########################################################################
    def _log(self, X_batch, Z_batch, y_batch, batch, max_batches, epoch, max_epochs, print_every,
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
        """ Call after training to get fixed_noise samples and losses.

        Parameters
        ----------
        by_epoch : bool, optional
            If true one loss value per epoch is returned for every logged_loss. Otherwise frequency is given
            by `save_losses_every` argument of `fit`, i.e. `save_losses_every=10` saves losses every 10th batch,
            `save_losses_every="0.25e` saves losses 4 times per epoch.
        agg : None, optional
            Aggregation function used if by_epoch is true, otherwise ignored. Default is np.mean for all batches
            in one epoch.

        Returns
        -------
        losses_dict : dict
            Dictionary containing all loss types logged during training
        """
        samples = self.generate(y=self.fixed_labels, z=self.fixed_noise).detach().cpu().numpy()
        losses = self.get_losses(by_epoch=by_epoch, agg=agg)
        return samples, losses


    #########################################################################
    # Utility functions
    #########################################################################
    def generate(self, y, z=None):
        """ Generate output with generator.

        Parameters
        ----------
        y : np.array
            Labels for outputs to be produced.
        z : None, optional
            Latent input vector to produce an output from.

        Returns
        -------
        np.array
            Output produced by generator.
        """
        return self(y=y, z=z)

    def predict(self, x, y):
        """ Use the critic / discriminator to predict if input is real / fake.

        Parameters
        ----------
        x : np.array
            Images or samples to be predicted.
        y : np.array
            Labels for outputs to be predicted.

        Returns
        -------
        np.array
            Array with one output per x indicating the realness of an input.
        """
        inpt = utils.concatenate(x, y).float().to(self.device)
        return self.adversariat(inpt)

    def __call__(self, y, z=None):
        if len(y.shape) == 1:
            y = self._one_hot_encoder.transform(y)
        if z is None:
            z = self.sample(n=len(y))

        inpt = utils.concatenate(z.to(self.device), y.to(self.device)).float()
        return self.generator(inpt)
