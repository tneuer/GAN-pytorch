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
from vegans.utils.utils import plot_losses


class GenerativeModel():
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(self, x_dim, z_dim, optim, optim_kwargs, fixed_noise_size, device, folder, ngpu):
        """ The GenerativeModel is the most basic building block of VeGAN. All GAN implementation should
        at least inherit from this class. If a conditional version is implemented look at `ConditionalGenerativeModel`.

        Parameters
        ----------
        x_dim : list, tuple
            Number of the output dimension of the generator and input dimension of the discriminator / critic.
            In the case of images this will be [nr_channels, nr_height_pixels, nr_width_pixels].
        z_dim : int, list, tuple
            Number of the latent dimension for the generator input. Might have dimensions of an image.
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
        self.x_dim = tuple([x_dim]) if isinstance(x_dim, int) else tuple(x_dim)
        self.z_dim = tuple([z_dim]) if isinstance(z_dim, int) else tuple(z_dim)
        self.ngpu = ngpu if ngpu is not None else 0
        self.fixed_noise_size = fixed_noise_size
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device not in ["cuda", "cpu"]:
            raise ValueError("device must be cuda or cpu.")

        if hasattr(self, "folder"):
            pass
        elif folder is None:
            self.folder = ""
        else:
            folder = folder if not folder.endswith("/") else folder[-1]
            if os.path.exists(folder):
                now = datetime.now()
                now = now.strftime("%Y%m%d_%H%M%S")
                folder += now
            self.folder = folder + "/"
            os.makedirs(self.folder)

        self._define_loss()
        self._define_optimizers(
            optim=optim, optim_kwargs=optim_kwargs,
        )
        self.to(self.device)

        self.images_produced = True if len(self.adversariat.input_size) > 1 else False
        self.fixed_noise = self.sample(n=fixed_noise_size)
        self.hyperparameters = {
            "x_dim": x_dim, "z_dim": z_dim, "ngpu": ngpu, "folder": folder, "optimizers": self.optimizers,
            "device": self.device, "generator_loss": self.generator_loss_fn, "adversariat_loss": self.adversariat_loss_fn
        }
        self._check_attributes()

    def _check_attributes(self):
        assert hasattr(self, "generator"), "Model must have attribute 'generator'."
        assert hasattr(self, "adversariat"), "Model must have attribute 'adversariat'."
        assert hasattr(self, "neural_nets"), "Model must have attribute 'neural_nets'."
        assert hasattr(self, "device"), "Model must have attribute 'device'."
        assert hasattr(self, "optimizers"), "Model must have attribute 'optimizers'."
        assert hasattr(self, "generator_loss_fn"), "DualGAN needs attribute 'generator_loss_fn'."
        assert hasattr(self, "adversariat_loss_fn"), "DualGAN needs attribute 'adversariat_loss_fn'."
        assert isinstance(self.ngpu, int) and self.ngpu >= 0, "ngpu must be positive integer. Given: {}.".format(ngpu)
        for name, _ in self.neural_nets.items():
            assert name in self.optimizers, "{} does not have a corresponding optimizer but is needed.".format(name)
        assert len(self.z_dim) == 1 or len(self.z_dim) == 3, (
            "z_dim must either have length 1 (for vector input) or 3 (for image input). Given: {}.".format(z_dim)
        )
        assert len(self.x_dim) == 1 or len(self.x_dim) == 3, (
            "x_dim must either have length 1 (for vector input) or 3 (for image input). Given: {}.".format(x_dim)
        )

    def _define_optimizers(self, optim, optim_kwargs):
        self._opt_kwargs = {}
        for name, _ in self.neural_nets.items():
            self._opt_kwargs[name] = {}
        if isinstance(optim_kwargs, dict):
            self._check_dict_keys(optim_kwargs, where="_define_optimizers")
            for name, opt_kwargs in optim_kwargs.items():
                self._opt_kwargs[name] = opt_kwargs

        self._opt = {}
        if optim is None:
            for name, _ in self.neural_nets.items():
                self._opt[name] = self._default_optimizer()
        elif isinstance(optim, dict):
            self._check_dict_keys(optim, where="_define_optimizers")
            for name, opt in optim.items():
                self._opt[name] = opt
            for name, _ in self.neural_nets.items():
                if name not in self._opt:
                    self._opt[name] = self._default_optimizer()
        else:
            for name, _ in self.neural_nets.items():
                self._opt[name] = optim

        self.optimizers = {}
        for name, network in self.neural_nets.items():
            self.optimizers[name] = self._opt[name](params=network.parameters(), **self._opt_kwargs[name])

    def _check_dict_keys(self, param_dict, where):
        for name, _ in param_dict.items():
            if name not in self.neural_nets:
                available_nets = [name for name, _ in self.neural_nets.items()]
                raise ValueError("Error in {}: {} not in neural_nets. Must be one of: {}.".format(
                    where, name, available_nets)
                )


    def _define_loss(self):
        raise NotImplementedError("'_define_loss' must be implemented for objects of type 'GenerativeModel'.")

    def _set_up_training(self, X_train, y_train, X_test, y_test, epochs, batch_size, steps,
        print_every, save_model_every, save_images_every, save_losses_every, enable_tensorboard):

        self._assert_shapes(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        nr_test = 0 if X_test is None else len(X_test)

        writer_train = writer_test = None
        if enable_tensorboard:
            writer_train = SummaryWriter(self.folder+"tensorboard/train/")

        if not isinstance(X_train, DataLoader):
            train_data = utils.DataSet(X=X_train, y=y_train)
            train_dataloader = DataLoader(train_data, batch_size=batch_size)

        test_dataloader = None
        if X_test is not None:
            test_data = utils.DataSet(X=X_test, y=y_test)
            test_dataloader = DataLoader(test_data, batch_size=batch_size)
            if enable_tensorboard:
                writer_test = SummaryWriter(self.folder+"tensorboard/test/")

        self._create_steps(steps=steps)
        save_periods = self._set_up_saver(
            print_every=print_every, save_model_every=save_model_every, save_images_every=save_images_every,
            save_losses_every=save_losses_every, nr_batches=len(train_dataloader)
        )
        self.hyperparameters.update({
            "epochs": epochs, "batch_size": batch_size, "steps": self.steps,
            "print_every": print_every, "save_model_every": save_model_every, "save_images_every": save_images_every,
            "enable_tensorboard": enable_tensorboard, "nr_train": len(X_train), "nr_test": nr_test
        })
        return train_dataloader, test_dataloader, writer_train, writer_test, save_periods

    def _assert_shapes(self, X_train, y_train, X_test, y_test):
        assert len(X_train.shape) == 2 or len(X_train.shape) == 4, (
            "X_train must be either have 2 or 4 shape dimensions. Given: {}.".format(X_train.shape) +
            "Try to use X_train.reshape(-1, 1) or X_train.reshape(-1, 1, height, width)."
        )
        assert X_train.shape[2:] == self.adversariat.input_size[1:], (
            "Wrong input shape for adversariat. Given: {}. Needed: {}.".format(X_train.shape, self.adversariat.input_size)
        )
        assert X_train.shape[0] == y_train.shape[0], (
            "Same number if X_train and y_train needed.Given: {} and {}.".format(X_train.shape[0], y_train.shape[0])
        )
        if X_test is not None:
            assert X_train.shape[1:] == X_test.shape[1:], (
                "X_train and X_test must have same dimensions. Given: {} and {}.".format(X_train.shape[1:], X_test.shape[1:])
            )
            assert X_test.shape[0] == y_test.shape[0], (
                "Same number if X_test and y_test needed.Given: {} and {}.".format(X_test.shape[0], y_test.shape[0])
            )

        if y_train is not None:
            assert len(y_train.shape) == 2 or len(y_train.shape) == 4, (
                "y_train must be either have 2 or 4 shape dimensions. Given: {}.".format(y_train.shape) +
                "Try to use y_train.reshape(-1, 1) or y_train.reshape(-1, 1, height, width)."
            )
            assert y_train.shape[2:] == self.y_dim[1:], (
                "Wrong input shape for y_train. Given: {}. Needed: {}.".format(y_train.shape, self.y_dim)
            )
            if X_test is not None:
                assert y_train.shape[1:] == y_test.shape[1:], (
                    "y_train and y_test must have same dimensions. Given: {} and {}.".format(y_train.shape[1:], y_test.shape[1:])
                )


    def _create_steps(self, steps):
        if steps is None:
            self.steps = {}
            for name, _ in self.neural_nets.items():
                self.steps[name] = 1
        else:
            assert isinstance(steps, dict), "steps parameter must be of type dict. Given: {}.".format(type(steps))
            self._check_dict_keys(steps, where="_create_steps")
            self.steps = steps
            for name, _ in self.neural_nets.items():
                if name not in self.steps:
                    self.steps[name] = 1

    def _set_up_saver(self, print_every, save_model_every, save_images_every, save_losses_every, nr_batches):
        if isinstance(print_every, str):
            save_epochs = float(print_every.split("e")[0])
            print_every = int(save_epochs*nr_batches)
        if save_model_every is not None:
            os.mkdir(self.folder+"models/")
            if isinstance(save_model_every, str):
                save_epochs = float(save_model_every.split("e")[0])
                save_model_every = int(save_epochs*nr_batches)
        if save_images_every is not None:
            os.mkdir(self.folder+"images/")
            if isinstance(save_images_every, str):
                save_epochs = float(save_images_every.split("e")[0])
                save_images_every = int(save_epochs*nr_batches)
        if isinstance(save_losses_every, str):
            save_epochs = float(save_losses_every.split("e")[0])
            save_losses_every = int(save_epochs*nr_batches)
        self.total_training_time = 0
        self.current_timer = time.perf_counter()
        self.batch_training_times = []

        return print_every, save_model_every, save_images_every, save_losses_every


    #########################################################################
    # Actions during training
    #########################################################################
    def fit(self, X_train, X_test=None, epochs=5, batch_size=32, steps=None,
        print_every="1e", save_model_every=None, save_images_every=None, save_losses_every="1e", enable_tensorboard=True):
        """ Method to call when the generative adversarial network should be trained.

        Parameters
        ----------
        X_train : np.array or torch.utils.data.DataLoader
            Training data for the generative adversarial network. Usually images.
        X_test : np.array, optional
            Testing data for the generative adversarial network. Must have same shape as X_train.
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
            X_train, y_train=None, X_test=X_test, y_test=None, epochs=epochs, batch_size=batch_size, steps=steps,
            print_every=print_every, save_model_every=save_model_every, save_images_every=save_images_every,
            save_losses_every=save_losses_every, enable_tensorboard=enable_tensorboard
        )
        max_batches = len(train_dataloader)
        test_x_batch = iter(test_dataloader).next().to(self.device) if X_test is not None else None
        print_every, save_model_every, save_images_every, save_losses_every = save_periods
        if test_x_batch is not None:
            self._log(
                X_batch=test_x_batch, Z_batch=self.sample(len(test_x_batch)), batch=0, max_batches=max_batches, epoch=0, max_epochs=epochs,
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
                Z = self.sample(n=len(X))
                for name, _ in self.neural_nets.items():
                    self._train(X_batch=X, Z_batch=Z, who=name)

                if print_every is not None and step % print_every == 0:
                    log_kwargs = {
                        "batch": batch, "max_batches": max_batches, "epoch": epoch, "max_epochs": epochs,
                        "print_every": print_every, "log_images": False
                    }
                    self._log(X_batch=X, Z_batch=Z, is_train=True, writer=writer_train, **log_kwargs)
                    if test_x_batch is not None:
                        self._log(
                            X_batch=test_x_batch, Z_batch=self.sample(len(test_x_batch)), batch=0, max_batches=max_batches,
                            epoch=0, max_epochs=epochs, print_every=print_every, is_train=False, log_images=False
                        )

                if save_model_every is not None and step % save_model_every == 0:
                    self.save(name="models/model_{}.torch".format(step))

                if save_images_every is not None and step % save_images_every == 0:
                    self._log_images(images=self.generator(self.fixed_noise), step=step, writer=writer_train)
                    self._save_losses_plot()

                if save_losses_every is not None and step % save_losses_every == 0:
                    self._log_losses(X_batch=X, Z_batch=Z, is_train=True)
                    if test_x_batch is not None:
                        self._log_losses(X_batch=test_x_batch, Z_batch=self.sample(len(test_x_batch)), is_train=False)

        self._clean_up(writers=[writer_train, writer_test])

    def _train(self, X_batch, Z_batch, who=None):
        raise NotImplementedError("'_train' must be implemented by subclass.")

    def calculate_losses(self, X_batch, Z_batch, who=None):
        raise NotImplementedError("'calculate_losses' must be implemented by subclass.")

    def _zero_grad(self, who=None):
        if who is not None:
            self.optimizers[who].zero_grad()
        else:
            [optimizer.zero_grad() for _, optimizer in self.optimizers.items()]

    def _backward(self, who=None):
        assert len(self._losses) != 0, "'self._losses' empty when performing '_backward'."
        if who is not None:
            self._losses[who].backward(retain_graph=True)
        else:
            [loss.backward(retain_graph=True) for _, loss in self._losses.items()]

    def _step(self, who=None):
        if who is not None:
            self.optimizers[who].step()
        else:
            [optimizer.step() for _, optimizer in self.optimizers.items()]


    #########################################################################
    # Logging during training
    #########################################################################
    def _log(self, X_batch, Z_batch, batch, max_batches, epoch, max_epochs, print_every,
            is_train=True, log_images=False, writer=None):
        step = epoch*max_batches + batch
        if X_batch is not None:
            self.calculate_losses(X_batch=X_batch, Z_batch=Z_batch)
            self._log_scalars(step=step, writer=writer)
        if log_images and self.images_produced:
            self._log_images(images=self.generator(self.fixed_noise), step=step, writer=writer)

        if is_train:
            self._summarise_batch(batch, max_batches, epoch, max_epochs, print_every)

    def _summarise_batch(self, batch, max_batches, epoch, max_epochs, print_every):
        step = epoch*max_batches + batch
        max_steps = max_epochs*max_batches
        remaining_batches = max_epochs*max_batches - step
        print("Step: {} / {} (Epoch: {} / {}, Batch: {} / {})".format(
            step, max_steps, epoch+1, max_epochs, batch, max_batches)
        )
        print("---"*20)
        for name, loss in self._losses.items():
            print("{}: {}".format(name, loss.item()))

        self.batch_training_times.append(time.perf_counter() - self.current_timer)
        self.total_training_time = np.sum(self.batch_training_times)
        time_per_batch = np.mean(self.batch_training_times) / print_every

        print("\n")
        print("Time left: ~{} minutes (Batches remaining: {}).".format(
            np.round(remaining_batches*time_per_batch/60, 3), remaining_batches
            )
        )
        print("\n")
        self.current_timer = time.perf_counter()

    def _log_scalars(self, step, writer):
        if writer is not None:
            for name, loss in self._losses.items():
                writer.add_scalar("Loss/{}".format(name), loss.item(), step)
            writer.add_scalar("Time/Total", self.total_training_time / 60, step)
            writer.add_scalar("Time/Batch", np.mean(self.batch_training_times) / 60, step)

    def _log_images(self, images, step, writer):
        assert len(self.adversariat.input_size) > 1, (
            "Called _log_images in GenerativeModel for adversariat.input_size = {}.".format(self.adversariat.input_size)
        )
        if writer is not None:
            grid = make_grid(images)
            writer.add_image('images', grid, step)

        fig, axs = self._build_images(images)
        plt.savefig(self.folder+"images/image_{}.png".format(step))
        plt.close()
        print("Images logged.")

    @staticmethod
    def _build_images(images):
        images = images.cpu().detach().numpy()
        if len(images.shape) == 4:
            images = images.reshape((-1, *images.shape[-2:]))
        nrows = int(np.sqrt(len(images)))
        ncols = len(images) // nrows
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))

        for i, (ax, image) in enumerate(zip(np.ravel(axs), images)):
            ax.imshow(image)
            ax.axis("off")
        return fig, axs

    def _log_losses(self, X_batch, Z_batch, is_train):
        mode = "Train" if is_train else "Test"
        self.calculate_losses(X_batch=X_batch, Z_batch=Z_batch)
        self._append_losses(mode=mode)

    def _append_losses(self, mode):
        if not hasattr(self, "logged_losses"):
            self._create_logged_losses()
        for name, loss in self._losses.items():
            self.logged_losses[mode][name].append(self._losses[name].item())

    def _create_logged_losses(self):
        with_test = self.hyperparameters["nr_test"] is not None
        self.logged_losses = {"Train": {}}
        if with_test:
            self.logged_losses["Test"] = {}

        for name, _ in self._losses.items():
            self.logged_losses["Train"][name] = []
            if with_test:
                self.logged_losses["Test"][name] = []

    def _save_losses_plot(self):
        if hasattr(self, "logged_losses"):
            fig, axs = plot_losses(self.logged_losses, show=False, share=False)
            plt.savefig(self.folder+"losses.png")
            plt.close()

    #########################################################################
    # After training
    #########################################################################
    def _clean_up(self, writers=None):
        [writer.close() for writer in writers if writer is not None]

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
        samples : np.array
            Images produced with the final model for the fixed_noise.
        losses_dict : dict
            Dictionary containing all loss types logged during training
        """
        samples = self.generate(self.fixed_noise).detach().cpu().numpy()
        losses = self.get_losses(by_epoch=by_epoch, agg=agg)
        return samples, losses

    def get_losses(self, by_epoch=False, agg=None):
        """ Get losses logged during training

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
        if agg is None:
            agg = np.mean
        assert callable(agg), "agg: Aggregation function must be callable."
        losses_dict = self.logged_losses.copy()
        if by_epoch:
            epochs = self.get_hyperparameters()["epochs"]
            for mode, loss_dict in losses_dict.items():
                for key, losses in loss_dict.items():
                    batches_per_epoch = len(losses) // epochs
                    loss_dict[key] = [losses[epoch*batches_per_epoch:(epoch+1)*batches_per_epoch] for epoch in range(epochs)]
                    loss_dict[key] = [agg(loss_epoch) for loss_epoch in loss_dict[key]]

        return losses_dict


    #########################################################################
    # Saving and loading
    #########################################################################
    def save(self, name=None):
        """ Saves model in the model folder as torch / pickle object.

        Parameters
        ----------
        name : str, optional
            name of the saved file. folder specified in the constructor used
            in absolute path.
        """
        if name is None:
            name = "model.torch"
        torch.save(self, self.folder+name)
        print("Model saved to {}.".format(self.folder+name))

    @staticmethod
    def load(path):
        """ Load an already trained model.

        Parameters
        ----------
        path : TYPE
            path to the saved file.

        Returns
        -------
        GenerativeModel
            Trained model
        """
        return torch.load(path)


    #########################################################################
    # Utility functions
    #########################################################################
    def sample(self, n):
        """ Sample from the latent generator distribution.

        Parameters
        ----------
        n : int
            Number of samples drawn from the latent generator distribution.

        Returns
        -------
        torch.tensor
            Random numbers with shape of [n, *z_dim]
        """
        return torch.randn(n, *self.z_dim, requires_grad=True, device=self.device)

    def generate(self, z=None, n=None):
        """ Generate output with generator.

        Parameters
        ----------
        z : None, optional
            Latent input vector to produce an output from.
        n : None, optional
            Number of outputs to be generated.

        Returns
        -------
        np.array
            Output produced by generator.
        """
        return self(z=z, n=n)

    def predict(self, x):
        """ Use the critic / discriminator to predict if input is real / fake.

        Parameters
        ----------
        x : np.array
            Images or samples to be predicted.

        Returns
        -------
        np.array
            Array with one output per x indicating the realness of an input.
        """
        return self.adversariat(x)

    def get_hyperparameters(self):
        """ Returns a dictionary containing all relevant hyperparameters.

        Returns
        -------
        dict
            Dictionary containing all relevant hyperparameters.
        """
        return self.hyperparameters

    def summary(self, save=False):
        """ Print summary of the model in Keras style way.

        Parameters
        ----------
        save : bool, optional
            If true summary is saved in model folder, printed to console otherwise.
        """
        if save:
            sys_stdout_temp = sys.stdout
            sys.stdout = open(self.folder+'summary.txt', 'w')
        for name, neural_net in self.neural_nets.items():
            print(neural_net)
            neural_net.summary()
            print("\n\n")
        if save:
            sys.stdout = sys_stdout_temp
            sys_stdout_temp

    def eval(self):
        """ Set all networks to evaluation mode.
        """
        [network.eval() for name, network in self.neural_nets.items()]

    def train(self):
        """ Set all networks to training mode.
        """
        [network.train() for name, network in self.neural_nets.items()]

    def to(self, device):
        """ Map all networks to device.
        """
        [network.to(device) for name, network in self.neural_nets.items()]

    def __call__(self, z=None, n=None):
        if z is not None and n is not None:
            raise ValueError("Only one of 'z' and 'n' is needed.")
        elif z is None and n is None:
            raise ValueError("Either 'z' or 'n' must be not None.")
        elif n is not None:
            z = self.sample(n=n)
        return self.generator(z)

    def __str__(self):
        self.summary()