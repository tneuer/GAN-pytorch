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
from torch.utils.tensorboard import SummaryWriter
from utils.utils import plot_losses


class GenerativeModel():
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(self, x_dim, z_dim, folder, ngpu, fixed_noise_size):
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.ngpu = ngpu
        self.fixed_noise_size = fixed_noise_size

        if folder is not None:
            folder = folder if not folder.endswith("/") else folder[-1]
            if os.path.exists(folder):
                now = datetime.now()
                now = now.strftime("%Y%m%d_%H%M%S")
                folder += now
            self.folder = folder + "/"
            os.makedirs(self.folder)
        if not hasattr(self, "folder"):
            self.folder = ""

        self._define_loss()
        self.to(self.device)

        self.images_produced = True if len(self.adversariat.input_size) > 1 else False
        self.fixed_noise = self.sample(n=fixed_noise_size)
        self.hyperparameters = {
            "x_dim": x_dim, "z_dim": z_dim, "ngpu": ngpu, "folder": folder, "optimizers": self.optimizers,
            "device": self.device, "generator_loss": self.generator_loss_fn, "adversariat_loss": self.adversariat_loss_fn
        }

        assert hasattr(self, "generator"), "Model must have attribute 'generator'."
        assert hasattr(self, "adversariat"), "Model must have attribute 'adversariat'."
        assert hasattr(self, "neural_nets"), "Model must have attribute 'neural_nets'."
        assert hasattr(self, "device"), "Model must have attribute 'device'."
        assert hasattr(self, "optimizers"), "Model must have attribute 'optimizers'."
        assert hasattr(self, "generator_loss_fn"), "DualGAN needs attribute 'generator_loss_fn'."
        assert hasattr(self, "adversariat_loss_fn"), "DualGAN needs attribute 'adversariat_loss_fn'."
        for name, _ in self.neural_nets.items():
            assert name in self.optimizers, "{} does not have a corresponding optimizer but is needed.".format(name)

    def _define_loss(self):
        raise NotImplementedError("'_define_loss' must be implemented for objects of type 'GenerativeModel'.")

    def _set_up_training(self, X_train, y_train, X_test, y_test, epochs, batch_size, steps,
        log_every, save_model_every, save_images_every, save_losses_every, enable_tensorboard):
        assert X_train.shape[2:] == self.adversariat.input_size[1:], (
            "Wrong input shape for adversariat. Given: {}. Needed: {}.".format(X_train.shape, self.adversariat.input_size)
        )
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
            log_every=log_every, save_model_every=save_model_every, save_images_every=save_images_every,
            save_losses_every=save_losses_every, nr_batches=len(train_dataloader)
        )
        self.hyperparameters.update({
            "epochs": epochs, "batch_size": batch_size, "steps": self.steps,
            "log_every": log_every, "save_model_every": save_model_every, "save_images_every": save_images_every,
            "enable_tensorboard": enable_tensorboard, "nr_train": len(X_train), "nr_test": nr_test
        })
        return train_dataloader, test_dataloader, writer_train, writer_test, save_periods

    def _create_steps(self, steps):
        if steps is None:
            self.steps = {}
            for name, _ in self.neural_nets.items():
                self.steps[name] = 1
        else:
            assert isinstance(steps, dict), "steps parameter must be of type dict. Given: {}.".format(type(steps))
            self.steps = steps
            for name, _ in self.neural_nets.items():
                if name not in self.steps:
                    self.steps[name] = 1

    def _set_up_saver(self, log_every, save_model_every, save_images_every, save_losses_every, nr_batches):
        if isinstance(log_every, str):
            save_epochs = float(log_every.split("e")[0])
            log_every = int(save_epochs*nr_batches)
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

        return log_every, save_model_every, save_images_every, save_losses_every


    #########################################################################
    # Actions during training
    #########################################################################
    def fit(self, X_train, X_test=None, epochs=5, batch_size=32, steps=None,
        log_every=100, save_model_every=None, save_images_every=None, save_losses_every=None, enable_tensorboard=True):
        train_dataloader, test_dataloader, writer_train, writer_test, save_periods = self._set_up_training(
            X_train, y_train=None, X_test=X_test, y_test=None, epochs=epochs, batch_size=batch_size, steps=steps,
            log_every=log_every, save_model_every=save_model_every, save_images_every=save_images_every,
            save_losses_every=save_losses_every, enable_tensorboard=enable_tensorboard
        )
        max_batches = len(train_dataloader)
        test_x_batch = iter(test_dataloader).next().to(self.device) if X_test is not None else None
        log_every, save_model_every, save_images_every, save_losses_every = save_periods
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
                for name, _ in self.neural_nets.items():
                    self._train(X_batch=X, Z_batch=Z, who=name)

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
                    self._save_losses_plot()

                if save_losses_every is not None and step % save_losses_every == 0:
                    self._log_losses(X_batch=X, Z_batch=Z, is_train=True)
                    self._log_losses(X_batch=test_x_batch, Z_batch=self.sample(len(test_x_batch)), is_train=False)

        self._clean_up(writers=[writer_train, writer_test])

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
    def log(self, X_batch, Z_batch, batch, max_batches, epoch, max_epochs, log_every,
            is_train=True, log_images=False, writer=None):
        step = epoch*max_batches + batch
        if X_batch is not None:
            self.calculate_losses(X_batch=X_batch, Z_batch=Z_batch)
            self._log_scalars(step=step, writer=writer)
        if log_images and self.images_produced:
            self._log_images(images=self.generator(self.fixed_noise), step=step, writer=writer)

        if is_train:
            self._summarise_batch(batch, max_batches, epoch, max_epochs, log_every)

    def _summarise_batch(self, batch, max_batches, epoch, max_epochs, log_every):
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
        time_per_batch = np.mean(self.batch_training_times) / log_every

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
            writer.add_scalar(
                "Loss/LossRatio", (self._losses["Adversariat_real"]/self._losses["Adversariat_fake"]).item(), step
            )
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
        samples = self.generate(self.fixed_noise).detach().cpu().numpy()
        losses = self.get_losses(by_epoch=by_epoch, agg=agg)
        return samples, losses

    def get_losses(self, by_epoch=False, agg=None):
        if agg is None:
            agg = np.mean
        assert callable(agg), "agg: Aggregation function must be callable."
        losses_dict = self.logged_losses.copy()
        if by_epoch:
            epochs = self.get_hyperparameters()["epochs"]
            for mode, loss_dict in losses_dict.items():
                for key, losses in loss_dict.items():
                    assert (len(losses) % epochs) == 0, (
                        "losses for {} (lenght={}) not divisible by epochs ({}).".format(key, len(losses), epochs)
                    )
                    batches_per_epoch = len(losses) // epochs
                    loss_dict[key] = [losses[epoch*batches_per_epoch:(epoch+1)*batches_per_epoch] for epoch in range(epochs)]
                    loss_dict[key] = [agg(loss_epoch) for loss_epoch in loss_dict[key]]

        return losses_dict


    #########################################################################
    # Saving and loading
    #########################################################################
    def save(self, name=None):
        if name is None:
            name = "model.torch"
        torch.save(self, self.folder+name)
        print("Model saved to {}.".format(self.folder+name))

    @staticmethod
    def load(path):
        return torch.load(path)


    #########################################################################
    # Utility functions
    #########################################################################
    def sample(self, n):
        if isinstance(self.z_dim, int):
            return torch.randn(n, self.z_dim, requires_grad=True, device=self.device)
        return torch.randn(n, *self.z_dim, requires_grad=True, device=self.device)

    def generate(self, z=None, n=None):
        return self(z=z, n=n)

    def predict(self, x):
        return self.adversariat(x)

    def get_hyperparameters(self):
        return self.hyperparameters

    def summary(self, save=False):
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
        [network.eval() for name, network in self.neural_nets.items()]

    def train(self):
        [network.train() for name, network in self.neural_nets.items()]

    def to(self, device):
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