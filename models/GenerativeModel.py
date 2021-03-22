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


class GenerativeModel():
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(self, folder, enable_tensorboard, ngpu):
        folder = folder if not folder.endswith("/") else folder[-1]
        if os.path.exists(folder):
            now = datetime.now()
            now = now.strftime("%Y%m%d_%H%M%S")
            folder += now
        self.folder = folder + "/"
        os.makedirs(self.folder)

        self._define_loss()
        self.tensorboard_enabled = enable_tensorboard
        self.ngpu = ngpu

        assert hasattr(self, "generator"), "Model must have attribute 'generator'."
        assert hasattr(self, "adversariat"), "Model must have attribute 'adversariat'."
        assert hasattr(self, "neural_nets"), "Model must have attribute 'neural_nets'."
        assert hasattr(self, "device"), "Model must have attribute 'device'."
        assert hasattr(self, "optimizers"), "Model must have attribute 'optimizers'."
        assert hasattr(self, "generator_loss_fn"), "DualGAN needs attribute 'generator_loss_fn'."
        assert hasattr(self, "adversariat_loss_fn"), "DualGAN needs attribute 'adversariat_loss_fn'."
        self.to(self.device)
        self.images_produced = True if len(self.adversariat.input_size) > 1 else False
        self.fixed_noise = self.sample(n=24)

    def _define_losses(self):
        raise NotImplementedError("'_define_losses' must be implemented for objects of type 'GenerativeModel'.")

    def set_up_training(self, X_train, X_test=None, batch_size=32, max_test=300):
        assert X_train.shape[1:] == self.adversariat.input_size, (
            "Wrong input shape for adversariat. Given: {}. Needed: {}.".format(X_train.shape, self.adversariat.input_size)
        )

        train_data = utils.DataSet(X=X_train)
        train_dataloader = DataLoader(train_data, batch_size=batch_size)

        if (X_test is not None) and len(X_test)>max_test:
            X_test = X_test[:max_test]
            print("Warning: Only {} examples from X_test used for testing!".format(max_test))
            X_test = torch.from_numpy(X_test).to(self.device)

        if self.tensorboard_enabled:
            self._set_writers(set_test_writer=X_test is not None)
            inpt = self.sample(n=1)
            self.writer_train.add_graph(self.generator, inpt)

        self.total_training_time = 0
        self.current_timer = time.perf_counter()
        self.batch_training_times = []
        return train_dataloader, X_test

    def _set_writers(self, set_test_writer):
        self.writer_train = SummaryWriter(self.folder+"tensorboard/train/")
        if set_test_writer:
            self.writer_test = SummaryWriter(self.folder+"tensorboard/test/")


    #########################################################################
    # Actions during training
    #########################################################################
    def calculate_losses(self, X_batch, who=None):
        raise NotImplementedError("'calculate_losses' must be implemented by subclass.")

    def _zero_grad(self, who=None):
        if who is not None:
            self.optimizers[who].zero_grad()
        else:
            [optimizer.zero_grad() for _, optimizer in self.optimizers.items()]

    def _backward(self, who=None):
        assert len(self.losses) != 0, "'self.losses' empty when performing '_backward'."
        if who is not None:
            self.losses[who].backward(retain_graph=True)
        else:
            [loss.backward(retain_graph=True) for _, loss in self.losses.items()]

    def _step(self, who=None):
        if who is not None:
            self.optimizers[who].step()
        else:
            [optimizer.step() for _, optimizer in self.optimizers.items()]


    #########################################################################
    # Logging during training
    #########################################################################
    def log(self, X_batch, batch, max_batches, epoch, max_epochs, print_every, is_train=True, log_images=False):
        step = epoch*max_batches + batch
        self.calculate_losses(X_batch=X_batch)
        if is_train:
            max_steps = max_epochs*max_batches
            remaining_batches = max_batches - batch
            print("Step: {} / {} (Epoch: {} / {})".format(step, max_steps, epoch+1, max_epochs))
            print("---"*20)
            for name, loss in self.losses.items():
                print("{}: {}".format(name, loss.item()))

            self.batch_training_times.append(time.perf_counter() - self.current_timer)
            self.total_training_time = np.sum(self.batch_training_times)
            time_per_batch = np.mean(self.batch_training_times) / print_every

            print("\n")
            print("Time left: ~", np.round(remaining_batches*time_per_batch/60, 3), "minutes.")
            print("\n")
            self.current_timer = time.perf_counter()

        if X_batch is not None and self.tensorboard_enabled:
            self._log_scalars(X_batch=X_batch, step=step, is_train=is_train)
        if log_images and self.images_produced:
            self._log_images(step=step)

    def _log_scalars(self, X_batch, step, is_train):
        writer = self.writer_train if is_train else self.writer_test

        for name, loss in self.losses.items():
            writer.add_scalar("Loss/{}".format(name), loss.item(), step)
        writer.add_scalar(
            "Loss/LossRatio", (self.losses["Adversariat_real"]/self.losses["Adversariat_fake"]).item(), step
        )
        writer.add_scalar("Time/Total", self.total_training_time / 60, step)
        writer.add_scalar("Time/Batch", np.mean(self.batch_training_times) / 60, step)

    def _log_images(self, step):
        assert len(self.adversariat.input_size) > 1, "Called _log_images in GenerativeModel for adversariat.input_size = 1."
        images = self.generator(self.fixed_noise)
        if self.tensorboard_enabled:
            grid = make_grid(images)
            self.writer_train.add_image('images', grid, step)

        if not os.path.exists(self.folder+"Images"):
            os.mkdir(self.folder+"Images")
        images = images.cpu().detach().numpy()
        if len(images.shape) == 4:
            images = images.reshape((-1, *images.shape[-2:]))
        nrows = int(np.sqrt(len(images)))
        ncols = len(images) // nrows
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))

        for i, (ax, image) in enumerate(zip(np.ravel(axs), images)):
            ax.imshow(image)
            ax.axis("off")
        plt.savefig(self.folder+"Images/image_{}.png".format(step))
        print("Images logged.")


    #########################################################################
    # Cleaning after training
    #########################################################################
    def _clean_up(self):
        if self.tensorboard_enabled:
            self.writer_train.close()
            if hasattr(self, "writer_test"):
                self.writer_test.close()


    #########################################################################
    # Saving and loading
    #########################################################################
    def save(self):
        torch.save(self, self.folder+"model.torch")
        print("Model saved to {}.".format(self.folder))

    @staticmethod
    def load(path):
        return torch.load(path)

    def save_as_json(self, save=True, name="model"):
        json_dict = {}
        for network in self.neural_nets:
            json_dict.update(network.save_as_json(path=None))
        if save:
            with open(self.folder+name+'.json', 'w') as f:
                json.dump(json_dict, f, indent=4)
        return json_dict

    @classmethod
    def load_from_json(cls, path):
        with open(path, "r") as f:
            json_dict = json.load(f)
        for _, architecture in json_dict.items():
            for i, (layer, params) in enumerate(architecture):
                architecture[i][0] = eval(layer)
        return cls(
            generator_architecture=json_dict["Generator"],
            adversariat_architecture=json_dict["Adversariat"]
        )


    #########################################################################
    # Utility functions
    #########################################################################
    def sample(self, n):
        return torch.randn(n, *self.generator.input_size, requires_grad=True, device=self.device)

    def generate(self, z=None, n=None):
        return self(z=z, n=n)

    def predict(self, x):
        return self.adversariat(x)

    def get_hyperparameters(self):
        hyperparameter_dict = self.__dict__.copy()
        pop_keys = ["generator", "adversariat", "neural_nets"]
        for key in pop_keys:
            hyperparameter_dict.pop(key)
        return hyperparameter_dict

    def summary(self, save=False):
        if save:
            sys_stdout_temp = sys.stdout
            sys.stdout = open(self.folder+'summary.txt', 'w')
        for neural_net in self.neural_nets:
            print(neural_net)
            neural_net.summary()
            print("\n\n")
        if save:
            sys.stdout = sys_stdout_temp
            sys_stdout_temp

    def eval(self):
        [network.eval() for network in self.neural_nets]

    def train(self):
        [network.train() for network in self.neural_nets]

    def to(self, device):
        [network.to(device) for network in self.neural_nets]

    def __call__(self, z=None, n=None):
        if z is not None and n is not None:
            raise ValueError("Only one of 'z' and 'n' is needed.")
        if z is None and n is None:
            raise ValueError("Either 'z' or 'n' must be not None.")
        if n is not None:
            z = self.sample(n=n)
        return self.generator(z)

    def __str__(self):
        self.summary()