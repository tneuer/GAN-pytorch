import os
import sys
import json
import torch
import shutil

import utils.utils as utils

from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from utils.networks import Generator, Discriminator


class VanillaGAN():
    def __init__(self, generator_architecture, adversariat_architecture, folder="./VanillaGAN/"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = Generator(generator_architecture).to(self.device)
        self.adversariat = Discriminator(adversariat_architecture).to(self.device)
        self.neural_nets = (self.generator, self.adversariat)
        self.was_compiled = False
        self.folder = folder if folder.endswith("/") else folder+"/"

        if os.path.exists(self.folder):
            shutil.rmtree(self.folder)
        os.makedirs(self.folder)

    def compile(self, optim, optim_kwargs):
        self.optimizer_generator = optim(params=self.generator.parameters(), **optim_kwargs)
        self.optimizer_adversariat = optim(params=self.adversariat.parameters(), **optim_kwargs)
        self._define_loss()
        self.was_compiled = True

    def _define_loss(self):
        self.generator_loss_fn = BCELoss()
        self.adversariat_loss_fn = BCELoss()

    def train(self, X_train, X_test=None, epochs=5, batch_size=32):
        assert X_train.shape[1:] == self.adversariat.input_size, (
            "Wrong input shape. Given: {}. Needed: {}.".format(X_train.shape, self.adversariat.input_size)
        )
        assert self.was_compiled, "Model needs to be compiled first. Call model.compile()."

        train_data = utils.DataSet(X=X_train)
        test_data = utils.DataSet(X=X_test)
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)
        nr_batches = len(train_data) // batch_size
        self.losses = {"Generator": [], "Adversariat": [], "Adversariat_fake": [], "Adversariat_real": []}

        self._set_writer()

        for epoch in range(epochs):
            print("---"*20)
            print("EPOCH:", epoch)
            print("---"*20)
            for batch, X in enumerate(train_dataloader):
                fake_images = self.generator.generate(n=batch_size)

                fake_predictions = self.adversariat(fake_images)
                real_predictions = self.adversariat(X.float())

                self.generator_loss = self.generator_loss_fn(fake_predictions, torch.ones_like(fake_predictions))
                self.adversariat_fake_loss = self.adversariat_loss_fn(fake_predictions, torch.zeros_like(fake_predictions))
                self.adversariat_real_loss = self.adversariat_loss_fn(real_predictions, torch.ones_like(real_predictions))
                self.adversariat_loss = 0.5*(self.adversariat_fake_loss + self.adversariat_real_loss)
                self._zero_grad()
                self._backward()
                self._step()

                if batch % 100 == 0:
                    step_nr = epoch*nr_batches + batch
                    print("Iteration: {} / {}".format(step_nr, epochs*nr_batches))
                    print("---"*20)
                    print("Generator: {}".format(self.generator_loss.item()))
                    print("Adversariat: {}".format(self.adversariat_loss.item()))
                    print("\n")

                    self.log(X_test, step=step_nr)
            self.log(X_test, step=step_nr, log_images=True)

        self.writer_train.close()
        self.writer_test.close()

    def _zero_grad(self):
        self.optimizer_generator.zero_grad()
        self.optimizer_adversariat.zero_grad()

    def _backward(self):
        self.generator_loss.backward(retain_graph=True)
        self.adversariat_loss.backward(retain_graph=True)

    def _step(self):
        self.optimizer_generator.step()
        self.optimizer_adversariat.step()

    def _set_writer(self):
        self.writer_train = SummaryWriter(self.folder+"tensorboard/train/")
        self.writer_test = SummaryWriter(self.folder+"tensorboard/test/")

    def log(self, X_test, step, log_images=False):
        self._log_train(step)
        self._log_test(X_test, step)
        if log_images:
            self._log_images(step=step)
            print("Images logged.")

    def _log_train(self, step):
        self.losses["Generator"].append(self.generator_loss.item())
        self.losses["Adversariat"].append(self.adversariat_loss.item())
        self.losses["Adversariat_fake"].append(self.adversariat_fake_loss.item())
        self.losses["Adversariat_real"].append(self.adversariat_real_loss.item())
        self.writer_train.add_scalar("Loss/Generator", self.generator_loss.item(), step)
        self.writer_train.add_scalar("Loss/Adversariat", self.adversariat_loss.item(), step)
        self.writer_train.add_scalar("Loss/Adversariat_fake", self.adversariat_fake_loss.item(), step)
        self.writer_train.add_scalar("Loss/Adversariat_real", self.adversariat_real_loss.item(), step)

    def _log_test(self, X_test, step):
        fake_images = self.generator.generate(n=len(X_test))

        fake_predictions = self.adversariat(fake_images)
        real_predictions = self.adversariat(torch.from_numpy(X_test).float())

        generator_loss = self.generator_loss_fn(fake_predictions, torch.ones_like(fake_predictions))
        adversariat_fake_loss = self.adversariat_loss_fn(fake_predictions, torch.zeros_like(fake_predictions))
        adversariat_real_loss = self.adversariat_loss_fn(real_predictions, torch.ones_like(real_predictions))
        adversariat_loss = 0.5*(adversariat_fake_loss + adversariat_real_loss)
        self.writer_test.add_scalar("Loss/Generator", generator_loss.item(), step)
        self.writer_test.add_scalar("Loss/Adversariat", adversariat_loss.item(), step)
        self.writer_test.add_scalar("Loss/Adversariat_fake", adversariat_fake_loss.item(), step)
        self.writer_test.add_scalar("Loss/Adversariat_real", adversariat_real_loss.item(), step)

    def _log_images(self, step):
        images = self.generator.generate(n=9).detach().numpy().reshape((-1, 1, 28, 28))
        images = torch.from_numpy(images)
        grid = make_grid(images)
        self.writer_train.add_image('images', grid, step)

    def save(self):
        torch.save(self, self.folder+"model.torch")

    @staticmethod
    def load(path):
        return torch.load(path)

    # TODO: JSON in/output
    def save_as_json(self, save=True, name="model"):
        json_dict = {}
        for network in self.neural_nets:
            json_dict.update(network.save_as_json(path=None))
        if save:
            with open(self.folder+name+'.json', 'w') as f:
                json.dump(json_dict, f, indent=4)
        return json_dict

    @staticmethod
    def load_from_json(path):
        with open(path, "r") as f:
            json_dict = json.load(f)
        for _, architecture in json_dict.items():
            for i, (layer, params) in enumerate(architecture):
                architecture[i][0] = eval(layer)
        return VanillaGAN(
            generator_architecture=json_dict["Generator"],
            adversariat_architecture=json_dict["Adversariat"]
        )

    def summary(self, save=False):
        if save:
            sys_stdout_temp = sys.stdout
            sys.stdout = open(self.folder+'summary.txt', 'w')
        for neural_net in self.neural_nets:
            print(neural_net)
            neural_net.summary()
        if save:
            sys.stdout = sys_stdout_temp


if __name__ == '__main__':

    from torch import nn
    datapath = "./data/mnist/"
    X_train, y_train, X_test, y_test = utils.load_mnist(datapath, normalize=True, pad=0, return_datasets=False)

    input_size = 784
    z_dim = 32
    generator_architecture = [
        (nn.Linear, {"in_features": z_dim, "out_features": 128}),
        (nn.ReLU, {}),
        (nn.Linear, {"out_features": 784}),
        (nn.Sigmoid, {})
    ]
    adversariat_architecture = [
        (nn.Linear, {"in_features": input_size, "out_features": 128}),
        (nn.ReLU, {}),
        (nn.Linear, {"out_features": 11}),
        (nn.Sigmoid, {})
    ]
    vanilla_gan = VanillaGAN(
        generator_architecture, adversariat_architecture, folder="TrainedModels/GAN"
    )

    vanilla_gan.compile(optim=torch.optim.Adam, optim_kwargs={"lr": 1e-3})
    vanilla_gan.train(
        X_train=X_train.reshape((-1, 28*28)),
        X_test=X_test.reshape((-1, 28*28)),
        batch_size=64,
        epochs=1
    )
    vanilla_gan.summary(save=True)
    vanilla_gan.save_as_json()
    vanilla_gan.save()