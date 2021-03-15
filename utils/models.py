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
from utils.networks import Generator, Adversariat


class GenerativeModel():
    def __init__(self, folder):
        self.was_compiled = False
        self.folder = folder if folder.endswith("/") else folder+"/"

        if os.path.exists(self.folder):
            shutil.rmtree(self.folder)
        os.makedirs(self.folder)

        assert hasattr(self, "generator"), "Model must have attribute 'generator'."
        assert hasattr(self, "adversariat"), "Model must have attribute 'adversariat'."
        assert hasattr(self, "neural_nets"), "Model must have attribute 'neural_nets'."

    def compile(self):
        self._define_loss()
        self.was_compiled = True
        assert hasattr(self, "optimizers"), "Model must have attribute 'optimizers'."

    def set_up_training(self, X_train, X_test=None, batch_size=32):
        assert X_train.shape[1:] == self.adversariat.input_size, (
            "Wrong input shape. Given: {}. Needed: {}.".format(X_train.shape, self.adversariat.input_size)
        )
        assert self.was_compiled, "Model needs to be compiled first. Call model.compile()."

        train_data = utils.DataSet(X=X_train)
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        self._set_writer(set_test_writer=X_test is not None)
        return train_dataloader

    def log(self, X_test, step, max_steps, log_images=False):
        print("Iteration: {} / {}".format(step, max_steps))
        print("---"*20)
        for name, loss in self.losses.items():
            print("{}: {}".format(name, loss.item()))
        print("\n")
        self._log_train(step=step)
        if X_test is not None:
            self._log_test(X_test=torch.from_numpy(X_test), step=step)
        if log_images:
            self._log_images(step=step)

    def _log_train(self, step):
        for name, loss in self.losses.items():
            self.writer_train.add_scalar("Loss/{}".format(name), loss.item(), step)

    def _log_test(self, X_test, step):
        self.calculate_losses(X_batch=X_test)
        for name, loss in self.losses.items():
            self.writer_test.add_scalar("Loss/{}".format(name), loss.item(), step)

    def _log_images(self, step):
        if len(self.adversariat.input_size) > 1:
            images = self.generator.generate(n=9)
            grid = make_grid(images)
            self.writer_train.add_image('images', grid, step)
            print("Images logged.")

    def _zero_grad(self, who=None):
        if who is not None:
            self.optimizers[who].zero_grad()
        else:
            [optimizer.zero_grad() for _, optimizer in self.optimizers.items()]

    def _step(self, who=None):
        if who is not None:
            self.optimizers[who].step()
        else:
            [optimizer.step() for _, optimizer in self.optimizers.items()]

    def _set_writer(self, set_test_writer):
        self.writer_train = SummaryWriter(self.folder+"tensorboard/train/")
        if set_test_writer:
            self.writer_test = SummaryWriter(self.folder+"tensorboard/test/")

    def _clean_up_training(self):
        self.writer_train.close()
        try:
            self.writer_test.close()
        except AttributeError:
            pass

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

    def summary(self, save=False):
        if save:
            sys_stdout_temp = sys.stdout
            sys.stdout = open(self.folder+'summary.txt', 'w')
        for neural_net in self.neural_nets:
            print(neural_net)
            neural_net.summary()
        if save:
            sys.stdout = sys_stdout_temp



class BivariateGAN(GenerativeModel):
    def __init__(self, generator_architecture, adversariat_architecture, folder):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = Generator(generator_architecture).to(self.device)
        self.adversariat = Adversariat(adversariat_architecture).to(self.device)
        self.neural_nets = (self.generator, self.adversariat)
        super(BivariateGAN, self).__init__(folder=folder)

    def compile(self, optim, optim_kwargs):
        self.optimizer_generator = optim(params=self.generator.parameters(), **optim_kwargs)
        self.optimizer_adversariat = optim(params=self.adversariat.parameters(), **optim_kwargs)
        self.optimizers = {"Generator": self.optimizer_generator, "Adversariat": self.optimizer_adversariat}
        super(BivariateGAN, self).compile()
        assert hasattr(self, "generator_loss_fn"), "BivariateGAN needs attribute 'generator_loss_fn'."
        assert hasattr(self, "adversariat_loss_fn"), "BivariateGAN needs attribute 'adversariat_loss_fn'."

    def train(self, X_train, X_test=None, epochs=5, batch_size=32, gen_steps=1, adv_steps=1):
        train_dataloader = self.set_up_training(X_train, X_test=X_test)
        nr_batches = len(train_dataloader)
        max_steps = epochs*nr_batches

        for epoch in range(epochs):
            print("---"*20)
            print("EPOCH:", epoch)
            print("---"*20)
            for batch, X in enumerate(train_dataloader):
                for _ in range(adv_steps):
                    self.calculate_losses(X_batch=X)
                    self._zero_grad(who="Adversariat")
                    self._backward(who="Adversariat")
                    self._step(who="Adversariat")

                for _ in range(gen_steps):
                    self.calculate_losses(X_batch=X)
                    self._zero_grad(who="Generator")
                    self._backward(who="Generator")
                    self._step(who="Generator")

                if batch % 100 == 0:
                    step_nr = epoch*nr_batches + batch
                    self.log(X_test, step=step_nr, max_steps=max_steps)
            self.log(X_test, step=step_nr, max_steps=max_steps, log_images=True)

        self._clean_up_training()


    def _backward(self, who=None):
        if who is not None:
            self.losses[who].backward(retain_graph=True)
        else:
            self.losses["Generator"].backward(retain_graph=True)
            self.losses["Adversariat"].backward(retain_graph=True)



class VanillaGAN(BivariateGAN):
    def __init__(self, generator_architecture, adversariat_architecture, folder):
        assert adversariat_architecture[-1][0] == torch.nn.Sigmoid, (
            "Last layer activation function needs to be 'torch.nn.sigmoid'."
        )
        super(VanillaGAN, self).__init__(generator_architecture, adversariat_architecture, folder)

    def _define_loss(self):
        from torch.nn import BCELoss
        self.generator_loss_fn = BCELoss()
        self.adversariat_loss_fn = BCELoss()

    def calculate_losses(self, X_batch):
        fake_images = self.generator.generate(n=len(X_batch)).detach()
        fake_predictions = self.adversariat(fake_images)
        real_predictions = self.adversariat(X_batch.float())

        gen_loss = self.generator_loss_fn(
            fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
        )
        adv_loss_fake = self.adversariat_loss_fn(
            fake_predictions, torch.zeros_like(fake_predictions, requires_grad=False)
        )
        adv_loss_real = self.adversariat_loss_fn(
            real_predictions, torch.ones_like(real_predictions, requires_grad=False)
        )
        adv_loss = 0.5*(adv_loss_fake + adv_loss_real)
        self.losses = {
            "Generator": gen_loss,
            "Adversariat": adv_loss,
            "Adversariat_fake": adv_loss_fake,
            "Adversariat_real": adv_loss_real,
        }


class WassersteinGAN(BivariateGAN):
    def __init__(self, generator_architecture, adversariat_architecture, folder):
        assert adversariat_architecture[-1][0] == torch.nn.Linear, (
            "Last layer activation function needs to be 'torch.nn.Linear'."
        )
        super(VanillaGAN, self).__init__(generator_architecture, adversariat_architecture, folder)

    def _define_loss(self):
        self.generator_loss_fn = self.wasserstein_loss
        self.adversariat_loss_fn = self.wasserstein_loss

    def wasserstein_loss(self, input, target):
        import numpy as np
        if np.all((target==1).numpy()):
            return -torch.mean(input)
        elif np.all((target==0).numpy()):
            return torch.mean(input)

    def calculate_losses(self, X_batch):
        fake_images = self.generator.generate(n=len(X_batch))
        fake_predictions = self.adversariat(fake_images)
        real_predictions = self.adversariat(X_batch.float())

        gen_loss = self.generator_loss_fn(fake_predictions, torch.ones_like(fake_predictions))
        adv_loss_fake = self.adversariat_loss_fn(fake_predictions, torch.zeros_like(fake_predictions))
        adv_loss_real = self.adversariat_loss_fn(real_predictions, torch.ones_like(real_predictions))
        adv_loss = 0.5*(adv_loss_fake + adv_loss_real)
        self.losses = {
            "Generator": gen_loss,
            "Adversariat": adv_loss,
            "Adversariat_fake": adv_loss_fake,
            "Adversariat_real": adv_loss_real,
        }

    def _step(self, who=None):
        if who is not None:
            self.optimizers[who].step()
            if who == "Adversariat":
                for p in self.adversariat.parameters():
                    p.data.clamp_(-0.01, 0.01)
        else:
            [optimizer.step() for _, optimizer in self.optimizers.items()]


class WassersteinGAN_GP(WassersteinGAN):
    def _define_loss(self):
        self.generator_loss_fn = self.wasserstein_loss
        self.adversariat_loss_fn = self.wasserstein_loss
        self.gradient_penalty_fn = self.gradient_penalty

    def gradient_penalty(self, real_images, fake_images):
        from torch import Tensor
        from torch.autograd import Variable
        import numpy as np
        alpha = Tensor(np.random.random((real_images.size(0), 1, 1, 1)))
        interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)
        d_interpolates = self.adversariat(interpolates)
        fake = Variable(Tensor(real_images.shape[0], 1).fill_(1.0), requires_grad=False)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def calculate_losses(self, X_batch, lmbda_gp=10):
        fake_images = self.generator.generate(n=len(X_batch))
        fake_predictions = self.adversariat(fake_images)
        real_predictions = self.adversariat(X_batch.float())

        gen_loss = self.generator_loss_fn(fake_predictions, torch.ones_like(fake_predictions))
        adv_loss_fake = self.adversariat_loss_fn(fake_predictions, torch.zeros_like(fake_predictions))
        adv_loss_real = self.adversariat_loss_fn(real_predictions, torch.ones_like(real_predictions))
        adv_grad_penatly = self.gradient_penalty_fn(X_batch, fake_images)
        adv_loss = 0.5*(adv_loss_fake + adv_loss_real) + lmbda_gp*adv_grad_penatly
        self.losses = {
            "Generator": gen_loss,
            "Adversariat": adv_loss,
            "Adversariat_fake": adv_loss_fake,
            "Adversariat_real": adv_loss_real,
            "Adversariat_grad": adv_grad_penatly,
        }

    def _step(self, who=None):
        if who is not None:
            self.optimizers[who].step()
            if who == "Adversariat":
                for p in self.adversariat.parameters():
                    p.data.clamp_(-0.01, 0.01)
        else:
            [optimizer.step() for _, optimizer in self.optimizers.items()]
