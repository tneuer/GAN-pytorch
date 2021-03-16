import os
import sys
import json
import time
import torch
import shutil

import numpy as np
import utils.utils as utils
import matplotlib.pyplot as plt

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

        max_test = 300
        if (X_test is not None) and len(X_test)>max_test:
            X_test = X_test[:max_test]
            print("Warning: Only {} examples from X_test used for testing!".format(max_test))

        train_data = utils.DataSet(X=X_train)
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        X_test = torch.from_numpy(X_test).to(self.device)
        self._set_writer(set_test_writer=X_test is not None)

        inpt = self.generator.sample(n=1)
        self.writer_train.add_graph(self.generator, inpt)
        inpt = next(iter(train_dataloader)).to(self.device).float()
        self.writer_test.add_graph(self.adversariat, inpt)
        return train_dataloader, X_test

    def _set_writer(self, set_test_writer):
        self.writer_train = SummaryWriter(self.folder+"tensorboard/train/")
        if set_test_writer:
            self.writer_test = SummaryWriter(self.folder+"tensorboard/test/")

    def log(self, X_test, batch, max_batches, epoch, max_epochs, log_images=False):
        step = epoch*max_batches + batch
        max_steps = max_epochs*max_batches
        print("Iteration: {} / {} (Epoch: {} / {})".format(step, max_steps, epoch+1, max_epochs))
        print("---"*20)
        for name, loss in self.losses.items():
            print("{}: {}".format(name, loss.item()))
        print("\n")
        print("Time left: ~", np.round((max_steps - step)*np.mean(self.interval_times)/self.batch_log_steps/60, 3), "minutes.")
        print("\n")
        self._log_train(step=step)
        if X_test is not None:
            self._log_test(X_test=X_test, step=step)
        if log_images:
            self._log_images(step=step)

    def _log_train(self, step):
        for name, loss in self.losses.items():
            self.writer_train.add_scalar("Loss/{}".format(name), loss.item(), step)
        self.writer_train.add_scalar(
            "Loss/LossRatio", (self.losses["Adversariat_real"]/self.losses["Adversariat_fake"]).item(), step
        )
        self.writer_train.add_scalar("Time/Total", self.total_training_time / 60, step)
        self.writer_train.add_scalar("Time/Batch", np.mean(self.interval_times) / 60, step)

    def _log_test(self, X_test, step):
        self.calculate_losses(X_batch=X_test)
        for name, loss in self.losses.items():
            self.writer_test.add_scalar("Loss/{}".format(name), loss.item(), step)

    def _log_images(self, step):
        if len(self.adversariat.input_size) > 1:
            images = self.generator.generate(n=9)
            grid = make_grid(images)
            self.writer_train.add_image('images', grid, step)
            if not os.path.exists(self.folder+"Images"):
                os.mkdir(self.folder+"Images")
            images = images.cpu().detach().numpy()
            if len(images.shape) == 4:
                images = images.reshape((-1, *images.shape[-2:]))
            fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
            for i, ax in enumerate(np.ravel(axs)):
                ax.imshow(images[i])
                ax.axis("off")
            plt.savefig(self.folder+"Images/image_{}.png".format(step))
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
            print("\n\n")
        if save:
            sys.stdout = sys_stdout_temp
            sys_stdout_temp


    def eval(self):
        self.generator.eval()
        self.adversariat.eval()

    def __call__(self, x=None, n=None):
        if x is not None and n is not None:
            raise ValueError("Only 'x' or 'n' is needed.")
        elif x is not None:
            return self.generator(x)
        elif n is not None:
            return self.generator.generate(n=n)


class BivariateGAN(GenerativeModel):
    def __init__(self, generator_architecture, adversariat_architecture, z_dim, in_dim, folder):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = Generator(generator_architecture, input_size=z_dim).to(self.device)
        self.adversariat = Adversariat(adversariat_architecture, input_size=in_dim).to(self.device)
        self.neural_nets = (self.generator, self.adversariat)
        super(BivariateGAN, self).__init__(folder=folder)

    def compile(self, optim, generator_kwargs, adversariat_kwargs):
        self.optimizer_generator = optim(params=self.generator.parameters(), **generator_kwargs)
        self.optimizer_adversariat = optim(params=self.adversariat.parameters(), **adversariat_kwargs)
        self.optimizers = {"Generator": self.optimizer_generator, "Adversariat": self.optimizer_adversariat}
        super(BivariateGAN, self).compile()
        assert hasattr(self, "generator_loss_fn"), "BivariateGAN needs attribute 'generator_loss_fn'."
        assert hasattr(self, "adversariat_loss_fn"), "BivariateGAN needs attribute 'adversariat_loss_fn'."

    def train(self, X_train, X_test=None, epochs=5, batch_size=32, gen_steps=1, adv_steps=1, batch_log_steps=200):
        train_dataloader, X_test = self.set_up_training(X_train, X_test=X_test)
        nr_batches = len(train_dataloader)

        self.total_training_time = 0
        self.current_timer = time.perf_counter()
        self.interval_times = []
        self.batch_log_steps = batch_log_steps
        self.calculate_losses(X_batch=X_test)
        self.log(
            X_test, batch=0, max_batches=nr_batches, epoch=0, max_epochs=epochs, log_images=True
        )
        for epoch in range(epochs):
            print("---"*20)
            print("EPOCH:", epoch+1)
            print("---"*20)
            for batch, X in enumerate(train_dataloader):
                batch += 1
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

                if batch % batch_log_steps == 0:
                    self.interval_times.append(time.perf_counter() - self.current_timer)
                    self.total_training_time = np.sum(self.interval_times)
                    self.calculate_losses(X_batch=X)
                    self.log(
                        X_test, batch=batch, max_batches=nr_batches, epoch=epoch, max_epochs=epochs, log_images=False
                    )
                    self.current_timer = time.perf_counter()

            self.calculate_losses(X_batch=X)
            self.log(
                X_test, batch=batch, max_batches=nr_batches, epoch=epoch, max_epochs=epochs, log_images=True
            )

        self._clean_up_training()


    def _backward(self, who=None):
        if who is not None:
            self.losses[who].backward(retain_graph=True)
        else:
            self.losses["Generator"].backward(retain_graph=True)
            self.losses["Adversariat"].backward(retain_graph=True)



class VanillaGAN(BivariateGAN):
    def __init__(self, generator_architecture, adversariat_architecture, z_dim, in_dim, folder="./VanillaGANModel/"):
        assert adversariat_architecture[-1][0] == torch.nn.Sigmoid, (
            "Last layer activation function needs to be 'torch.nn.sigmoid'."
        )
        super(VanillaGAN, self).__init__(
            generator_architecture=generator_architecture, adversariat_architecture=adversariat_architecture,
            z_dim=z_dim, in_dim=in_dim, folder=folder
        )

    def _define_loss(self):
        from torch.nn import BCELoss
        self.generator_loss_fn = BCELoss()
        self.adversariat_loss_fn = BCELoss()

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
        fake_images = self.generator.generate(n=len(X_batch))
        fake_predictions = self.adversariat(fake_images)
        gen_loss = self.generator_loss_fn(
            fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
        )
        self.losses.update({"Generator": gen_loss})

    def _calculate_adversariat_loss(self, X_batch):
        fake_images = self.generator.generate(n=len(X_batch)).detach()
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
        self.real_predictions = real_predictions
        self.fake_predictions = fake_predictions


class WassersteinGAN(BivariateGAN):
    def __init__(self, generator_architecture, adversariat_architecture, z_dim, in_dim, folder="./WassersteinModel/"):
        assert adversariat_architecture[-1][0] == torch.nn.Linear, (
            "Last layer activation function needs to be 'torch.nn.Linear'."
        )
        super(WassersteinGAN, self).__init__(
            generator_architecture=generator_architecture, adversariat_architecture=adversariat_architecture,
            z_dim=z_dim, in_dim=in_dim, folder=folder
        )

    def _define_loss(self):
        self.generator_loss_fn = self.wasserstein_loss
        self.adversariat_loss_fn = self.wasserstein_loss

    def wasserstein_loss(self, input, target):
        import numpy as np
        if np.all((target==1).cpu().numpy()):
            return -torch.mean(input)
        elif np.all((target==0).cpu().numpy()):
            return torch.mean(input)

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
        fake_images = self.generator.generate(n=len(X_batch))
        fake_predictions = self.adversariat(fake_images)
        gen_loss = self.generator_loss_fn(
            fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
        )
        self.losses.update({"Generator": gen_loss})

    def _calculate_adversariat_loss(self, X_batch):
        fake_images = self.generator.generate(n=len(X_batch)).detach()
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
        self.real_predictions = real_predictions
        self.fake_predictions = fake_predictions

    def _step(self, who=None):
        if who is not None:
            self.optimizers[who].step()
            if who == "Adversariat":
                for p in self.adversariat.parameters():
                    p.data.clamp_(-0.01, 0.01)
        else:
            [optimizer.step() for _, optimizer in self.optimizers.items()]



class WassersteinGAN_GP(BivariateGAN):
    def __init__(self, generator_architecture, adversariat_architecture, z_dim, in_dim, folder="./WassersteinGPModel/"):
        assert adversariat_architecture[-1][0] == torch.nn.Linear, (
            "Last layer activation function needs to be 'torch.nn.Linear'."
        )
        super(WassersteinGAN_GP, self).__init__(
            generator_architecture=generator_architecture, adversariat_architecture=adversariat_architecture,
            z_dim=z_dim, in_dim=in_dim, folder=folder
        )

    def compile(self, optim, generator_kwargs, adversariat_kwargs, lmbda_grad=10):
        super(WassersteinGAN_GP, self).compile(optim, generator_kwargs, adversariat_kwargs)
        self.lmbda_grad = lmbda_grad

    def _define_loss(self):
        self.generator_loss_fn = self.wasserstein_loss
        self.adversariat_loss_fn = self.wasserstein_loss
        self.gradient_penalty_fn = self.gradient_penalty

    def gradient_penalty(self, real_images, fake_images):
        alpha = torch.Tensor(np.random.random((real_images.size(0), 1, 1, 1))).to(self.device)
        interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True).float()
        d_interpolates = self.adversariat(interpolates).to(self.device)
        dummy = torch.ones_like(d_interpolates, requires_grad=False).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=dummy,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def wasserstein_loss(self, input, target):
        import numpy as np
        if np.all((target==1).cpu().numpy()):
            return -torch.mean(input)
        elif np.all((target==0).cpu().numpy()):
            return torch.mean(input)

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
        fake_images = self.generator.generate(n=len(X_batch))
        fake_predictions = self.adversariat(fake_images)
        gen_loss = self.generator_loss_fn(
            fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
        )
        self.losses.update({"Generator": gen_loss})

    def _calculate_adversariat_loss(self, X_batch):
        fake_images = self.generator.generate(n=len(X_batch)).detach()
        fake_predictions = self.adversariat(fake_images)
        real_predictions = self.adversariat(X_batch.float())

        adv_loss_fake = self.adversariat_loss_fn(
            fake_predictions, torch.zeros_like(fake_predictions, requires_grad=False)
        )
        adv_loss_real = self.adversariat_loss_fn(
            real_predictions, torch.ones_like(real_predictions, requires_grad=False)
        )
        adv_loss_grad = self.gradient_penalty_fn(X_batch, fake_images)
        adv_loss = 0.5*(adv_loss_fake + adv_loss_real) + self.lmbda_grad*adv_loss_grad
        self.losses.update({
            "Adversariat": adv_loss,
            "Adversariat_fake": adv_loss_fake,
            "Adversariat_real": adv_loss_real,
            "Adversariat_grad": adv_loss_grad
        })
        self.real_predictions = real_predictions
        self.fake_predictions = fake_predictions



class VariationalAutoencoder(BivariateGAN):
    def __init__(self, generator_architecture, adversariat_architecture, z_dim, in_dim, folder="./VanillaGANModel/"):
        assert adversariat_architecture[-1][0] == torch.nn.Sigmoid, (
            "Last layer activation function needs to be 'torch.nn.sigmoid'."
        )
        super(VariationalAutoencoder, self).__init__(
            generator_architecture=generator_architecture, adversariat_architecture=adversariat_architecture,
            z_dim=z_dim, in_dim=in_dim, folder=folder
        )

    def _define_loss(self):
        from torch.nn import BCELoss
        self.generator_loss_fn = BCELoss()
        self.adversariat_loss_fn = BCELoss()

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
        fake_images = self.generator.generate(n=len(X_batch))
        fake_predictions = self.adversariat(fake_images)
        gen_loss = self.generator_loss_fn(
            fake_predictions, torch.ones_like(fake_predictions, requires_grad=False)
        )
        self.losses.update({"Generator": gen_loss})

    def _calculate_adversariat_loss(self, X_batch):
        fake_images = self.generator.generate(n=len(X_batch)).detach()
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
        self.real_predictions = real_predictions
        self.fake_predictions = fake_predictions