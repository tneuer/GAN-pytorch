import torch
import utils.utils as utils
from utils.models import VanillaGAN, WassersteinGAN, WassersteinGAN_GP


if __name__ == '__main__':

    from torch import nn
    datapath = "./data/mnist/"
    X_train, y_train, X_test, y_test = utils.load_mnist(datapath, normalize=True, pad=0, return_datasets=False)

    z_dim = 32
    generator_architecture = [
        (nn.Linear, {"in_features": z_dim, "out_features": 128}),
        (nn.Dropout, {"p": 0.5}),
        (nn.ReLU, {}),
        (nn.Linear, {"out_features": 784}),
        (nn.Sigmoid, {})
    ]
    input_size = 784
    adversariat_architecture = [
        (nn.Linear, {"in_features": input_size, "out_features": 128}),
        (nn.ReLU, {}),
        (nn.BatchNorm1d, {"num_features": 128}),
        (nn.Linear, {"out_features": 1}),
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