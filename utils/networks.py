import re
import json
import torch

from torch import nn
from torchsummary import summary
from torch.nn import Module, Sequential


class NeuralNetwork(Module):
    def __init__(self, network, name, input_size, ngpu):
        super(NeuralNetwork, self).__init__()
        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.ngpu = ngpu
        if isinstance(input_size, int):
            self.input_size = tuple([input_size])
        elif isinstance(input_size, list):
            self.input_size = tuple(input_size)

        assert isinstance(network, torch.nn.Module), "network must inherit from nn.Module."
        try:
            type(network[-1])
            self.input_type = "Sequential"
        except TypeError:
            self.input_type = "Custom"
        self.network = network

        if self.ngpu > 1:
            self.network = torch.nn.DataParallel(self.network)


    def forward(self, x):
        output = self.network(x)
        return output


    #########################################################################
    # Utility functions
    #########################################################################
    def summary(self):
        print("Input shape: ", self.input_size)
        return summary(self, input_size=self.input_size)

    def __str__(self):
        return self.name


class Generator(NeuralNetwork):
    def __init__(self, network, input_size, ngpu):
        super(Generator, self).__init__(network, input_size=input_size, name="Generator", ngpu=ngpu)


class Adversariat(NeuralNetwork):
    def __init__(self, network, input_size, adv_type, ngpu):
        valid_types = ["Discriminator", "Critic"]
        if adv_type == "Discriminator":
            valid_last_layer = torch.nn.Sigmoid
            self._type = "Discriminator"
        elif adv_type == "Critic":
            valid_last_layer = torch.nn.Linear
            self._type = "Critic"
        else:
            raise TypeError("adv_type must be one of {}.".format(valid_types))

        try:
            last_layer_type = type(network[-1])
        except TypeError:
            last_layer_type = type(network.__dict__["_modules"]["output"])
        assert last_layer_type == valid_last_layer, (
            "Last layer activation function of {} needs to be '{}'.".format(adv_type, valid_last_layer)
        )

        super(Adversariat, self).__init__(network, input_size=input_size, name="Adversariat", ngpu=ngpu)

    def predict(self, x):
        return self(x)
