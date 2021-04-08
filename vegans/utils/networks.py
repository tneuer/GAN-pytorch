import re
import json
import torch

import numpy as np

from torch import nn
from torchsummary import summary
from torch.nn import Module, Sequential


class NeuralNetwork(Module):
    def __init__(self, network, name, input_size, device, ngpu):
        super(NeuralNetwork, self).__init__()
        self.name = name
        self.input_size = input_size
        self.device = device
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
            self.input_type = "Object"
        self.network = network.to(self.device)
        self._validate_input()

        if self.device=="cuda" and self.ngpu is not None:
            if self.ngpu > 1:
                self.network = torch.nn.DataParallel(self.network)

    def forward(self, x):
        output = self.network(x)
        return output

    def _validate_input(self):
        iterative_layers = self._get_iterative_layers()

        for layer in iterative_layers:
            if "in_features" in layer.__dict__:
                first_input = layer.__dict__["in_features"]
                break
            elif "in_channels" in layer.__dict__:
                first_input = layer.__dict__["in_channels"]
                break
            elif "num_features" in layer.__dict__:
                first_input = layer.__dict__["num_features"]
                break

        if np.prod([first_input]) == np.prod(self.input_size):
            pass
        elif (len(self.input_size) > 1) & (self.input_size[0] == first_input):
            pass
        else:
            raise TypeError(
                "\n\tInput mismatch for {}:\n".format(self.name) +
                "\t\tFirst input layer 'in_features' or 'in_channels': {}. self.input_size: {}.\n".format(
                    first_input, self.input_size
                ) +
                "\t\tMaybe forgot to adjust size of input layer for y_dim."
            )
        return True

    def _get_iterative_layers(self):
        if self.input_type == "Sequential":
            return self.network
        elif self.input_type == "Object":
            iterative_net = []
            for _, layers in self.network.__dict__["_modules"].items():
                try:
                    for layer in layers:
                        iterative_net.append(layer)
                except TypeError:
                    iterative_net.append(layers)
            return iterative_net
        else:
            raise NotImplemented("Network must be Sequential or Object.")


    #########################################################################
    # Utility functions
    #########################################################################
    def summary(self):
        print("Input shape: ", self.input_size)
        return summary(self, input_size=self.input_size, device=self.device)

    def __str__(self):
        return self.name


class Generator(NeuralNetwork):
    def __init__(self, network, input_size, device, ngpu):
        super().__init__(network, input_size=input_size, name="Generator", device=device, ngpu=ngpu)


class Adversariat(NeuralNetwork):
    def __init__(self, network, input_size, adv_type, device, ngpu):
        valid_types = ["Discriminator", "Critic"]
        if adv_type == "Discriminator":
            valid_last_layer = [torch.nn.Sigmoid]
            self._type = "Discriminator"
        elif adv_type == "Critic":
            valid_last_layer = [torch.nn.Linear, torch.nn.Identity]
            self._type = "Critic"
        else:
            raise TypeError("adv_type must be one of {}.".format(valid_types))

        try:
            last_layer_type = type(network[-1])
        except TypeError:
            last_layer_type = type(network.__dict__["_modules"]["output"])
        assert last_layer_type in valid_last_layer, (
            "Last layer activation function of {} needs to be one of '{}'.".format(adv_type, valid_last_layer)
        )

        super().__init__(network, input_size=input_size, name="Adversariat", device=device, ngpu=ngpu)

    def predict(self, x):
        return self(x)


class Encoder(NeuralNetwork):
    def __init__(self, network, input_size, device, ngpu):
        super().__init__(network, input_size=input_size, name="Encoder", device=device, ngpu=ngpu)