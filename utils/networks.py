import json
import torch

from torch.nn import Module, Sequential
from torchsummary import summary


class NeuralNetwork(Module):
    def __init__(self, architecture, name, input_size):
        super(NeuralNetwork, self).__init__()
        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        if isinstance(input_size, int):
            self.input_size = tuple([input_size])
        elif isinstance(input_size, list):
            self.input_size = tuple(input_size)

        if isinstance(architecture, torch.nn.Module):
            self.network = architecture
        else:
            self.architecture = architecture
            self.network = self._build_network()

        print(self.network)
        raise

    def summary(self):
        print("Input shape: ", self.input_size)
        return summary(self, input_size=self.input_size)

    def save_as_json(self, path=None):
        json_dict = {}
        json_dict[self.name] = [["torch.nn."+layer.__name__, params] for layer, params in self.architecture]
        if path is not None:
            with open(self.folder+name+'.json', 'w') as f:
                json.dump(json_dict, f, indent=4)
        return json_dict

    @staticmethod
    def load_from_json(path):
        with open(path, "r") as f:
            json_dict = json.load(f)
        for name, architecture in json_dict.items():
            for i, (layer, params) in enumerate(architecture):
                architecture[i][0] = eval(layer)
        return NeuralNetwork(architecture=architecture, name=name)

    def _build_network(self):
        for i, (layer, params) in enumerate(self.architecture):
            if "out_features" in params:
                try:
                    input_size = params["in_features"]
                except KeyError:
                    raise KeyError("First layer with 'out_features' has to have 'in_features'.")
                break
            elif "out_channels" in params:
                try:
                    input_size = params["in_channels"]
                except KeyError:
                    raise KeyError("First layer with 'out_channels' has to have 'in_channels'.")
                break
        else:
            raise ValueError("At least one layer has to contain 'out_features' or 'out_channels'.")

        self.out_features = []
        for i, (layer, params) in enumerate(self.architecture):
            if "out_features" in params:
                if "in_features" not in params:
                    params["in_features"] = self.out_features[-1]
                self.out_features.append(params["out_features"])
            elif "out_channels" in params:
                if "in_channels" not in params:
                    params["in_channels"] = self.out_features[-1]
                self.out_features.append(params["out_channels"])
            elif "batchnorm" in layer.__name__.lower():
                params["num_features"] = self.out_features[-1]


        self.layers = []
        for layer, params in self.architecture:
            self.layers.append(layer(**params))

        network = Sequential(
            *self.layers
        )
        return network

    def forward(self, x):
        output = self.network(x)
        return output

    def __str__(self):
        return self.name


class Generator(NeuralNetwork):
    def __init__(self, architecture, input_size):
        super(Generator, self).__init__(architecture, input_size=input_size, name="Generator")


class Adversariat(NeuralNetwork):
    def __init__(self, architecture, input_size):
        super(Adversariat, self).__init__(architecture, input_size=input_size, name="Adversariat")


