import json
import torch

from torch.nn import Module, Sequential
from torchsummary import summary

class NeuralNetwork(Module):
    def __init__(self, architecture, name):
        super(NeuralNetwork, self).__init__()
        self.architecture = architecture
        self.name = name

        self._build_network()

    def summary(self):
        return summary(self, input_size=self.input_size)

    # TODO: JSON in/output
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
                self.input_size = tuple([input_size]) if isinstance(input_size, int) else input_size
                break

        self.out_features = []
        for i, (layer, params) in enumerate(self.architecture):
            if "out_features" in params:
                if "in_features" not in params:
                    params["in_features"] = self.out_features[-1]
                self.out_features.append(params["out_features"])

        self.network = Sequential(
            *[layer(**params) for layer, params in self.architecture]
        )

    def __str__(self):
        return self.name


class Generator(NeuralNetwork):
    def __init__(self, architecture):
        super(Generator, self).__init__(architecture, name="Generator")

    def forward(self, x):
        output = self.network(x)
        return output

    def sample(self, n):
        return torch.empty(n, self.input_size[0]).normal_(mean=0, std=1)

    def generate(self, n):
        sample_noise = self.sample(n=n)
        return self(sample_noise)


class Adversariat(NeuralNetwork):
    def __init__(self, architecture):
        super(Adversariat, self).__init__(architecture, "Adversariat")

    def forward(self, x):
        logits = self.network(x)
        return logits

