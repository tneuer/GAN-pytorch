# VeGANs

A library to easily train various existing GANs (Generative Adversarial Networks) in PyTorch.

This library targets mainly GAN users, who want to use existing GAN training techniques with their own generators/discriminators.
However researchers may also find the GAN base class useful for quicker implementation of new GAN training techniques.

The focus is on simplicity and providing reasonable defaults.

## How to install
You need python 3.5 or above. Then:
`pip install vegans`

## How to use
The basic idea is that the user provides discriminator and generator networks, and the library takes care of training them in a selected GAN setting:
```
from vegans.models.GAN import WassersteinGAN
from vegans.utils import plot_losses, plot_images

generator = ### Your generator (torch.nn.Module)
adversariat = ### Your discriminator/critic (torch.nn.Module)
X_train = ### Your dataloader (torch.utils.data.DataLoader) or pd.DataFrame

z_dim = 64
x_dim = X_train.shape[1:] # [nr_channels, height, width]

# Build a WassersteinGAN
gan = WassersteinGAN(generator, discriminator, z_dim, x_dim)

# Fit the WassersteinGAN
gan.fit(X_train)

# Vizualise results
images, losses = gan.get_training_results()
images = images.reshape(-1, *samples.shape[2:]) # remove nr_channels
plot_images(images)
plot_losses(losses)
```

You can currently use the following GANs:
* `VanillaGAN`: [Classic minimax GAN](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf), in its non-saturated version
* `WassersteinGAN`: [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
* `WassersteinGANGP`: [Wasserstein GAN with gradient penalty](https://arxiv.org/abs/1704.00028)

All current GAN implementations come with a conditional variant to allow for the usage of training labels to produce specific outputs:

- ConditionalVanillaGAN
- ConditionalWassersteinGAN
- ...

### Slightly More Details:
All of the GAN objects inherit from a `GenerativeModel` base class. When building any such GAN, you must give in argument a generator and discriminator networks (some `torch.nn.Module`), as well as a the dimensions of the latent space `z_dim` and input dimension of the images `x_dim`. In addition, you can specify some parameters supported by all GAN implementations:
* `optim`:
* `optim_kwargs`:
* `fixed_noise_size`: The number of samples to save (from fixed noise vectors)
* `device`:
* `folder`:
* `ngpu`:

The fit function takes the following optional arguments:

- 



If you are researching new GAN training algorithms, you may find it useful to inherit from the `GAN` base class.

### Learn more:
Currently the best way to learn more about how to use VeGANs is to have a look at the example [notebooks](https://github.com/unit8co/vegans/tree/master/notebooks).
You can start with this [simple example](https://github.com/unit8co/vegans/blob/master/notebooks/00_univariate_gaussian.ipynb) showing how to sample from a univariate Gaussian using a GAN.
Alternatively, can run example [scripts](https://github.com/unit8co/vegans/tree/master/examples).

## Contribute
PRs and suggestions are welcome. Look [here](https://github.com/unit8co/vegans/blob/master/CONTRIBUTING) for more details on the setup.

## Credits
Some of the code has been inspired by some existing GAN implementations:
* https://github.com/eriklindernoren/PyTorch-GAN
* https://github.com/martinarjovsky/WassersteinGAN
* https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

## TODO

- GAN Implementations
  - BEGAN
  - EBGAN
  - LR-GAN
  - BicycleGAN
  - VAEGAN
  - CycleGAN
  - InfoGAN
  - Least Squares GAN
  - Pix2Pix
  - WassersteinGAN SpectralNorm
  - DiscoGAN
  - Adversarial Autoencoder
- Layers
  - Inception
  - Residual Block
- Other
  - Feature loss
  - Do not save Discriminator 
  - Translate examples to jupyter
  - How to make your own architecture 
    - _define_optimizers
    - fit
    - calculate_losses
  - ~~folder = None~~
  - save every = None
  - ~~default optimizers~~

