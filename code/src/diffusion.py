"""Diffusion noise schedule, fixed.

Bug fixed vs. the original notebook: it was instantiated with ``T=1``, which makes
``alpha_hat[0] ~= 1 - beta_start ~= 0.9999`` so almost no noise is ever added — the
paper's headline "diffusion-based denoising regularizer" was inert. Default restored
to the paper's T=1000.

Design note (see ANALYSIS.md): the network's head is a bounded ``sigmoid`` mask
output, not an unbounded noise residual, so it cannot be plugged into the standard
DDPM reverse update rule the paper's text describes (x_{t-1} = f(x_t, eps_pred)).
We therefore use the diffusion process for what it actually is in this architecture:
a stochastic noise-augmentation regularizer applied to the *input image* during
training (random t, then denoise-and-segment). At evaluation we score on the clean
image (t=0, no injected noise) since real deployment images are not synthetically
noised — this is the honest way to use this mechanism without pretending the model
does iterative generative sampling.
"""
import torch


class Diffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.device = device
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        return sqrt_alpha_hat * x_start + sqrt_one_minus_alpha_hat * noise, noise
