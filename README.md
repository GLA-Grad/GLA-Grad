# [GLA-Grad](https://github.com/GLA-Grad/GLA-Grad)

![Python](https://img.shields.io/badge/Python-3.10.12-blue.svg)

![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-%23EE4C2C)



[GLA-Grad](https://github.com/GLA-Grad/GLA-Grad) aims at minimizing the conditioning error and increasing the efficiency of the noise diffusion process, and consists in introducing a phase recovery algorithm such as the Griffin-Lim algorithm (GLA) at each step of the regular diffusion process. The architecture is described in [GLA-Grad: A Griffin-lim Extended Waveform Generation Diffusion Model](https://arxiv.org/abs/2402.15516). 

*For a detailed explanation of each implementation please refer to the comments in the file.*



## Getting started 🚀

**We provide necessary informations to run the extention.**

### Inference with GLA

#### Initialization

Before applying the GLA extension, you need to initialize the `gla_extension` object: 

```
from gla_extension import gla_extension
gla_ext = gla_extension(model.params, GLA_steps, GLA_iteration, momentum)
```

- `GLA_steps`: The number of steps that the GLA intervenes the reverse process.
- `GLA_iteration`: The number of the alternative projection iterations within each GLA step.
- `momentum`:  Momentum factor for GLA iterations.

#### Pseudo Inversion

To generate a pseudo-inverse spectrogram using the Mel-scale, use the `pseudo_inverse` method:

```
spectrogram_pseudo = gla_ext.pseudo_inverse(spectrogram)
```

#### GLA steps

The `gla_update` method is used within the GLA extension to apply a single step of the Griffin-Lim Algorithm with momentum. The usage and parameters for this method are as follows:

```
audio = gla_ext.gla_update(alpha, n, audio, spectrogram_pseudo)
```

- `alpha` (list): A list of alpha values that are from the noise schedule of the diffusion process. 
-  `n` (int): This represents the current step number in the reverse diffusion process.
- `audio` (Tensor): This is the current audio signal tensor that the GLA step is being applied to. It should contain the audio data that has been processed up to the current step in the reverse diffusion process.
-  `spectrogram_pseudo` (Tensor): The pseudo-inverse spectrogram tensor. It is the result of applying the pseudo-inverse operation to the Mel spectrogram. 

#### Inference Loop

The following code presents how the GLA extension can be simply integrated into the waveform generative diffusion models, using the [inference of WaveGrad](https://github.com/lmnt-com/wavegrad) as an example:

```
for n in range(len(alpha) - 1, -1, -1):
  c1 = 1 / alpha[n]**0.5
  c2 = (1 - alpha[n]) / (1 - alpha_cum[n])**0.5
  audio = c1 * (audio - c2 * model(audio, spectrogram, noise_scale[n]).squeeze(1))
	
	# Apply the GLA step
  audio = gla_ext.gla_update(alpha, n, audio, spectrogram_pseudo)

  if n > 0:
    noise = torch.randn_like(audio)
    sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
    audio += sigma * noise
    audio = torch.clamp(audio, -1.0, 1.0)

audio = torch.clamp(audio, -1.0, 1.0)
```



## References

- [GLA-Grad: A Griffin-lim Extended Waveform Generation Diffusion Model](https://)
- [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/pdf/2009.00713.pdf)
- [Code for WaveGrad](https://github.com/lmnt-com/wavegrad)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Code for Denoising Diffusion Probabilistic Models](https://github.com/hojonathanho/diffusion)

