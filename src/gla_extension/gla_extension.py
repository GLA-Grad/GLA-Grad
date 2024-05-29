import torch
from preprocess import get_spec
from librosa.filters import mel as librosa_mel_fn
from torch.nn.functional import normalize


class gla_extension():
  """
  Attributes:
      params (dict): Parameters dictionary containing detailed configuration.
      n_mels (int): Number of Mel bands to generate.
      fmax (float): Highest frequency (in Hz) when constructing Mel-scale.
      fmin (float): Lowest frequency (in Hz) when constructing Mel-scale.
      hop (int): Hop length for STFT.
      win (int): Window size for STFT.
      n_fft (int): Number of FFT components.
      momentum (float): Momentum factor for GLA iterations.
  """

  def __init__(self, params, GLA_steps=3, GLA_iteration=32, momentum=0.1, n_mels=128, fmax=11025, fmin=20):
    """
    Parameters:
      GLA_steps: The number of steps that the GLA intervenes the reverse process.
      GLA_iteration: The number of the alternative projection iterations within each GLA step.
      momentum: Momentum factor for GLA iterations.
      params (dict): Parameters dictionary containing detailed configuration.
      n_mels (int, optional): Number of Mel bands to generate. Defaults to 128.
      fmax (float, optional): Highest frequency (in Hz) when constructing Mel-scale. Defaults to 11025.
      fmin (float, optional): Lowest frequency (in Hz) when constructing Mel-scale. Defaults to 20.
    """
    self.mel_basis = {}
    self.hann_window = {}
    self.params = params

    self.hop = self.params.hop_samples
    self.win = self.hop * 4
    self.n_fft = 2 ** ((self.win - 1).bit_length())

    self.n_mels = n_mels
    self.fmax = fmax
    self.fmin = fmin

    self.momentum = momentum
    self.GLA_steps = GLA_steps
    self.GLA_iteration = GLA_iteration

  def pseudo_inverse(self, spectrogram):
    """
    Computes the pseudo-inverse STFT of a given mel-spectrogram using the Mel scale.

    Parameters:
        spectrogram (Tensor): The input spectrogram for which to compute the pseudo-inverse.

    Returns:
        Tensor: The computed pseudo-inverse STFT spectrogram.
    """
    device = spectrogram.device

    if self.fmax not in self.mel_basis:
      mel = librosa_mel_fn(sr=self.params.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
      self.mel_basis[str(self.fmax) + '_' + str(device)] = torch.from_numpy(mel).float().to(device)
      self.hann_window[str(device)] = torch.hann_window(self.win).to(device)
    fb_pseudo_inverse = torch.linalg.pinv(self.mel_basis[str(self.fmax) + '_' + str(device)])
    spectrogram_pseudo = fb_pseudo_inverse.to(device) @ torch.exp(spectrogram.to(device))

    # Expand rank 2 tensors by adding a batch dimension.
    if len(spectrogram.shape) == 2:
      spectrogram_pseudo = spectrogram_pseudo.unsqueeze(0)
      spectrogram_pseudo = torch.cat((spectrogram_pseudo, torch.zeros((1, spectrogram_pseudo.shape[1], 1)).to(device)), dim=2)

    return spectrogram_pseudo

  def gla_update(self, alpha, n, audio, spectrogram_pseudo):
    """
    Performs one step of the Griffin-Lim Algorithm (GLA) with momentum.

    Parameters:
        alpha (list or Tensor): List of alpha values from noise schedule.
        n (int): The current step number.
        audio (Tensor): The current audio signal to process.
        spectrogram_pseudo (Tensor): The pseudo-inverse spectrogram.

    Returns:
        Tensor: The updated audio signal after applying the GLA step in reverse process.
    """

    if n > (len(alpha) - int(self.GLA_steps) - 1):
      # print(f'Inverse step: {n}, applying GLA')
      STFT_in = get_spec(audio, self.params, center=True, if_mel=False)
      tprev = torch.tensor(0.0, dtype=STFT_in.dtype, device=STFT_in.device)
      STFT_P1 = STFT_in
      tprev = STFT_P1

      if spectrogram_pseudo.shape[-1] != STFT_P1.shape[-1]:
        spectrogram_pseudo = torch.nn.functional.pad(spectrogram_pseudo, (0, 1))

      for i in range(int(self.GLA_iteration)):
        # 2nd projection
        STFT_P2 = spectrogram_pseudo * \
          (STFT_P1 - tprev.mul_(self.momentum)).div(STFT_P1.abs().add(1e-12))

        tprev = STFT_P1

        # 1st projection
        STFT_P1 = torch.stft(torch.istft(STFT_P2,
                                       n_fft=self.n_fft,
                                       hop_length=self.hop,
                                       win_length=self.win,
                                       window=self.hann_window[str(STFT_P2.device)]),
                             n_fft=self.n_fft,
                             hop_length=self.hop,
                             win_length=self.win,
                             window=self.hann_window[str(STFT_P2.device)],
                             pad_mode='reflect',
                             normalized=True,
                             onesided=True,
                             return_complex=True)[:, :, :spectrogram_pseudo.shape[-1]]

      audio = torch.istft(STFT_P1,
                          n_fft=self.n_fft,
                          hop_length=self.hop,
                          win_length=self.win,
                          window=self.hann_window[str(STFT_P2.device)])

      audio = normalize(audio, p=float('inf'), dim=-1, eps=1e-12) * 0.95

    return audio
