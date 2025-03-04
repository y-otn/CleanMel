import torch
import librosa
import torch.nn as nn
from torch import Tensor
from typing import Optional
from torchaudio.transforms import Spectrogram
from models.io.norm import recursive_normalization
from torchaudio.transforms import Spectrogram, MelScale


class InputSTFT(nn.Module):
    """
    The STFT of the input signal of CleanMel (STFT coefficients);
    In online mode, the recursive normalization is used.
    """
    def __init__(
        self, 
        n_fft: int,
        n_win: int, 
        n_hop: int, 
        center: bool,
        normalize: bool,
        onesided: bool,
        online: bool = False):
        super().__init__()
        
        self.n_fft = n_fft
        self.win_length = n_win
        self.hop_length = n_hop
        self.online = online
        self.normalized = normalize
        self.stft=Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            normalized=self.normalized,
            center=center,
            onesided=onesided,
            power=None
        )
        
    def forward(self, x):
        x = self.stft(x)
        if self.online:
            x_mag = x.abs()
            x_norm = recursive_normalization(x_mag)
            x = x / x_norm.clamp(min=1e-8)
            x = torch.view_as_real(x)
            return x, x_norm
        else:
            x = torch.view_as_real(x)
            return x


class LibrosaMelScale(nn.Module):
    r"""Pytorch implementation of librosa mel scale; 
    To align with the Mel transformation in common ASRs;
    Noted taht some ASR systems use Kaldi backends, if so,
    you might need to define a Kaldi backbone to make sure the 
    model output is consistent with Kaldi-backend ASR system inputs.
    """
    __constants__ = ["n_mels", "sample_rate", "f_min", "f_max"]

    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_stft: int = 201,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> None:
        super(LibrosaMelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.norm = norm
        self.mel_scale = mel_scale

        if f_min > self.f_max:
            raise ValueError("Require f_min: {} <= f_max: {}".format(f_min, self.f_max))
        _mel_options = dict(
            sr=sample_rate,
            n_fft=(n_stft - 1) * 2,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
            htk=mel_scale=="htk",
            norm=norm
        )
        fb = torch.from_numpy(librosa.filters.mel(**_mel_options).T).float()
        self.register_buffer("fb", fb)
    
    def forward(self, specgram):
        mel_specgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)
        return mel_specgram


class TargetMel(nn.Module):
    """
    This class generates the enhancement TARGET mel spectrogram;
    """
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        n_win: int,
        n_hop: int,
        n_mels: int,
        f_min: int,
        f_max: int,
        power: int,
        center: bool,
        normalize: bool,
        onesided: bool,
        mel_norm: str | None,
        mel_scale: str,
        librosa_mel: bool = True,
        online: bool = False,
        ):
        super().__init__()
        # This implementation vs torchaudio.transforms.MelSpectrogram: Add librosa melscale
        # librosa melscale is numerically different from the torchaudio melscale (x_diff > 1e-5)
        self.n_fft = n_fft
        self.center = center
        self.n_hop = n_hop
        self.online = online
        self.stft = Spectrogram(
            n_fft=n_fft,
            win_length=n_win,
            hop_length=n_hop,
            power=None if online else power,
            normalized=normalize,
            center=center,
            onesided=onesided,
        )
        mel_method = LibrosaMelScale if librosa_mel else MelScale
        self.mel_scale = mel_method(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_stft=n_fft // 2 + 1,
            norm=mel_norm,
            mel_scale=mel_scale,
        )
        
    def forward(self, x: Tensor, x_norm=None, ilens=None):
        """
        ilens is used for ASR inference in enhancement model validation;
        the ASR validation is excluded in this version of the code,
        since it requires high GPU memory.
        """
        # Compute Spectrogram
        spectrogram = self.stft(x)
        if self.online:
            # apply norm to target spectrogram
            if x_norm is not None:
                spectrogram = spectrogram / (x_norm + 1e-8)
            spectrogram = spectrogram.abs().pow(2)  # to power spectrogram
        mel_specgram = self.mel_scale(spectrogram)
        
        if ilens is not None:
            if self.center:
                pad = self.n_fft // 2
                ilens = ilens + 2 * pad
            olens = torch.div(ilens - self.n_fft, self.n_hop, rounding_mode="trunc") + 1
            return mel_specgram, olens
        return mel_specgram