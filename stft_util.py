# -*- coding: utf-8 -*-

import warnings
from typing import Optional, Union
import torch as th
import numpy
from torch import Tensor
from torchaudio import functional as F
import librosa as audio_lib
import scipy.io.wavfile as wf


class STFTBase(th.nn.Module):
    """
    Base layer for (i)STFT
    NOTE:
        1) Recommend sqrt_hann window with 2**N frame length, because it 
           could achieve perfect reconstruction after overlap-add
        2) Now haven't consider padding problems yet
    """
    def __init__(self,
                 frame_len,
                 frame_hop,
                 window="sqrt_hann",
                 device="cuda",
                 normalize=False):
        super(STFTBase, self).__init__()
        K = init_kernel(frame_len, frame_hop)
        self.K = th.nn.Parameter(K, requires_grad=False).to(device)
        self.stride = frame_hop
        self.window = window
        self.normalize = normalize
        self.num_bins = self.K.shape[0] // 2

    def extra_repr(self):
        return (f"window={self.window}, stride={self.stride}, " +
                f"kernel_size={self.K.shape[0]}x{self.K.shape[2]}, " +
                f"normalize={self.normalize}")
        

class iSTFT(STFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """
    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p, cplx=False, squeeze=False):
        """
        Accept phase & magnitude and output raw waveform
        args
            m, p: N x F x T
        return
            s: N x S
        """
        if p.dim() != m.dim() or p.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                p.dim()))
        # if F x T, reshape 1 x F x T
        if p.dim() == 2:
            p = th.unsqueeze(p, 0)
            m = th.unsqueeze(m, 0)
        if cplx:
            # N x 2F x T
            c = th.cat([m, p], dim=1)
        else:
            r = m * th.cos(p)
            i = m * th.sin(p)
            # N x 2F x T
            c = th.cat([r, i], dim=1)
        # N x 2F x T
        s = th.nn.functional.conv_transpose1d(c, self.K, stride=self.stride, padding=0)
        # N x S
        s = s.squeeze(1)
        if squeeze:
            s = th.squeeze(s)
        return s
    
class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """
    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, **kwargs)

    def forward(self, x, cplx=False):
        """
        Accept (single or multiple channel) raw waveform and output magnitude and phase
        args
            x: input signal, N x C x S or N x S
        return
            m: magnitude, N x C x F x T or N x F x T
            p: phase, N x C x F x T or N x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} expect 2D/3D tensor, but got {:d}D signal".format(
                    self.__name__, x.dim()))
        # if N x S, reshape N x 1 x S
        if x.dim() == 2:
            x = th.unsqueeze(x, 1)
            # N x 2F x T
            c = th.nn.functional.conv1d(x, self.K, stride=self.stride, padding=0)
            # N x F x T
            r, i = th.chunk(c, 2, dim=1)
            # if self.conjugate:
                # to match with science pipeline, we need to do conjugate
                # i = -i
        # else reshape NC x 1 x S
        else:
            N, C, S = x.shape
            x = x.reshape(N * C, 1, S)
            # x = x.view(N * C, 1, S)
            # NC x 2F x T
            c = th.nn.functional.conv1d(x, self.K, stride=self.stride, padding=0)
            # N x C x 2F x T
            c = c.reshape(N, C, -1, c.shape[-1])
            # c = c.view(N, C, -1, c.shape[-1])
            # N x C x F x T
            r, i = th.chunk(c, 2, dim=2)
            # if self.conjugate:
                # to match with science pipeline, we need to do conjugate
                # i = -i
        if cplx:
            return r, i
        m = (r**2 + i**2)**0.5
        p = th.atan2(i, r)
        return m, p

    

def init_kernel(frame_len, frame_hop):
    # FFT points
    N = frame_len
    # window
    W = th.hann_window(frame_len)
    if N//4 == frame_hop:
        const = (2/3)**0.5       
        W = const*W
        S =  0.5**(2/3) * (N * N / frame_hop)**0.5
    elif N//2 == frame_hop:
        W = W**0.5
        S = 0.5 * (N * N / frame_hop)**0.5

    # K = th.rfft(th.eye(N) / S, 1)[:frame_len]
    K = th.fft.rfft(th.eye(N) / S, dim=1)[:frame_len]
    K = th.stack((th.real(K), th.imag(K)),dim=2)
    # 2 x N/2+1 x F
    K = th.transpose(K, 0, 2) * W
    # N+2 x 1 x F
    K = th.reshape(K, (N + 2, 1, frame_len))
    return K

def IPD(spec_ref, spec, cos_sin_opt=False):
    ipd = th.angle(spec) - th.angle(spec_ref)
    yr = th.cos(ipd)
    yi = th.sin(ipd)
    yrm = yr.mean(1, keepdims=True)
    yim = yi.mean(1, keepdims=True)
    
    if cos_sin_opt:
        return th.cat([yi - yim, yr - yrm], dim=-1)
    else:
        return th.atan2(yi - yim, yr - yrm)
    
    
    