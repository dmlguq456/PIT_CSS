# -*- coding: utf-8 -*-

import warnings
from typing import Optional, Union
from stft_util import STFT, iSTFT
import torch as th
import numpy
from torch import Tensor
from torchaudio import functional as F
import librosa as audio_lib
import scipy.io.wavfile as wf
import scipy

class MVDR(th.nn.Module):
    def __init__(self,
                 ref_channel: int = 0,
                 diag_eps: float = 1.0e-15,
                 device=th.device('cpu'),
                 ):
        super().__init__()
        self.ref_channel = ref_channel
        self.device = device
        self.diag_eps = diag_eps
        self.mask_threshold = th.nn.Threshold(0.5,1.0e-4)
    def rtf_evd(self, psd_s: Tensor) -> Tensor:
        r"""Estimate the relative transfer function (RTF) or the steering vector by eigenvalue decomposition."""

        _, v = th.linalg.eigh(psd_s)  # v is sorted along with eigenvalues in ascending order
        rtf = v[..., -1]  # choose the eigenvector with max eigenvalue
        rtf = rtf / rtf[..., [self.ref_channel]]
        rtf = rtf * rtf.shape[-1]**0.5 / th.norm(rtf, dim=-1, keepdim=True)
        return rtf

    def rtf_pm(self, psd_s: Tensor) -> Tensor:
        r"""Estimate the relative transfer function (RTF) or the steering vector by power method."""
        rtf = psd_s[..., 0]
        rtf = rtf / rtf[..., [self.ref_channel]]
        rtf = rtf * rtf.shape[-1]**0.5 / th.norm(rtf, dim=-1, keepdim=True)
        return rtf


    def get_scm(self,
                 x: Tensor, # [... , ch , freq , time]
                 mask: Optional[Tensor] = None, # [... , freq , time]
                 ) -> Tensor: # [... , freq , ch , ch]

        """Compute weighted Spatial Covariance Matrix."""    

        if mask is not None:
            x = x * th.unsqueeze(th.clamp(mask,min=1.0e-4),-3)

        x = x - th.mean(x, dim=-1, keepdim=True)
        x = th.transpose(x, -3, -2)  # shape (freq, ch, time)

        # outer product:
        # (..., ch_1, time) x (..., ch_2, time) -> (..., time, ch_1, ch_2)    
        scm = th.einsum("...ct,...et->...tce", [x, x.conj()])
        scm = th.mean(scm, dim=-3)
        return scm

    def mvdr_weights_rtf(self, rtf: Tensor, nscm: Tensor) -> Tensor:
        # th.backends.cuda.preferred_linalg_library("cusolver")
        # nscm = self._tik_reg(nscm, reg=self.diag_eps)
        nscm = nscm + self.diag_eps*th.eye(nscm.size(-1),device=self.device)
        numerator = th.linalg.solve(nscm, th.unsqueeze(rtf,dim=-1))  # (..., freq, ch)
        numerator = th.squeeze(numerator,dim=-1)  # (..., freq, ch)
        denominator = th.einsum("...d,...d->...", [rtf.conj(), numerator])
        beamform_weights = numerator / (denominator.real.unsqueeze(-1))

        return beamform_weights

    def forward(self,
                x: Tensor, # [... , ch , freq , time]
                mask: Tensor, # [... , freq , time]
                ) -> Tensor:
        if not x.is_complex():
            raise TypeError(f"The type of input STFT must be ``torch.cfloat`` or ``torch.cdouble``. Found {x.dtype}.")

        mask = th.clamp(mask,max=1)
        # MLDR-SVE
        mask_t = self.mask_threshold(mask)
        tscm = self.get_scm(x, mask=mask_t)
        # steering_vector = self.rtf_pm(tscm)
        steering_vector = self.rtf_evd(tscm)

        # MVDR Beamforming
        nscm = self.get_scm(x, mask=1-mask)
        w_mvdr = self.mvdr_weights_rtf(steering_vector, nscm)

        y = th.einsum("...fc,...cft->...ft", [w_mvdr.conj(), x])
    
        return y


if __name__ == '__main__':

    MVDR_beam = MVDR()
    stft = STFT(512, 256,'hann',device="cpu")
    istft = iSTFT(512, 256, 'hann',device="cpu")
    x, _ = audio_lib.load('/home/nas/user/Uihyeop/etc/sample_data/utterance_14.wav', sr=None, mono=False)
    # x, _ = audio_lib.load('/home/nas/user/Uihyeop/etc/sample_data/chime_sample_caf.wav', sr=None, mono=False)
    mask = scipy.io.loadmat('/home/nas/user/Uihyeop/etc/sample_data/utterance_14_0.mat')
    mask = th.tensor(mask['mask']).to('cpu')
    mask = th.transpose(mask,-1,-2)
    X = stft(th.tensor(x).to('cpu'),cplx=True)
    X = th.complex(X[0],X[1])
    mask = mask[...,:-2]
    Y = MVDR_beam(X, mask=mask)
    Y = th.view_as_real(Y)
    Y = [Y[...,0], Y[...,1]]
    y = istft(Y[0], Y[1],cplx=True).cpu().data.numpy()

    wf.write('/home/nas/user/Uihyeop/etc/sample_data/MVDR_utterance_14_0.wav', 16000, y.transpose(1,0))
    
    
    
