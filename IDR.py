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

class IDR(th.nn.Module):

    def __init__(self,
                 ref_channel: int = 0,
                 diag_eps: float = 1e-12,
                 eps: float = 1e-8, # floor value for TVV
                 ):
        super().__init__()
        self.ref_channel = ref_channel
        self.diag_eps = diag_eps
        self.eps = eps

    def rtf_evd(self, psd_s: Tensor) -> Tensor:
        r"""Estimate the relative transfer function (RTF) or the steering vector by eigenvalue decomposition."""

        _, v = th.linalg.eigh(psd_s)  # v is sorted along with eigenvalues in ascending order
        rtf = v[..., -1]  # choose the eigenvector with max eigenvalue
        scale = rtf[..., self.ref_channel]
        return rtf / scale[..., None]

    def TVV_estimator(self,
                    x: Tensor, # [... , freq , time]
                    y: Optional[Tensor] = None, # [... , freq , time]
                    mask: Optional[Tensor] = None,
                    mode: str = "Mask"):
        if mode == 'Mask':
            print('hi! IDR!')
            return th.nn.functional.conv1d((th.abs(x*mask)**2)/2,th.ones(x.shape[-2],1,3)/3, padding='same', groups=x.shape[-2])**0.5 * th.abs(y)
            # return (th.abs(x*mask)**2 + th.abs(y)**2)/3


    def get_wscm(self,
                 x: Tensor, # [... , ch , freq , time]
                 TVV: Optional[Tensor] = None, # [... , freq , time]
                 mask: Optional[Tensor] = None, # [... , freq , time]
                 normalized: bool = False
                 ) -> Tensor: # [... , freq , ch , ch]

        """Compute weighted Spatial Covariance Matrix.
            if TVV is None, simlple SCM is returned with TVV set to unit.
        """    

        if mask is not None:
            x = x * th.clamp(th.unsqueeze(mask,-3), min=1.0e-10)
        x = x.transpose(-3, -2)  # shape (freq, channel, time)

        # outer product:
        # (..., ch_1, time) x (..., ch_2, time) -> (..., time, ch_1, ch_2)    
        scm = th.einsum("...ct,...et->...tce", [x, x.conj()])
            
        if TVV is not None:   
            wscm = scm / th.clamp(TVV[..., None, None],min=self.eps)
        else:
            wscm = scm

        if normalized:
            wscm = wscm.sum(dim=-3) / (1/th.clamp(TVV, min=self.eps)).sum(dim=-1)[..., None, None]
        else:
            wscm = wscm.mean(dim=-3) # time-averaging to get time-invariant WSCM
        return wscm

    def _tik_reg(self, mat: th.Tensor, reg: float = 1e-12, eps: float = 1e-12) -> th.Tensor:
        """Perform Tikhonov regularization (only modifying real part).
            reg (float, optional): Regularization factor. (Default: 1e-8)
            eps (float, optional): Value to avoid the correlation matrix is all-zero. (Default: ``1e-8``)
         """
        # Add eps
        C = mat.size(-1)
        eye = th.eye(C, dtype=mat.dtype, device=mat.device)
        epsilon = th.diagonal(mat, 0, dim1=-1, dim2=-2).sum(dim=-1).real[..., None, None] * reg
        # in case that correlation_matrix is all-zero
        epsilon = epsilon + eps
        mat = mat + epsilon * eye[..., :, :]
        return mat

    def mvdr_weights_rtf(self, rtf: Tensor, wscm: Tensor) -> Tensor:

        wscm = self._tik_reg(wscm, reg=self.diag_eps)
        numerator = th.linalg.solve(wscm, rtf.unsqueeze(-1)).squeeze(-1)  # (..., freq, ch)
        denominator = th.einsum("...d,...d->...", [rtf.conj(), numerator])
        beamform_weights = numerator / (denominator.real.unsqueeze(-1))

        return beamform_weights

    def forward(self,
                x: Tensor, # [... , ch , freq , time]
                y: Optional[Tensor] = None, # [... , freq , time]
                mask: Optional[Tensor] = None, # [... , freq , time]
                TVV_estimation_mode: str = 'MLE'
                ) -> Tensor:
        if not x.is_complex():
            raise TypeError(f"The type of input STFT must be ``torch.cfloat`` or ``torch.cdouble``. Found {x.dtype}.")
        TVV = self.TVV_estimator(x[self.ref_channel], y=y, mask=mask, mode=TVV_estimation_mode)

        # MLDR-SVE
        input_scm_masked = self.get_wscm(X, mask=mask, normalized=False)
        input_wscm_masked = self.get_wscm(x, TVV=TVV, mask=mask, normalized=True)
        tscm = input_scm_masked - input_wscm_masked
        steering_vector = self.rtf_evd(tscm)

        # MLDR Beamforming
        wscm = self.get_wscm(x, TVV=TVV, normalized=False)
        w_mldr = self.mvdr_weights_rtf(steering_vector, wscm)

        y = th.einsum("...fc,...cft->...ft", [w_mldr.conj(), x])
    
        return y


if __name__ == '__main__':

    IDR_beam = IDR()
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
    Y = X[...,0,:,:]
    for i in range(5):
        Y = IDR_beam(X, y=Y, mask=mask, TVV_estimation_mode='Mask')
    Y = th.view_as_real(Y)
    Y = [Y[...,0], Y[...,1]]
    y = istft(Y[0], Y[1],cplx=True).cpu().data.numpy()

    wf.write('/home/nas/user/Uihyeop/etc/sample_data/IDR_utterance_14_0.wav', 16000, y.transpose(1,0))
    
    
    
