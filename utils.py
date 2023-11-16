import os
import warnings
import yaml
import logging

import librosa as audio_lib
import scipy.io.wavfile as wf
import numpy as np
import soundfile
import random

MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps

config_keys = [
    "model", "spectrogram_reader", "dataloader", "train_scp_conf",
    "valid_scp_conf", "debug_scp_conf", "test_scp_conf"
]


def nfft(window_size):
    return int(2**np.ceil(int(np.log2(window_size))))


# return F x T or T x F
def stft(samps,
         frame_length=1024,
         frame_shift=256,
         center=False,
         window="hann",
         return_samps=False,
         apply_log=False,
         apply_pow=False,
         transpose=True):
    ## STFT
    # if num_mic != samps.shape[0]:
    #     raise TypeError("the number of input channel does not match to configuration yaml")
    stft_mat = audio_lib.stft(
        samps,
        nfft(frame_length),
        frame_shift,
        frame_length,
        window=window,
        center=center)
    if apply_pow:
        stft_mat = np.power(stft_mat, 2)
    if apply_log:
        stft_mat = np.log(np.maximum(stft_mat, EPSILON))
    if transpose:
        stft_mat = np.transpose(stft_mat)
    return stft_mat if not return_samps else (samps, stft_mat)


def istft(file,
          stft_mat,
          frame_length=1024,
          frame_shift=256,
          center=False,
          window="hann",
          transpose=True,
          norm=None,
          fs=16000,
          nsamps=None):
    if transpose:
        stft_mat = np.transpose(stft_mat)
    samps = audio_lib.istft(
        stft_mat,
        frame_shift,
        frame_length,
        window=window,
        center=center,
        length=nsamps)
    # renorm if needed
    if norm:
        samps_norm = np.linalg.norm(samps, np.inf)
        samps = samps * norm / samps_norm
    # same as MATLAB and kaldi
    samps_int16 = (samps * MAX_INT16).astype(np.int16)

    fdir = os.path.dirname(file)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    # NOTE: librosa 0.6.0 seems could not write non-float narray
    #       so use scipy.io.wavfile instead
    wf.write(file, fs, samps_int16)


def IPD(spec_ref, spec, cos_sin_opt=False):
    ipd = np.angle(spec) - np.angle(spec_ref)
    yr = np.cos(ipd)
    yi = np.sin(ipd)
    yrm = yr.mean(0, keepdims=True)
    yim = yi.mean(0, keepdims=True)
    
    if cos_sin_opt:
        return np.concatenate((yi - yim, yr - yrm), axis=1)
    else:
        return np.arctan2(yi - yim, yr - yrm)


def apply_cmvn(feats):
    feats = feats - feats.mean(0, keepdims=True)
    feats = feats / feats.std(0, keepdims=True)
    return feats


def parse_scps(scp_path):
    assert os.path.exists(scp_path)
    scp_dict = dict()
    with open(scp_path, 'r') as f:
        for scp in f:
            scp_tokens = scp.strip().split()
            if len(scp_tokens) != 2:
                raise RuntimeError(
                    "Error format of context \'{}\'".format(scp))
            key, addr = scp_tokens
            if key in scp_dict:
                raise ValueError("Duplicate key \'{}\' exists!".format(key))
            scp_dict[key] = addr
    return scp_dict


def filekey(path):
    fname = os.path.basename(path)
    if not fname:
        raise ValueError("{}(Is directory path?)".format(path))
    token = fname.split(".")
    if len(token) == 1:
        return token[0]
    else:
        return '.'.join(token[:-1])


def parse_yaml(yaml_conf):
    if not os.path.exists(yaml_conf):
        raise FileNotFoundError(
            "Could not find configure files...{}".format(yaml_conf))
    with open(yaml_conf, 'r') as f:
        config_dict = yaml.full_load(f)

    # for key in config_keys:
    #     if key not in config_dict:
    #         raise KeyError("Missing {} configs in yaml".format(key))
    batch_size = config_dict["dataloader"]["batch_size"]
    if batch_size <= 0:
        raise ValueError("Invalid batch_size: {}".format(batch_size))
    num_frames = config_dict["spectrogram_reader"]["frame_length"]
    num_bins = nfft(num_frames) // 2 + 1
    # if len(config_dict["train_scp_conf"]) != len(
    #         config_dict["valid_scp_conf"]):
    #     raise ValueError("Check configures in train_scp_conf/valid_scp_conf")
    # num_spks = 0
    # for key in config_dict["train_scp_conf"]:
    #     if key[:3] == "spk":
    #         num_spks += 1
    # for key in config_dict["test_scp_conf"]:
    #     if key[:3] == "spk":
    #         num_spks += 1
    # if num_spks != config_dict["model"]["num_spks"]:
    #     warnings.warn(
    #         "Number of speakers configured in trainer do not match *_scp_conf, "
    #         " correct to {}".format(num_spks))
    #     config_dict["model"]["num_spks"] = num_spks
    return num_bins, config_dict


def get_logger(
        name,
        format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(format_str)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
