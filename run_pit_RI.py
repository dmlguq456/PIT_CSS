import argparse
import os
import torch as th
from torch.nn.utils.rnn import pack_sequence, pad_sequence

from trainer_RI import PITrainer_RI
from tester_RI import PITester_RI
from trainer_RI_NBF import PITrainer_RI_NBF
from dataset_RI import SpectrogramReader, Datasets

from model.DPMCN_v15 import DPMCN_v15
from model.DPMCN_v15_NBF import DPMCN_v15_NBF

from utils import nfft, parse_yaml, get_logger
from torch.utils.data import DataLoader

import scipy.signal as ss
import random
import numpy as np


logger = get_logger(__name__)


def _collate(egs):
    """
        Transform utterance index into a minbatch

        Arguments:
            index: a list type [{},{},{}]

        Returns:
            input_sizes: a tensor correspond to utterance length
            input_feats: packed sequence to feed networks
            source_attr/target_attr: dictionary contains spectrogram/phase needed in loss computation
        """
    num_spks = 2 #you need to set this paramater by yourself

    if type(egs) is not list:
        raise ValueError("Unsupported index type({})".format(type(egs)))

    def prepare_target_rir(dict_lsit, index):
        return pad_sequence([th.tensor(d["ref_rir"][index], dtype=th.float32)  for d in dict_lsit], batch_first=True)

    def prepare_target(dict_lsit, index):
        return pad_sequence([th.tensor(d["ref"][index], dtype=th.float32) for d in dict_lsit], batch_first=True)

    dict_list = sorted([eg for eg in egs],
                        key=lambda x: x['num_sample'], reverse=True)
    mixture = pad_sequence([th.tensor(d['mix'], dtype=th.float32) for d in dict_list], batch_first=True)
    target_rir = [prepare_target_rir(dict_list, index) for index in range(num_spks)]
    target = [prepare_target(dict_list, index) for index in range(num_spks)]
    noise = pad_sequence([th.tensor(d['noise'], dtype=th.float32) for d in dict_list], batch_first=True)
    input_sizes = th.tensor([d['num_sample'] for d in dict_list], dtype=th.float32)

    
    return input_sizes, mixture, target_rir, target, noise


def rir_load(rir_dir, RT_list):
    h = {}
    for room in ['medium', 'small']:
        h[room] = []
        for idx, radius in enumerate([0.04, 0.0425, 0.045]):
        # for idx, radius in enumerate([0.04, 0.0425, 0.045, 0.0475, 0.05]):
            h[room].append([])
            for RT in RT_list:
                h[room][idx].append(np.load(rir_dir+'/'+room+'/mic_deg_0_center_1.8_1.8_mic_r_'+str(radius)+'/target_RT_'+str(RT)+'.npy'))
    return h



def uttloader(scp_config, noise_dir, reader_kwargs, loader_kwargs, rir_list, num_mics, loss, rir_mode=0, train=True):
    scp_config_mix = [scp_config[spk_key] for spk_key in scp_config if spk_key[:3] == 'spk']

    mix_reader = SpectrogramReader(scp_config_mix, rir_list, noise_dir, rir_mode, train, num_mics, loader_kwargs['max_frame'], loss, **reader_kwargs)
    dataset = Datasets(mix_reader)
    # modify shuffle status
    loader_kwargs["shuffle"] = train
    utt_loader = DataLoader(dataset, batch_size=loader_kwargs['batch_size'],shuffle=loader_kwargs['shuffle'],
                            num_workers=loader_kwargs['num_workers'], sampler=None,drop_last=True,
                            collate_fn=_collate, pin_memory=True)
    return utt_loader


def train(args):
    gpuid = tuple(map(int, args.gpus.split(',')))
    logger.info("Start training in {} model")
    num_bins, config_dict = parse_yaml(args.config)
    reader_conf = config_dict["spectrogram_reader"]
    loader_conf = config_dict["dataloader"]
    dcnnet_conf = config_dict["model"]
    rir_conf = config_dict["rir_conf"]

    batch_size = loader_conf["batch_size"]
    logger.info(
        "Training in {}".format("per utterance" if batch_size == 1 else
                                '{} utterance per batch'.format(batch_size)))
    ## RIR generation
    RT_list = rir_conf["RT_list"]
    h = rir_load(rir_conf["rir_dir"], RT_list)
    logger.info("RIR fiter for RT = {} loading Done".format(RT_list))

    loader_conf["crm"] = config_dict['crm']

<<<<<<< HEAD
    if args.test_mode > 0:
=======
    if args.test_mode:
>>>>>>> 3fd70388add25ea7793ba02455bbcfe6e9837fb7
        test_loader = uttloader(
            config_dict["valid_scp_conf"],
            config_dict["noise_dir"],
            reader_conf,
            loader_conf,
            [h, RT_list],
            dcnnet_conf["num_mics"],
            config_dict["trainer"]["loss"],
<<<<<<< HEAD
            rir_mode=args.test_mode, 
=======
            rir_mode=1, 
>>>>>>> 3fd70388add25ea7793ba02455bbcfe6e9837fb7
            train=False)    
    else:
        train_loader = uttloader(
            config_dict["train_scp_conf"],
            config_dict["noise_dir"],
            reader_conf,
            loader_conf,
            [h, RT_list],
            dcnnet_conf["num_mics"],
            config_dict["trainer"]["loss"],
            rir_mode=0,
            train=True)
        valid_loader = uttloader(
            config_dict["valid_scp_conf"],
            config_dict["noise_dir"],
            reader_conf,
            loader_conf,
            [h, RT_list],
            dcnnet_conf["num_mics"],
            config_dict["trainer"]["loss"],
            rir_mode=0,
            train=False)
    checkpoint = config_dict["trainer"]["checkpoint"]
    logger.info("Training for {} epoches -> {}...".format(
        args.num_epoches, "default checkpoint"
        if checkpoint is None else checkpoint))
    dcnnet_conf["IPD_sincos"] = config_dict["trainer"]['IPD_sincos']
    dcnnet_conf["crm"] = config_dict['crm']
    config_dict["trainer"]["crm"] = config_dict['crm']
    log_dir = config_dict["trainer"]["checkpoint"] + '/log'

    if config_dict["model_type"] == "DPMCN_v1":
        nnet = DPMCN_v1(**dcnnet_conf)
    elif config_dict["model_type"] == "DPMCN_v15":
        nnet = DPMCN_v15(**dcnnet_conf)
    elif config_dict["model_type"] == "DPMCN_v15_NBF":
        nnet = DPMCN_v15_NBF(**dcnnet_conf)

    if "NBF" in config_dict["model_type"]:
        trainer = PITrainer_RI_NBF(nnet, reader_conf, **config_dict["trainer"], gpuid=gpuid)
    else:
        if args.test_mode:
            tester = PITester_RI(nnet, reader_conf, **config_dict["trainer"], gpuid=gpuid)
            tester.run(test_loader, num_epoches=args.num_epoches, log_dir=log_dir)
        else:
            trainer = PITrainer_RI(nnet, reader_conf, **config_dict["trainer"], gpuid=gpuid)
            trainer.run(train_loader, valid_loader, num_epoches=args.num_epoches, log_dir=log_dir)
    # trainer = PITrainer(nnet, reader_conf, **config_dict["trainer"], gpuid=gpuid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Command to start PIT training, configured by .yaml files")
    parser.add_argument(
        "--flags",
        type=str,
        default="",
        help="This option is used to show what this command is runing for")
    parser.add_argument(
        "--config",
        type=str,
        default="train.yaml",
        dest="config",
        help="Location of .yaml configure files for training")
    parser.add_argument(
        "--test-mode",
<<<<<<< HEAD
        type=str,
        default=0,
        dest="test_mode",
        help="If non_negative, start testing")
=======
        default=False,
        action="store_true",
        dest="test_mode",
        help="If true, start testing")
>>>>>>> 3fd70388add25ea7793ba02455bbcfe6e9837fb7
    parser.add_argument(
        "--num-epoches",
        type=int,
        default=150,
        dest="num_epoches",
        help="Number of epoches to train")
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        # default="0,1,2,3,4,5,6,7",
        help="Training on which GPUs "
        "(one or more, egs: 0, \"0,1\")")
    args = parser.parse_args()
    train(args)
