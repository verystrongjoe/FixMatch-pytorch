import os
import logging
import math
import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torch.multiprocessing as mp
import wandb
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets.samplers import ImbalancedDatasetSampler
from tqdm import tqdm

from datasets.dataset import DATASET_GETTERS
from datasets.dataset import WM811K
from utils import AverageMeter, accuracy
from utils.common import get_args, de_interleave, interleave, save_checkpoint
from utils.common import set_seed, create_model, get_cosine_schedule_with_warmup
from datetime import datetime
import yaml
from argparse import Namespace
from copy import deepcopy
from models.resnet import ResNetBackbone
from models.vggnet import VggNetBackbone
from models.alexnet import AlexNetBackbone

from models.network_configs import RESNET_BACKBONE_CONFIGS, VGGNET_BACKBONE_CONFIGS


if __name__ == '__main__':

    args = Namespace()
    args.num_channel = 1
    args.decouple_input = False
    args.arch = 'resnet18'
    args.num_classes = 9
    args.local_rank = 0
    x = torch.rand((32, 1, 96, 96)).to(args.local_rank)
    # m = AdvancedCNN(args)

    m = ResNetBackbone(RESNET_BACKBONE_CONFIGS['18'], in_channels=1)
    m.to(args.local_rank)
    y = m(x)
    y = deepcopy(m)(x)
    print(y.shape)

    m = VggNetBackbone(VGGNET_BACKBONE_CONFIGS['16'], in_channels=1)
    m.to(args.local_rank)
    y = deepcopy(m)(x)
    print(y.shape)

    m = AlexNetBackbone('bn', in_channels=1)
    m.to(args.local_rank)
    y = deepcopy(m)(x)
    print(y.shape)


    print(m.named_parameters())

