"""
1. 레이블 데이터에 대한 augmentation 진행
2. 그리드 서치를 통해서 가장 적합한 magnitude를 찾아냄
3. pretrained 된 모델 기준하여 populated dataset을 만드는 정책을 찾는 문제
"""
import os
import logging
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import wandb
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets.samplers import ImbalancedDatasetSampler
from tqdm import tqdm
from datasets.dataset import DATASET_GETTERS
from datasets.dataset import WM811KEvaluated
from utils import AverageMeter, accuracy
from utils.common import get_args, de_interleave, interleave, save_checkpoint
from utils.common import set_seed, create_model, get_cosine_schedule_with_warmup
from datasets.transforms import WM811KTransformOnlyOne, TransformFixMatchWaferEval
from datetime import datetime
import yaml
from argparse import Namespace
from PIL import Image
import collections
import pandas as pd
import argparse
import wandb


if __name__ == '__main__':
    args = Namespace()
    args.local_rank = 0
    args.proportion = 1.0
    args.batch_size = 1024
    args.num_workers = 1
    args.arch = 'resnet18'
    args.size_xy =  96
    args.keep = False
    args.exclude_none = False
    args.num_classes = 9

    checkpoint = torch.load(f'results/wm811k-supervised-{args.proportion}/model_best.pth.tar')
    model = create_model(args)
    model.to(args.local_rank)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    

    for mode in ['cutout', 'noise', 'shift']: # 'crop', 'cutout', 'noise', 'shift', 'rotate'
        for magnitude in np.arange(0.1, 1.1, 0.1): 

            assert wandb.run is None
            with wandb.init(project="find_magnitude") as run:
                kwargs = {
                    'phrase': 'train',
                    'transform': TransformFixMatchWaferEval(args, mode, magnitude),
                    'args': args
                }
                eval_dataset = WM811KEvaluated('./data/wm811k/labeled/train/', **kwargs)
                trainloader = DataLoader(dataset=eval_dataset,
                        batch_size=args.batch_size,
                        sampler=SequentialSampler(eval_dataset),
                        num_workers=args.num_workers,
                        pin_memory=True)
                
                acc_cnt = 0
                total =  len(eval_dataset)

                for batch_idx, (weak, strong, target, saliency_map) in enumerate(trainloader):
                    weak = weak.to(args.local_rank)
                    strong = strong.to(args.local_rank)
                    target = target.to(args.local_rank)

                    weak = weak.permute(0, 3, 1, 2).float() 
                    strong = strong.permute(0, 3, 1, 2).float() 

                    weak_output = model(weak)
                    strong_output = model(strong)

                    cnt = (torch.argmax(weak_output, dim=1).cpu().detach().numpy() == torch.argmax(strong_output, dim=1).cpu().detach().numpy()).sum()
                    acc_cnt += cnt

                print(f"{mode}_{magnitude}  : {acc_cnt}/{total}({acc_cnt/total}%)")
                
                wandb.run.summary["mode"] = mode  
                wandb.run.summary["magnitude"] = magnitude
                wandb.run.summary["acc_cnt"] = acc_cnt 
                wandb.run.summary["total"] = total
                wandb.run.summary["percent"] = "{:.2f}".format(acc_cnt/total * 100)


