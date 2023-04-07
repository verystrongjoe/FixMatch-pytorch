import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torch.multiprocessing as mp
import torch.distributed as dist

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from datasets.samplers import ImbalancedDatasetSampler
from tqdm import tqdm
import wandb
# os.environ['WANDB_SILENT']="true"

from datasets.dataset import DATASET_GETTERS
from datasets.dataset import WM811K, WM811KEnsemble
from utils import AverageMeter, accuracy
from utils.common import get_args, save_checkpoint, set_seed, create_model, \
    get_cosine_schedule_with_warmup
from datetime import datetime
import argparse
import wandb
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)

alpha = 0.1
beta = 30
tau = 0.9
K = 2
dropout_rate = 0.5
limit_unlabled = 200000
percent_test_dataset = 0.2

# check (Table 4) 
# TODO: Milestones이라고 언급된건 어떤걸까?
batch_size = 256
lr = 0.003

epochs_1 = 125  # number of epochs for supervised learning (Section 4.2.)
epochs_2 = 150  # number of epochs for semi-supervised learning (Section 4.2.)

nm_optim = 'sgd'


def get_args():
    
    parser = argparse.ArgumentParser(description='PyTorch Ensemble Baseline Training')

    # project settings
    parser.add_argument('--project-name', required=True, type=str)
    parser.add_argument('--out', type=str, default='')

    # dataset
    parser.add_argument('--dataset', default='wm811k', type=str, choices=['wm811k', 'cifar10', 'cifar100'], help='dataset name')
    parser.add_argument('--proportion', type=float, help='percentage of labeled data used', default=0.05)
    parser.add_argument('--num_channel', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--size-xy', type=int, default=96)

    parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
    parser.add_argument('--decouple_input', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--limit-unlabled', type=int, default=20000)

    # model
    parser.add_argument('--arch', type=str, default='wideresnet',
                        choices=('resnet18', 'resnet50', 'vggnet', 'vggnet-bn', 'alexnet', 'alexnet-lrn', 'wideresnet', 'resnext'))

    # experiment
    parser.add_argument('--epochs', default=150, type=int, help='number of total steps to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int, help='train batchsize')
    parser.add_argument('--nm-optim', type=str, default='sgd', choices=('sgd', 'adamw'))
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float, help='warmup epochs (unlabeled data based)')  # 이게 어떤 의미가 있을라나??
    parser.add_argument('--wdecay', default=3e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True, help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate')
    parser.add_argument('--tau', default=0.3, type=float, help='tau')

    # fixmatch
    parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size') # todo : default 7
    parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')  # todo : default 1
    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int, help="random seed")

    args = parser.parse_args()
    args.local_rank = 0

    return args


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        n_classes = inputs.size(1)
        log_preds = F.log_softmax(inputs, dim=1)
        loss = -log_preds.sum(dim=1)
        smooth_loss = -log_preds.mean(dim=1)
        loss = loss.mean()
        smooth_loss = smooth_loss.mean()
        loss = (1.0 - self.smoothing) * loss + self.smoothing * smooth_loss
        return loss


def label_smoothing(label, alpha):
    n_classes = label.size(1)
    smooth_label = (1 - alpha) * label + (alpha / (n_classes - 1)) * torch.ones_like(label)
    return smooth_label

if __name__ == '__main__':

    ###################################################################################################################
    # 초기 설정
    ###################################################################################################################
    args = get_args()
    torch.cuda.set_device(args.local_rank)
    args.logger = logging.getLogger(__name__)

    train_supervised_dataset = WM811KEnsemble(args, mode='train', type='labeled')
    train_semi_dataset = WM811KEnsemble(args, mode='train', type='all')
    valid_dataset = WM811KEnsemble(args, mode='valid')
    test_dataset = WM811KEnsemble(args, mode='test')

    sueprvised_trainloader = DataLoader(dataset=train_supervised_dataset,
                      shuffle=True,
                      batch_size=args.batch_size,
                      pin_memory=True)

    semi_supervised_trainloader = DataLoader(
        train_semi_dataset,
        shuffle=True,
        batch_size=args.batch_size)

    valid_loader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=args.batch_size)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size)

    ###################################################################################################################
    # 지도학습
    ###################################################################################################################    
    #TODO: Table 2 ResNet-10, ResNet-18을 WaPIRL과 비교해야함
    models, optimizers, schedulers = []
    for i in range(K):
        m = create_model(args)
        o = optim.SGD(model.parameters(), lr=0.003)
        s = MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
        models.append(m)
        optimizers.append(o)
        schedulers.append(s)

    for k in range(K):
        model = models[k]
        optimizer = optimizers[k]
        scheduler = schedulers[k]

        for epoch in range(0, epochs_1):
            losses = AverageMeter()
            for batch_idx, (inputs_x, targets_x) in enumerate(sueprvised_trainloader):
                targets_x = targets_x.to(0)
                inputs_x = inputs_x.to(0)
                inputs_x = inputs_x.permute(0, 3, 1, 2).float()  # (c, h, w)
                logits = m(inputs_x)
                # criterion = LabelSmoothingLoss(smoothing=0.1)
                # loss = criterion(logits, targets_x.long())
                loss = F.cross_entropy(logits, targets_x.long())
                loss.backward()
                losses.update(loss.item())
                optimizer.step()
                scheduler.step()
                model.zero_grad()
        models.append(m)
        models.append(optimizers)
        models.append(schedulers)


    ###################################################################################################################
    # 준지도 학습
    ###################################################################################################################  
    optimizer = optim.SGD(model.parameters(), lr=0.003)
    scheduler = MultiStepLR(optimizer, milestones=[125], gamma=0.1)

    for epoch in range(epochs_2):
        # label + unlabled data 합쳐 mini batch
        for batch_idx, (inputs_x, targets_x) in enumerate(semi_supervised_trainloader):

            targets_x = targets_x.to(0)
            inputs_x  = inputs_x.to(0)
            inputs_x  = inputs_x.permute(0, 3, 1, 2).float()  # (b, c, h, w)
                
            flags_unlabeled = [targets_x=='_']
            flags_labeled = not flags_unlabeled

            k_logits_u = []
            k_logits_l = []

            # forward every inputs_x to every K models
            for k in range(K):
                logits = models[k](inputs_x) # (b, o)
                # in Section 3.2.2, p_bar_ic is the average of the probability values obainedfrom all the classfier
                k_logits_u.append(F.log_softmax(logits[flags_unlabeled], -1))
                k_logits_l.append(F.log_softmax(logits[flags_labeled], -1))  
            
            #################################################################################################
            # in case of unlabeled data
            #################################################################################################
            p_u = torch.mean(torch.stack(k_logits_u, axis=0), axis=0)  # k, b, o -> b, o
            q_u = p_u / torch.sum(p_u, dim=-1) # b, o -> b, o

            # get wc, uic in advance
            # calculate the class weight using equation 10    
            w = []  
            pseudo = torch.stack(k_logits_u, axis=0).gt(tau)  # k, b, c
            for c in range(args.num_classes):
                w.append( 1 / ( torch.sum(targets_x[flags_labeled] == c) + torch.sum(pseudo[:,:, c]) ) ) # (c, )
                        
            # TODO: 논문에선 u_i로 c가 빠져있는데 저자 답 없음
            # calculate instance weight by euqaiton 9
            u_ic = 1 / (1 + torch.exp(-beta*(q_u-tau)))  # b, o

            # calculate pseduo label by equation 7
            # y_u_hat_ic = torch.zeros_like(q_u)
            # y_u_hat_ic.scatter_(1, torch.argmax(q_u, dim=1, keepdim=True), 1)

            # smooth version of above like equation 4
            # y_u_s_hat_ic = label_smoothing(y_u_hat_ic)

            for k in range(K):
                L_k = 0.
                # y_l_hat_ic = torch.zeros_like(k_logits_l[k])
                # y_l_hat_ic.scatter_(1, torch.argmax(k_logits_l[k], dim=1, keepdim=True), 1)

                # Apply label smoothing both on the original labels of the labeled, and pseduo-alabels of the unlabeled samples using equation 4                
                ce = CrossEntropyLoss(weight=w, reduction='mean', label_smoothing=0.1)

                # calculate the loss by euantion 8 and update ensemble model    
                # Compute the cross-entropy loss for supervised
                L_k_super = ce(k_logits_l[k], targets_x[flags_labeled])
                # Compute the cross-entropy loss for semi-supervised
                L_k_semi = ce(k_logits_u[k], targets_x[flags_labeled]* u_ic)
                (L_k_super + L_k_semi).backward()
        