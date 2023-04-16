import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision.models as models

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
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import CrossEntropyLoss
from torchviz import make_dot


logger = logging.getLogger(__name__)

alpha = 0.1
beta = 30
tau = 0.9
dropout_rate = 0.5
use_supervised_pretrained = False

# check (Table 4) 
# TODO: Milestones이라고 언급된건 어떤걸까?
# lr = 0.003


#TODO: 원래데로 돌려놓자.
epochs_1 = 1  # 125  # number of epochs for supervised learning (Section 4.2.)
epochs_2 = 2  # 150  # number of epochs for semi-supervised learning (Section 4.2.)
# nm_optim = 'sgd'


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
    parser.add_argument('--limit-unlabled', type=int, default=200000)

    # model
    parser.add_argument('--arch', type=str, default='wideresnet',
                        choices=('resnet18', 'resnet50', 'vggnet', 'vggnet-bn', 'alexnet', 'alexnet-lrn', 'wideresnet', 'resnext'))

    # experiment
    parser.add_argument('--epochs', default=150, type=int, help='number of total steps to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=256, type=int, help='train batchsize')
    parser.add_argument('--nm-optim', type=str, default='sgd', choices=('sgd', 'adamw'))
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float, help='initial learning rate')
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
    parser.add_argument('--K', default=2, type=int, help='number of cnn models') # todo : default 2

    args = parser.parse_args()
    args.local_rank = 0

    print(args)

    return args



class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))



def custom_cross_entropy_loss(logits, labels, class_weights=None, smoothing=0.0, instance_weights=None):
    # apply label smoothing to labels
    num_classes = args.num_classes
    if smoothing > 0:
        smoothed_labels = (1 - smoothing) * F.one_hot(labels, num_classes=num_classes) + smoothing / num_classes
    else:
        smoothed_labels = F.one_hot(labels, num_classes=num_classes)

    if instance_weights is not None:
        # compute cross-entropy loss with class weights
        loss = F.cross_entropy(logits, smoothed_labels * instance_weights, weight=class_weights)
    else:
        loss = F.cross_entropy(logits, smoothed_labels, weight=class_weights)

    return loss



def evaluate(args, models, data_loader):
    total_preds = []
    total_reals = []

    # evlaute the model
    for batch_idx, (inputs_x, targets) in enumerate(data_loader):
        logits_k = []

        inputs_x = inputs_x.to(args.local_rank)
        targets = targets.to(args.local_rank)
        
        inputs_x = F.one_hot(inputs_x.long(), num_classes=3).squeeze()
        # calculate mean and standard deviation
        mean = torch.mean(inputs_x.float())
        std = torch.std(inputs_x.float())
        # normalize the tensor
        inputs_x = ((inputs_x.float() - mean) / std).permute(0,3,1,2)


        for k in range(args.K):
            outputs = train_models[k](inputs_x) # (b, c)
            logits_k.append(outputs)  # (k, b, c)
        
        logits = torch.mean(torch.stack(logits_k, axis=0), axis=0)  # (b, c)
        total_preds.append(torch.argmax(logits, dim=1).cpu().detach().numpy())
        total_reals.append(targets.cpu().detach().numpy())

    total_preds = np.concatenate(total_preds)
    total_reals = np.concatenate(total_reals)   
    final_f1 = f1_score(y_true=total_reals, y_pred=total_preds, average='macro')
    return final_f1

    

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
    train_models, optimizers, schedulers = [], [], []

    for k in range(args.K):
        # TODO: change the pretrained model
        # m = create_model(args).to(args.local_rank)
        m  = models.resnet18(pretrained=True)
        
        # Modify last layer to have a softmax output of size 9 and add a dropout layer
        num_ftrs = m.fc.in_features
        m.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, 9),
            torch.nn.LogSoftmax(dim=1)
        )
        
        m = m.to(args.local_rank)
        
        o = optim.SGD(m.parameters(), lr=0.003)
        s = MultiStepLR(o, milestones=[50, 100, 125], gamma=0.1)
        train_models.append(m)
        optimizers.append(o)
        schedulers.append(s)

    if not use_supervised_pretrained:
        for k in range(args.K):
            train_models[k].zero_grad()
            train_models[k].train()

            for epoch in range(0, epochs_1):
                losses = AverageMeter()
                for batch_idx, (inputs_x, targets_x) in enumerate(sueprvised_trainloader):
                    targets_x = targets_x.to(args.local_rank)
                    inputs_x = inputs_x.to(args.local_rank)
                    
                    inputs_x = F.one_hot(inputs_x.long(), num_classes=3).squeeze()
                    # calculate mean and standard deviation
                    mean = torch.mean(inputs_x.float())
                    std = torch.std(inputs_x.float())
                    # normalize the tensor
                    inputs_x = ((inputs_x.float() - mean) / std).permute(0,3,1,2)
                    
                    train_models[k](inputs_x)
                    logits = train_models[k](inputs_x)
                    # criterion = LabelSmoothingLoss(smoothing=0.1)
                    # loss = criterion(logits, targets_x.long())
                    loss = F.cross_entropy(logits, targets_x.long())
                    loss.backward()
                    losses.update(loss.item())
                    optimizers[k].step()
                    schedulers[k].step()
                    train_models[k].zero_grad()
                print(f"Epoch {epoch} Loss {losses.avg}")
            
        # save_checkpoint for models
        for k in range(args.K):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': train_models[k].state_dict(),
                'optimizer': optimizers[k].state_dict(),
                'scheduler': schedulers[k].state_dict(),
            }, False, checkpoint='checkpoints', filename='checkpoint_{}.pth.tar'.format(k))
    else:
        # load saved checkpoint for models 
        for k in range(args.K):
            state = torch.load('checkpoints/checkpoint_{}.pth.tar'.format(k), map_location='cpu')
            train_models[k].load_state_dict(state['state_dict'])
            train_models[k].zero_grad()
            train_models[k].train()

            optimizers[k].load_state_dict(state['optimizer'])
            schedulers[k].load_state_dict(state['scheduler'])    

    ###################################################################################################################
    # 준지도 학습
    ###################################################################################################################  
    f1_valid_best  = 0
    f1_test_best = 0 

    for epoch in range(epochs_1, epochs_2):
        losses_super = AverageMeter()
        losses_semi = AverageMeter()

        # label + unlabled data 합쳐 mini batch
        for batch_idx, (inputs_x, targets_x) in enumerate(semi_supervised_trainloader):

            targets_x = targets_x.to(0)
            inputs_x  = inputs_x.to(0)

            inputs_x = F.one_hot(inputs_x.long(), num_classes=3).squeeze()
            # calculate mean and standard deviation
            mean = torch.mean(inputs_x.float())
            std = torch.std(inputs_x.float())
            # normalize the tensor
            inputs_x = ((inputs_x.float() - mean) / std).permute(0,3,1,2)
    
            flags_unlabeled = targets_x==9
            flags_labeled = ~flags_unlabeled

            k_logits_u = []
            k_logits_l = []

            # forward every inputs_x to every K models
            for k in range(args.K):
                logits = train_models[k](inputs_x) # (b, o)
                # in Section 3.2.2, p_bar_ic is the average of the probability values obainedfrom all the classfier
                k_logits_u.append(F.log_softmax(logits[flags_unlabeled], -1))
                k_logits_l.append(F.log_softmax(logits[flags_labeled], -1))  
            
            #################################################################################################
            # in case of unlabeled data
            #################################################################################################
            p_u = torch.mean(torch.stack(k_logits_u, axis=0), axis=0)  # k, b, o -> b, o
            q_u = p_u / torch.unsqueeze(torch.sum(p_u, dim=-1), -1) # b, o -> b, o

            # get wc, uic in advance
            # calculate the class weight using equation 10    
            w = []  
            pseudo = torch.stack(k_logits_u, axis=0).gt(tau)  # k, b, c
            for c in range(args.num_classes):
                w.append( torch.nan_to_num(1 / ( torch.sum(targets_x[flags_labeled] == c) + torch.sum(pseudo[:,:, c]) ), nan=0., posinf=0.) ) # (c, )
            w = torch.stack(w, axis=0).squeeze()
                        
            # TODO: 논문에선 u_i로 c가 빠져있는데 수도 레이블에 대한 confidence 만 활용하는지 아님 전체를 활용하는지 모르겠음
            # calculate instance weight by euqaiton 9
            u_ic = 1 / (1 + torch.exp(-beta*(q_u-tau)))  # b, o

            # calculate pseduo label by equation 7
            # y_u_hat_ic = torch.zeros_like(q_u)
            # y_u_hat_ic.scatter_(1, torch.argmax(q_u, dim=1, keepdim=True), 1)

            # smooth version of above like equation 4
            # y_u_s_hat_ic = label_smoothing(y_u_hat_ic)


            L = 0
            for k in range(args.K):
                # y_l_hat_ic = torch.zeros_like(k_logits_l[k])
                # y_l_hat_ic.scatter_(1, torch.argmax(k_logits_l[k], dim=1, keepdim=True), 1)

                # Apply label smoothing both on the original labels of the labeled, and pseduo-alabels of the unlabeled samples using equation 4                
                # ce = CrossEntropyLoss(weight=w, reduction='mean', label_smoothing=0.1)
                # ce = CustomCrossEntropyLoss(weight=w, smoothing=0.1)
                # LabelSmoothingLoss

                # calculate the loss by euantion 8 and update ensemble model    
                # Compute the cross-entropy loss for supervised
                L_k_super = custom_cross_entropy_loss(k_logits_l[k], targets_x[flags_labeled].long(), w, smoothing=0.1)

                # Compute the cross-entropy loss for semi-supervised
                L_k_semi = custom_cross_entropy_loss(k_logits_u[k], torch.argmax(q_u, dim=-1).long(), w, smoothing=0.1, instance_weights=u_ic)

                L_k = L_k_super + L_k_semi
                L += L_k
                
                losses_super.update(L_k_super.item())
                losses_semi.update(L_k_semi.item())

            L.backward()
            for k in range(args.K):
                optimizers[k].step()
                schedulers[k].step()
                train_models[k].zero_grad()

            # print('Epoch: [{0}][{1}/{2}]\t' 'Loss {losses.val:.4f} ({losses.avg:.4f})\t'.format(epoch, batch_idx, len(semi_supervised_trainloader), loss=losses))   

        print(f"Epoch {epoch} Supervised Loss: {losses_super.avg}, Semi Loss: {losses_semi.avg}")
        
        # set model train mode
        for k in range(args.K):
            train_models[k].eval()

        # validation                
        f1_valid = evaluate(args, models, valid_loader)

        if f1_valid_best < f1_valid:
            f1_valid_best = f1_valid
            
            # test
            f1_test = evaluate(args, models, test_loader)

            if best_final_f1_test < f1_test:
                best_final_f1_test = f1_test

        print(f"Epoch {epoch} F1 Score: {f1_valid}", f"Best F1 Score: {f1_valid_best}, f1_test: {f1_test}, best_f1_test: {best_final_f1_test} ")
        wandb.log({"Epoch": epoch, "F1 Score": f1_valid, "Supervised Loss": losses_super.avg, "Semi Loss": losses_semi.avg})
        wandb.run.summary["f1_test"] = f1_test
        wandb.run.summary["f1_test_best"] = f1_test_best

