import logging
import math
import os
import time
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
from torchviz import make_dot

logger = logging.getLogger(__name__)
use_supervised_pretrained = False

def get_args():
    # set default parameters based on the paper
    alpha = 0.1
    beta = 30
    tau = 0.9
    K = 2
    dropout_rate = 0.5
    limit_unlabled = 200000
    batch_size = 256
    lr = 0.003
    epochs_1 = 125  # 125  # number of epochs for supervised learning (Section 4.2.)
    epochs_2 = 150  # 150  # number of epochs for semi-supervised learning (Section 4.2.)

    nm_optim = 'sgd' # fixed

    parser = argparse.ArgumentParser(description='PyTorch Ensemble Baseline Training')

    # project settings
    parser.add_argument('--project-name', required=True, type=str)
    parser.add_argument('--proportion', type=float, help='percentage of labeled data used', default=0.05)
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--size-xy', type=int, default=96)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--sweep', action='store_true')
    
    # model
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=('resnet18', 'resnet50', 'vggnet', 'vggnet-bn', 'alexnet', 'alexnet-lrn', 'wideresnet', 'resnext'))

    # experiment
    parser.add_argument('--alpha', default=alpha, type=int, help='alpha')
    parser.add_argument('--beta', default=beta, type=int, help='beta')
    parser.add_argument('--tau', default=tau, type=float, help='tau')
    parser.add_argument('--K', default=K, type=int, help='K') 
    parser.add_argument('--dropout_rate', default=dropout_rate, type=float) 
    parser.add_argument('--limit-unlabled', type=int, default=limit_unlabled)
    parser.add_argument('--batch-size', default=batch_size, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=lr, type=float, help='initial learning rate')
    parser.add_argument('--epoch_supervised', default=epochs_1, type=int, help='number of total steps to run')
    parser.add_argument('--epoch_semi', default=epochs_2, type=int, help='number of total steps to run')
    parser.add_argument('--seed', default=None, type=int, help="random seed")

    args = parser.parse_args()
    args.local_rank = 0

    return args


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
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        logits_k = []
        for k in range(args.K):
            inputs3 = inputs.to(args.local_rank)
            targets = targets.to(args.local_rank)
            inputs3 = inputs3.permute(0, 3, 1, 2).float()  # (b, h, w, c) -> (b, c, h, w)
            outputs = models[k](inputs3) # (b, c)
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
    set_seed(args)
    torch.cuda.set_device(args.local_rank)
    args.logger = logging.getLogger(__name__)
    print(args)

    wandb.init(project=args.project_name, config=args)
    run_name = f"K_{args.K}_prop_{args.proportion}_arch_{args.arch}_seed{args.seed}"
    wandb.run.name = run_name

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
    models, optimizers_supervised, schedulers_supervised = [], [], []

    for k in range(args.K):
        m = create_model(args).to(args.local_rank)
        o = optim.SGD(m.parameters(), lr=0.003)
        s = MultiStepLR(o, milestones=[50, 100], gamma=0.1)
        models.append(m)
        optimizers_supervised.append(o)
        schedulers_supervised.append(s)

    if not use_supervised_pretrained:
        for k in range(args.K):
            models[k].zero_grad()
            models[k].train()

            for epoch in range(0, args.epoch_supervised):
                losses = AverageMeter()
                for batch_idx, (inputs_x, targets_x) in enumerate(sueprvised_trainloader):
                    targets_x = targets_x.to(args.local_rank)
                    inputs_x = inputs_x.to(args.local_rank)
                    inputs_x = inputs_x.permute(0, 3, 1, 2).float()  # (c, h, w)
                    logits = models[k](inputs_x)
                    loss = F.cross_entropy(logits, targets_x.long())
                    loss.backward()
                    losses.update(loss.item())
                    optimizers_supervised[k].step()
                    schedulers_supervised[k].step()
                    models[k].zero_grad()
                print(f"Epoch {epoch} Loss {losses.avg}")
            
        # save_checkpoint for models
        for k in range(args.K):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': models[k].state_dict(),
                'optimizer': optimizers_supervised[k].state_dict(),
                'scheduler': schedulers_supervised[k].state_dict(),
            }, False, checkpoint='checkpoints', filename='checkpoint_{}.pth.tar'.format(k))
    else:
        # load saved checkpoint for models 
        for k in range(args.K):
            state = torch.load('checkpoints/checkpoint_{}.pth.tar'.format(k), map_location='cpu')
            models[k].load_state_dict(state['state_dict'])
            models[k].zero_grad()
            models[k].train()

            optimizers_supervised[k].load_state_dict(state['optimizer'])
            schedulers_supervised[k].load_state_dict(state['scheduler'])    

    ###################################################################################################################
    # 준지도 학습
    ###################################################################################################################  
    optimizers_semi_supervised = []
    schedulers_semi_supervised = []
    
    f1_valid_best  = 0
    f1_test_best = 0 

    #TODO: 여기 semi쪽 타는거 optimizer는 공유해도 되는지 확인
    for k in range(args.K):
        s = MultiStepLR(optimizers_supervised[k], milestones=[125], gamma=0.1)
        schedulers_semi_supervised.append(s)
        o = optim.SGD(models[k].parameters(), lr=0.003)
        optimizers_semi_supervised.append(o)
        
    for epoch in range(args.epoch_semi):
        losses_super = AverageMeter()
        losses_semi = AverageMeter()

        # set model train mode
        for k in range(args.K):
            models[k].train()

        # label + unlabled data 합쳐 mini batch
        for batch_idx, (inputs_x, targets_x) in enumerate(semi_supervised_trainloader):

            targets_x = targets_x.to(0)
            inputs_x  = inputs_x.to(0)
            inputs_x  = inputs_x.permute(0, 3, 1, 2).float()  # (b, c, h, w)
                
            flags_unlabeled = targets_x==9
            flags_labeled = ~flags_unlabeled

            k_logits_u = []
            k_logits_l = []

            # forward every inputs_x to every K models
            for k in range(args.K):
                logits = models[k](inputs_x) # (b, o)
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
            pseudo = torch.stack(k_logits_u, axis=0).gt(args.tau)  # k, b, c
            for c in range(args.num_classes):
                w.append( torch.nan_to_num(1 / ( torch.sum(targets_x[flags_labeled] == c) + torch.sum(pseudo[:,:, c]) ), nan=0., posinf=0.) ) # (c, )
            w = torch.stack(w, axis=0).squeeze()
                        
            # TODO: 논문에선 u_i로 c가 빠져있는데 수도 레이블에 대한 confidence 만 활용하는지 아님 전체를 활용하는지 모르겠음
            # calculate instance weight by euqaiton 9
            u_ic = 1 / (1 + torch.exp(-args.beta*(q_u-args.tau)))  # b, o

            # calculate pseduo label by equation 7
            # smooth version of above like equation 4
            L_k_supers = []
            L_k_semis = []

            for k in range(args.K):
                # calculate the loss by euantion 8 and update ensemble model    
                # Compute the cross-entropy loss for supervised
                L_k_super = custom_cross_entropy_loss(k_logits_l[k], targets_x[flags_labeled].long(), w, smoothing=0.1)
                # Compute the cross-entropy loss for semi-supervised
                L_k_semi = custom_cross_entropy_loss(k_logits_u[k], torch.argmax(q_u, dim=-1).long(), w, smoothing=0.1, instance_weights=u_ic)

                L_k_semis.append(L_k_semi)
                L_k_supers.append(L_k_super)

                losses_super.update(L_k_super.item())
                losses_semi.update(L_k_semi.item())
                # print('Epoch: [{0}][{1}/{2}]\t' 'Loss {losses.val:.4f} ({losses.avg:.4f})\t'.format(epoch, batch_idx, len(semi_supervised_trainloader), loss=losses))   
                # print(f"Epoch {epoch} step {batch_idx}, k {k} Supervised Loss: {losses_super.avg}, Semi Loss: {losses_semi.avg}")

            L_k = 0
            for k in range(args.K):
                L_k += L_k_supers[k] + L_k_semis[k]
            L_k.backward()
            for k in range(args.K):
                schedulers_semi_supervised[k].step()
                optimizers_semi_supervised[k].step()
                models[k].zero_grad()
        
        print(f"Epoch {epoch} Supervised Loss: {losses_super.avg}, Semi Loss: {losses_semi.avg}")

        # set model train mode
        for k in range(args.K):
            models[k].eval()

        # validation                
        f1_valid = evaluate(args, models, valid_loader)

        if f1_valid_best < f1_valid:
            f1_valid_best = f1_valid
            
            # test
            f1_test = evaluate(args, models, test_loader)

            if f1_test_best < f1_test:
                f1_test_best = f1_test

        print(f"Epoch {epoch} F1 Score: {f1_valid}", f"Best F1 Score: {f1_valid_best}, f1_test: {f1_test}, best_f1_test: {f1_test_best} ")
        wandb.log({"Epoch": epoch, "F1 Score": f1_valid, "Supervised Loss": losses_super.avg, "Semi Loss": losses_semi.avg})
        wandb.run.summary["f1_test"] = f1_test
        wandb.run.summary["f1_test_best"] = f1_test_best


    
