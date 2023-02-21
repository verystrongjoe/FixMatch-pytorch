import os
import random
import shutil
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import math
import argparse
#from models.advanced import AdvancedCNN
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics



def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')

    parser.add_argument('--server', type=str, choices=('ukjo-ubuntu', 'ukjo-window', 'richgo90',  'dgx', 'workstation1', 'workstation2'))
    parser.add_argument('--num_nodes', type=int, default=1, help='')
    parser.add_argument('--node_rank', type=int, default=0, help='')
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:3500', help='')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='')

    # project settings
    parser.add_argument('--project-name', required=True, type=str)

    # dataset
    parser.add_argument('--dataset', default='wm811k', type=str, choices=['wm811k', 'cifar10', 'cifar100'], help='dataset name')
    parser.add_argument('--proportion', type=float, help='percentage of labeled data used', default=1.)

    parser.add_argument('--num_channel', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--size-xy', type=int, default=32)

    parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
    parser.add_argument('--decouple_input', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--sweep', action='store_true')

    # model
    parser.add_argument('--arch', type=str, default='wideresnet',
                        choices=('resnet', 'vggnet', 'alexnet', 'wideresnet', 'resnext'))
    # parser.add_argument('--arch-config', default='18', type=str)

    # experiment
    parser.add_argument('--epochs', default=150, type=int, help='number of total steps to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int, help='train batchsize')
    parser.add_argument('--nm-optim', type=str, default='sgd', choices=('sgd', 'adamw'))
    parser.add_argument('--lr', '--learning-rate', default=0.04, type=float, help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float, help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=3e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True, help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate')
    parser.add_argument('--tau', default=0.3, type=float, help='tau')

    # fixmatch
    parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size') # todo : default 7
    parser.add_argument('--lambda-u', default=10, type=float, help='coefficient of unlabeled loss')  # todo : default 1
    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.7, type=float, help='pseudo label threshold')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int, help="random seed")

    # augmentations
    parser.add_argument('--n-weaks-combinations', type=int, default=2,
                        help="how many weak augmentations to make stronger augmentation")
    parser.add_argument("--aug_types", nargs='+', type=str, default=['crop', 'cutout', 'noise', 'rotate', 'shift'])
    parser.add_argument('--keep', action='store_true', help='keep-cutout or keep-paste')

    args = parser.parse_args()

    num_gpus_per_node = len(args.gpus)
    world_size = args.num_nodes * num_gpus_per_node
    args.world_size = world_size
    args.num_gpus_per_node = num_gpus_per_node
    
    if args.world_size == 1:
        args.local_rank = 0

    return args


def create_model(args, keep=False):
    if args.arch == 'wideresnet' or keep:
        import models.wideresnet as models
        args.model_depth = 28
        args.model_width = 2
        model = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=0,
                                        num_classes=args.num_classes)
    elif args.arch == 'resnext':
        args.model_cardinality = 4
        args.model_depth = 28
        args.model_width = 4
        import models.resnext as models
        model = models.build_resnext(cardinality=args.model_cardinality,
                                     depth=args.model_depth,
                                     width=args.model_width,
                                     num_classes=args.num_classes)
    return model



class MultiAUPRC(nn.Module):
    def __init__(self, num_classes: int):
        super(MultiAUPRC, self).__init__()
        self.num_classes = num_classes
        self.multi_prc = MulticlassPrecisionRecall(num_classes=num_classes)

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor):
        multi_prcs = self.multi_prc(
            pred=logits.softmax(dim=1),
            target=labels,
            sample_weight=None
        )
        avg_auprc = 0.
        for precision_, recall_, _ in multi_prcs:
            avg_auprc += auc(x=precision_, y=recall_, reorder=True)

        return [(avg_auprc / self.num_classes).clone().detach()]


class MultiAUROC(nn.Module):
    def __init__(self, num_classes: int):
        super(MultiAUROC, self).__init__()
        self.num_classes = num_classes
        self.multi_roc = MulticlassROC(num_classes=num_classes)

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor):
        multi_rocs = self.multi_roc(
            pred=logits.softmax(dim=1),
            target=labels,
            sample_weight=None
        )
        avg_auroc = 0.
        for fpr, tpr, _ in multi_rocs:
            avg_auroc += auc(x=fpr, y=tpr, reorder=True)

        return torch.Tensor([avg_auroc / self.num_classes])


class MultiAccuracy(nn.Module):
    def __init__(self, num_classes: int):
        super(MultiAccuracy, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor):

        assert logits.ndim == 2
        assert labels.ndim == 1
        assert len(logits) == len(labels)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct = torch.eq(preds, labels)

            return torch.mean(correct.float())


class TopKAccuracy(nn.Module):
    def __init__(self, num_classes: int, k: int, threshold: float = 0.):
        super(TopKAccuracy, self).__init__()
        self.num_classes = num_classes
        self.k = k
        self.threshold = threshold

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):

        assert logits.ndim == 2
        assert labels.ndim == 1
        assert len(logits) == len(labels)

        with torch.no_grad():
            topk_probs, topk_indices = torch.topk(F.softmax(logits, dim=1), self.k, dim=1)
            labels = labels.view(-1, 1).expand_as(topk_indices)                 # (B, k)
            correct = labels.eq(topk_indices) * (topk_probs >= self.threshold)  # (B, k)
            correct = correct.sum(dim=1).bool().float()                         # (B, ) & {0, 1}

            return torch.mean(correct)


class MultiPrecision(nn.Module):
    def __init__(self, num_classes: int, average='macro'):
        super(MultiPrecision, self).__init__()
        self.num_classes = num_classes
        assert average in ['macro', 'micro', 'weighted']
        self.average = average

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor):
        assert logits.ndim == 2
        assert labels.ndim == 1

        with torch.no_grad():
            if self.average == 'macro':
                return precision(
                    pred=nn.functional.softmax(logits, dim=1),
                    target=labels,
                    num_classes=self.num_classes,
                    reduction='elementwise_mean'
                )
            else:
                raise NotImplementedError


class MultiRecall(nn.Module):
    def __init__(self, num_classes: int, average='macro'):
        super(MultiRecall, self).__init__()
        self.num_classes = num_classes
        assert average in ['macro', 'micro', 'weighted']
        self.average = average

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor):
        assert logits.ndim == 2
        assert labels.ndim == 1

        with torch.no_grad():
            if self.average == 'macro':
                return recall(
                    pred=nn.functional.softmax(logits, dim=1),
                    target=labels,
                    num_classes=self.num_classes,
                    reduction='elementwise_mean',
                )
            else:
                raise NotImplementedError


class MultiF1Score(nn.Module):
    def __init__(self, num_classes: int, average: str = 'macro'):
        super(MultiF1Score, self).__init__()

        self.num_classes = num_classes

        assert average in ['macro', 'micro', 'weighted']
        self.average = average

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor):
        assert logits.ndim == 2
        assert labels.ndim == 1

        with torch.no_grad():
            if self.average == 'macro':
                f1_scores = torch.zeros(self.num_classes, device=logits.device)
                for c in range(self.num_classes):
                    pred = logits.argmax(dim=1) == c
                    true = labels == c
                    f1 = BinaryFBetaScore.macro_f_beta_score(pred, true, beta=1)
                    f1_scores[c] = f1
                return torch.mean(f1_scores)
            elif self.average == 'micro':
                raise NotImplementedError
            elif self.average == 'weighted':
                raise NotImplementedError
            else:
                raise NotImplementedError


class BinaryFBetaScore(nn.Module):
    def __init__(self, beta=1, threshold=.5, average='macro'):
        super(BinaryFBetaScore, self).__init__()
        self.beta = beta
        self.threshold = threshold
        self.average = average

    def forward(self, logit: torch.Tensor, label: torch.Tensor):
        assert logit.ndim == 1
        assert label.ndim == 1

        with torch.no_grad():
            pred = torch.sigmoid(logit)
            pred = pred > self.threshold   # boolean
            true = label > self.threshold  # boolean

            if self.average == 'macro':
                return self.macro_f_beta_score(pred, true, self.beta)
            elif self.average == 'micro':
                return self.micro_f_beta_score(pred, true, self.beta)
            elif self.average == 'weighted':
                return self.weighted_f_beta_score(pred, true, self.beta)
            else:
                raise NotImplementedError

    @staticmethod
    def macro_f_beta_score(pred: torch.Tensor, true: torch.Tensor, beta=1):

        assert true.ndim == 1
        assert pred.ndim == 1

        pred = pred.float()  # inputs could be boolean values
        true = true.float()  # inputs could be boolean values

        tp = (pred * true).sum().float()          # True positive
        _  = ((1-pred) * (1-true)).sum().float()  # True negative
        fp = ((pred) * (1-true)).sum().float()    # False positive
        fn = ((1-pred) * true).sum().float()      # False negative

        precision_ = tp / (tp + fp + 1e-7)
        recall_ = tp / (tp + fn + 1e-7)

        f_beta = (1 + beta**2) * precision_ * recall_ / (beta**2 * precision_ + recall_ + 1e-7)

        return f_beta

    @staticmethod
    def micro_f_beta_score(pred: torch.Tensor, true: torch.Tensor, beta=1):
        raise NotImplementedError

    @staticmethod
    def weighted_f_beta_score(pred: torch.Tensor, true: torch.Tensor, beta=1):
        raise NotImplementedError


class BinaryF1Score(BinaryFBetaScore):
    def __init__(self, threshold=.5, average='macro'):
        super(BinaryF1Score, self).__init__(beta=1, threshold=threshold, average=average)


if __name__ == '__main__':
    targets = torch.LongTensor([2, 2, 0, 2, 1, 1, 1])
    predictions = torch.FloatTensor(
        [
            [1, 2, 7],  # 2
            [1, 3, 7],  # 2
            [3, 9, 0],  # 1
            [1, 2, 3],  # 2
            [3, 7, 0],  # 1
            [8, 1, 1],  # 0
            [9, 1, 1],  # 0
        ]
    )

    # f1_function = MultiF1Score(num_classes=3, average='macro')
    # f2_function = MultiAUPRC(num_classes=3)
    # # f1_val = f1_function(logits=predictions, labels=targets)
    # # print(f1_val)
    # f2_val = f2_function(logits=predictions, labels=targets)
    
    auprc = torchmetrics.AveragePrecision(task="multiclass", num_classes=3, average='macro')
    f1score = torchmetrics.F1Score(task="multiclass", num_classes=3, average='macro')

    print(auprc(predictions, targets))
    print(f1score(predictions, targets))

