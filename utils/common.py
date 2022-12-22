import os
import random
import shutil
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import math
import argparse
from models.advanced import AdvancedCNN


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
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


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
    parser.add_argument('--num_gpu', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=30, help='number of workers')

    # dataset
    parser.add_argument('--dataset', default='wm811k', type=str, choices=['wm811k', 'cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000, help='number of labeled data')
    parser.add_argument('--num_channel', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
    parser.add_argument('--decouple_input', action='store_true')

    # model
    # parser.add_argument('--arch', type=str, default='resnet',
    #                     choices=('resnet', 'vggnet', 'alexnet', 'wideresnet', 'resnext'))
    # parser.add_argument('--arch-config', default='18', type=str)
    parser.add_argument('--arch', type=str, default='wideresnet',
                        choices=('resnet', 'vggnet', 'alexnet', 'wideresnet', 'resnext'))


    # experiment
    parser.add_argument('--total-steps', default=2 ** 20, type=int, help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int, help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float, help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float, help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True, help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
    parser.add_argument('--out', default='result', help='directory to output the result')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int, help="random seed")
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")

    # augmentations
    parser.add_argument('--n-weaks-combinations', type=int, default=2,
                        help="how many weak augmentations to make stronger augmentation")
    parser.add_argument("--aug_types", nargs='+', type=str, default=['crop', 'cutout', 'noise', 'rotate', 'shift'])

    args = parser.parse_args()
    return args


def create_model(args):
    if args.arch == 'wideresnet':
        import models.wideresnet as models
        model = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=0,
                                        num_classes=args.num_classes)
    elif args.arch == 'resnext':
        import models.resnext as models
        model = models.build_resnext(cardinality=args.model_cardinality,
                                     depth=args.model_depth,
                                     width=args.model_width,
                                     num_classes=args.num_classes)
    else:
        model = AdvancedCNN(args)

    return model
