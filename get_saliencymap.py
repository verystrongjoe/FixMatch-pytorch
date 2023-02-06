"""
CUDA_VISIABLE_DEVICES="MIG-5ceca708-e5aa-5675-b039-37a77bd4b6cf"
CUDA_VISIABLE_DEVICES="MIG-5ceca708-e5aa-5675-b039-37a77bd4b6cf"
CUDA_VISIABLE_DEVICES="MIG-5ceca708-e5aa-5675-b039-37a77bd4b6cf"
CUDA_VISIABLE_DEVICES="MIG-5ceca708-e5aa-5675-b039-37a77bd4b6cf"
CUDA_VISIABLE_DEVICES="MIG-5ceca708-e5aa-5675-b039-37a77bd4b6cf"
"""

import argparse
import logging
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"]="MIG-5ceca708-e5aa-5675-b039-37a77bd4b6cf"



import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

from datasets.dataset import WM811KSaliency
from datasets.transforms import WM811KTransform
from utils.common import create_model
from PIL import Image as im
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--num_gpu', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')
    parser.add_argument('--proportion', type=float, help='percentage of labeled data used', default=1.)
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--size-xy', type=int, default=32)

    # model
    parser.add_argument('--arch', type=str, default='wideresnet', choices=('resnet', 'vggnet', 'alexnet', 'wideresnet', 'resnext'))

    # experiment
    parser.add_argument('--epochs', default=150, type=int, help='number of total steps to run')

    parser.add_argument('--total-steps', default=318 * 150, type=int, help='number of total steps to run')
    parser.add_argument('--eval-step', default=318, type=int, help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=32, type=int, help='train batchsize')
    parser.add_argument('--nm-optim', type=str, default='sgd', choices=('sgd', 'adamw'))

    parser.add_argument('--lr', '--learning-rate', default=0.04, type=float, help='initial learning rate')

    parser.add_argument('--warmup', default=0, type=float, help='warmup epochs (unlabeled data based)')
    parser.add_argument('--use-ema', action='store_true', default=True, help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate')
    parser.add_argument('--decouple_input', action='store_true')

    parser.add_argument('--seed', default=None, type=int, help="random seed")
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    args.logger = logger

    return args


def main():
    args = get_args()
    device = torch.device('cuda', args.num_gpu)
    args.num_classes = 8
    args.local_rank = 0

    assert args.arch in ('wideresnet', 'resnext')
    if args.arch == 'wideresnet':
        args.model_depth = 28
        args.model_width = 2
    elif args.arch == 'resnext':
        args.model_cardinality = 4
        args.model_depth = 28
        args.model_width = 4

    checkpoint = torch.load(f'results/wm811k-supervised-{args.proportion}/model_best.pth.tar')
    model = create_model(args)
    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)
    model.load_state_dict(checkpoint['state_dict'])
    if args.use_ema:
        ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])  # todo : EMA 모델 사용하는게 맞는지 비교 필요
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    kwargs = {
        'transform': WM811KTransform(size=(args.size_xy, args.size_xy), mode='test'),
        'args': args
    }

    dataset = WM811KSaliency('./data/wm811k/unlabeled/train/', **kwargs)
    loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    for batch_idx, (inputs_x, paths_x) in tqdm(enumerate(loader)):
        inputs_x = inputs_x.to(device)

        # make 3 channels
        images_ = F.one_hot(inputs_x.long(), num_classes=3).squeeze().float()
        images_ = images_.permute(0, 3, 1, 2)  # (c, h, w)
        images_ = images_.to(device)
        images_.requires_grad = True
        preds = model(images_)  # forward model
        score, _ = torch.max(preds, 1)
        score.mean().backward()
        slc_, _ = torch.max(torch.abs(images_.grad), dim=1)
        b, h, w = slc_.shape
        slc_ = slc_.view(slc_.size(0), -1)
        slc_ -= slc_.min(1, keepdim=True)[0]
        slc_ /= slc_.max(1, keepdim=True)[0]
        slc_ = slc_.view(b, h, w)          # (128, 32, 32)

        for bi in range(len(slc_)):
            image = slc_[bi:bi+1, :, :].detach().cpu().numpy()
            path = paths_x[bi]

            folder, name = os.path.split(path)
            postfix = f"_saliency_{str(args.proportion)}"
            npy_name = name.replace(".png", postfix + ".npy")
            jpg_name = name.replace(".png", postfix + ".jpg")

            np.save(os.path.join(folder, npy_name), image)
            im.fromarray((image * 255).astype(np.uint8).squeeze()).save(os.path.join(folder, jpg_name))


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    main()
