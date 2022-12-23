import logging
import math
import numpy as np
from PIL import Image
from torchvision import datasets
from numpy import random
from datasets.transforms import WM811KTransform, TransformFixMatch, TransformFixMatchWafer
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import glob
import os
import pathlib
import cv2
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class WM811K(Dataset):
    label2idx = {
        'center'    : 0,
        'donut'     : 1,
        'edge-loc'  : 2,
        'edge-ring' : 3,
        'loc'       : 4,
        'random'    : 5,
        'scratch'   : 6,
        'near-full' : 7,
        'none'      : 8,
        '-'         : 9,
    }
    idx2label = [k for k in label2idx.keys()]
    num_classes = len(idx2label) - 1  # exclude unlabeled (-)

    def __init__(self, root, transform=None, **kwargs):
        super(WM811K, self).__init__()

        self.root = root
        self.transform = transform
        self.args = kwargs.get('args', 0)

        images  = sorted(glob.glob(os.path.join(root, '**/*.png'), recursive=True))  # Get paths to images
        labels  = [pathlib.PurePath(image).parent.name for image in images]          # Parent directory names are class label strings
        targets = [self.label2idx[l] for l in labels]                                # Convert class label strings to integer target values
        
        if self.args.proportion != 1.:  
            X_train, X_test, y_train, y_test = train_test_split(
                images, targets, train_size=len(targets)*self.args.proportion, stratify=targets,
                shuffle=True,random_state=1993 + self.args.seed)
            images = X_train
            targets = y_train
            self.args.logger.info(f'It uses only {len(targets)} samples of train set because args.proportion is {self.args.proportion}')
        else:
            self.args.logger.info(f'It uses 100% {len(targets)} samples of train set because args.proportion is 1.')

        self.targets = targets
        self.samples = list(zip(images, targets)) # Make (path, target) pairs
        
    def get_labels(self):
        return self.targets


    def __getitem__(self, idx):
        path, y = self.samples[idx]
        x = self.load_image_cv2(path)

        if self.transform is not None:
            x = self.transform(x)

        if self.args.decouple_input:
            x = self.args.decouple_mask(x)

        return x, y

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_image_pil(filepath: str):
        """Load image with PIL. Use with `torchvision`."""
        return Image.open(filepath)

    @staticmethod
    def load_image_cv2(filepath: str):
        """Load image with cv2. Use with `albumentations`."""
        out = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # 2D; (H, W)
        return np.expand_dims(out, axis=2)                # 3D; (H, W, 1)

    @staticmethod
    def decouple_mask(x: torch.Tensor):
        """
        Decouple input with existence mask.
        Defect bins = 2, Normal bins = 1, Null bins = 0
        """
        m = x.gt(0).float()
        x = torch.clamp(x - 1, min=0., max=1.)
        return torch.cat([x, m], dim=0)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_wm811k(args, root):
    train_labeld_data_kwargs = {
        'transform': WM811KTransform(mode='weak'),
        'args': args
    }
    train_unlabeld_data_kwargs = {
        'transform': TransformFixMatchWafer(args),
        'args': args,
    }
    test_data_kwargs = {
        'transform': WM811KTransform(mode='test'),
        'args': args,
    }

    train_labeled_dataset = WM811K('./data/wm811k/labeled/train/', **train_labeld_data_kwargs)
    train_unlabeled_dataset = WM811K('./data/wm811k/unlabeled/train/', **train_unlabeld_data_kwargs)
    test_dataset = WM811K('./data/wm811k/unlabeled/test/', **test_data_kwargs)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'wm811k': get_wm811k
                   }
