import logging
import math
import numpy as np
from PIL import Image
from torchvision import datasets
from numpy import random
from datasets.transforms import WM811KTransform, TransformFixMatch, TransformFixMatchWafer, TransformFixMatchWaferLinearUCB
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import glob
import os
import pathlib
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import random

logger = logging.getLogger(__name__)


# Ensemble 논문을 위해 추가한 데이터셋
class WM811KEnsemble(Dataset):
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
    num_classes = len(idx2label) 

    def __init__(self, args, transform=None, mode='train', type='labeled'): 
        super(WM811KEnsemble, self).__init__()
        self.args = args
        self.transform = WM811KTransform(args, mode='test')


        assert mode in ['train', 'valid', 'test']
        assert type in ['labeled', 'all']

        assert not (mode in ['valid', 'test'] and type == 'all')
        
        ##############################################################################################################################
        # labeled data
        ##############################################################################################################################
        lableld_images = glob.glob(os.path.join(f'./data/wm811k/labeled/{mode}/', '**/*.png'), recursive=True)
        lableld_labels = [pathlib.PurePath(image).parent.name for image in lableld_images]
        lableld_targets = [self.label2idx[l] for l in lableld_labels] # Convert class label strings to integer target values
        
        if mode =='train' and self.args.proportion != 1.:
            assert self.args.proportion in [0.05, 0.1, 0.25, 0.5, 1.]
            X_train, X_test, y_train, y_test = train_test_split(
                lableld_images, lableld_targets, train_size=int(len(lableld_targets)*self.args.proportion), 
                stratify=lableld_targets, shuffle=True, random_state=1993 + self.args.seed)
            lableld_images = X_train
            lableld_targets = y_train
            print(f"we are using {self.args.proportion}% {len(lableld_targets)} labeled data samples..")            

        ##############################################################################################################################
        # unlabeled data
        ##############################################################################################################################
        if mode =='train' and type == 'all':
            unlabeled_images = glob.glob(os.path.join('./data/wm811k/unlabeled/train/', '**/*.png'), recursive=True)

            # 논문에서 언급한 데로 파라미터는 200,000 images만 뽑도록 되어 있음 (4.3. Experimental setting)
            if self.args.limit_unlabled != -1: 
                assert self.args.limit_unlabled > 0
                # 랜덤 샘플링
                # unlabeled_images = unlabeled_images[:self.args.limit_unlabled]
                unlabeled_images = random.sample(unlabeled_images, self.args.limit_unlabled)
                assert len(unlabeled_images) == self.args.limit_unlabled
                print(f"we are using {self.args.limit_unlabled} unlabeled data samples..")            

            unlabeled_labels = [pathlib.PurePath(image).parent.name for image in unlabeled_images]          # Parent directory names are class label strings
            unlableld_targets = [self.label2idx[l] for l in unlabeled_labels] # Convert class label strings to integer target values

            assert list(set(unlableld_targets))[0] == 9 # all labels must be '-'

        if mode == 'train' and type == 'all':
            self.targets = (lableld_targets + unlableld_targets)
            self.samples = list(zip(lableld_images + unlabeled_images, lableld_targets + unlableld_targets)) # Make (path, target) pairs
        else:
            self.targets = lableld_targets
            self.samples = list(zip(lableld_images, lableld_targets)) # Make (path, target) pairs
        
    def get_labels(self):
        return self.targets

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        x = self.load_image_cv2(path)
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_image_cv2(filepath: str):
        """Load image with cv2. Use with `albumentations`."""
        out = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # 2D; (H, W)
        return np.expand_dims(out, axis=2)                # 3D; (H, W, 1)


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

        images = sorted(glob.glob(os.path.join(root, '**/*.png'), recursive=True))  # Get paths to images
        labels = [pathlib.PurePath(image).parent.name for image in images]          # Parent directory names are class label strings

        # remove none!
        if self.args.exclude_none:
            none_idxes = (np.asarray(labels) == 'none')
            images = np.asarray(images)[~none_idxes]
            labels = np.asarray(labels)[~none_idxes]
            num_classes = num_classes - 1
            WM811K.num_classes = num_classes - 1
            assert self.args.num_classes == 8
            assert WM811K.num_classes == 8
        else:
            assert self.args.num_classes == 9
            assert WM811K.num_classes == 9

        targets = [self.label2idx[l] for l in labels]                                # Convert class label strings to integer target values
        
        if self.args.proportion != 1. and kwargs['phrase'] == 'train':
            X_train, X_test, y_train, y_test = train_test_split(
                images, targets, train_size=int(len(targets)*self.args.proportion), stratify=targets,
                shuffle=True,random_state=1993 + self.args.seed)
            images = X_train
            targets = y_train

        self.targets = targets
        self.samples = list(zip(images, targets)) # Make (path, target) pairs
        
    def get_labels(self):
        return self.targets

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        x = self.load_image_cv2(path)
        if self.args.decouple_input:
            x = self.decouple_mask(x)
        if self.transform is not None:
            x = self.transform(x)
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


class WM811KEvaluated(Dataset):
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
        super(WM811KEvaluated, self).__init__()

        self.root = root
        self.transform = transform
        self.args = kwargs.get('args', 0)

        images = sorted(glob.glob(os.path.join(root, '**/*.png'), recursive=True))  # Get paths to images
        labels = [pathlib.PurePath(image).parent.name for image in images]          # Parent directory names are class label strings

        # remove none!
        if self.args.exclude_none:
            none_idxes = (np.asarray(labels) == 'none')
            images = np.asarray(images)[~none_idxes]
            labels = np.asarray(labels)[~none_idxes]
            num_classes = num_classes - 1
            WM811K.num_classes = num_classes - 1
            assert self.args.num_classes == 8
            assert WM811K.num_classes == 8
        else:
            assert self.args.num_classes == 9
            assert WM811K.num_classes == 9

        targets = [self.label2idx[l] for l in labels]                                # Convert class label strings to integer target values

        if self.args.proportion != 1. and kwargs['phrase'] == 'train':
            X_train, X_test, y_train, y_test = train_test_split(
                images, targets, train_size=int(len(targets)*self.args.proportion), stratify=targets,
                shuffle=True,random_state=1993 + self.args.seed)
            images = X_train
            targets = y_train
        self.targets = targets

        if self.args.keep:
            saliency_maps = np.asarray([image.replace('.png', f'_saliency_{self.args.proportion if self.args.fix_keep_proportion < 0 else self.args.fix_keep_proportion}.npy') for image in images])
        else:
            saliency_maps = np.asarray(['' for image in images])

        self.samples = list(zip(images, targets, saliency_maps))  

    def __getitem__(self, idx):
        path, target, saliency_map = self.samples[idx]
        x = self.load_image_cv2(path)
        weak, strong = self.transform(x, np.load(saliency_map) if self.args.keep else None)
        return weak, strong, target, saliency_map
    
    
    # def __getitem__(self, idx):
    #     path, y = self.samples[idx]
    #     x = self.load_image_cv2(path)
    #     if self.args.decouple_input:
    #         x = self.decouple_mask(x)
    #     if self.transform is not None:
    #         x = self.transform(x)
    #     return x, y


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


class WM811KUnlabled(Dataset):
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
        super(WM811KUnlabled, self).__init__()

        self.root = root
        self.transform = transform
        self.args = kwargs.get('args', 0)

        images = sorted(glob.glob(os.path.join(root, '**/*.png'), recursive=True))  # Get paths to images
        
        if self.args.limit_unlabled != -1: 
            assert self.args.limit_unlabled > 0
            print(f"we are using {self.args.limit_unlabled} unlabeled data sampele")
            images = images[:self.args.limit_unlabled]
        
        if self.args.keep:
            saliency_maps = np.asarray([image.replace('.png', f'_saliency_{self.args.proportion if self.args.fix_keep_proportion < 0 else self.args.fix_keep_proportion}.npy') for image in images])
        else:
            saliency_maps = np.asarray(['' for image in images])

        self.samples = list(zip(images, saliency_maps))  # Make (path, target) pairs

    def __getitem__(self, idx):
        path, saliency_map = self.samples[idx]
        x = self.load_image_cv2(path)
        weak, strong, caption = self.transform(x, np.load(saliency_map) if self.args.keep else None)
        # caption 앞에 파일 경로 추가        
        return weak, strong, path+caption, saliency_map

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


class WM811KSaliency(Dataset):
    label2idx = {
        'center': 0,
        'donut': 1,
        'edge-loc': 2,
        'edge-ring': 3,
        'loc': 4,
        'random': 5,
        'scratch': 6,
        'near-full': 7,
        'none': 8,
        '-': 9,
    }

    idx2label = [k for k in label2idx.keys()]
    num_classes = len(idx2label) - 1  # exclude unlabeled (-)

    def __init__(self, root, transform=None, **kwargs):
        super(WM811KSaliency, self).__init__()

        self.root = root
        self.transform = transform
        self.args = kwargs.get('args', 0)
        self.samples = sorted(glob.glob(os.path.join(root, '**/*.png'), recursive=True))  # Get paths to images

    def __getitem__(self, idx):
        path = self.samples[idx]
        x = self.load_image_cv2(path)
        if self.transform is not None:
            x = self.transform(x)
        if self.args.decouple_input:
            x = self.decouple_mask(x)
        return x, path

    def __len__(self):
        return len(self.samples)

    def decouple_mask(x: torch.Tensor):
        """
        Decouple input with existence mask.
        Defect bins = 2, Normal bins = 1, Null bins = 0
        """
        m = x.gt(0).float()
        x = torch.clamp(x - 1, min=0., max=1.)
        return torch.cat([x, m], dim=0)

    @staticmethod
    def load_image_pil(filepath: str):
        """Load image with PIL. Use with `torchvision`."""
        return Image.open(filepath)

    @staticmethod
    def load_image_cv2(filepath: str):
        """Load image with cv2. Use with `albumentations`."""
        out = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # 2D; (H, W)
        return np.expand_dims(out, axis=2)                # 3D; (H, W, 1)


def get_wm811k(args, root):
    train_labeld_data_kwargs = {
        'phrase': 'train',
        'transform': WM811KTransform(args, mode='weak'),
        'args': args
    }

    if args.ucb:
        train_unlabeld_data_kwargs = {
            'phrase': 'train',
            'transform': TransformFixMatchWaferLinearUCB(args),
            'args': args,
        }
    else:
        train_unlabeld_data_kwargs = {
            'phrase': 'train',
            'transform': TransformFixMatchWafer(args),
            'args': args,
        }        
        
    
    test_data_kwargs = {
        'phrase': 'test',
        'transform': WM811KTransform(args, mode='test'),
        'args': args,
    }

    train_labeled_dataset = WM811K('./data/wm811k/labeled/train/', **train_labeld_data_kwargs)
    train_unlabeled_dataset = WM811KUnlabled('./data/wm811k/unlabeled/train/', **train_unlabeld_data_kwargs)
    valid_dataset = WM811K('./data/wm811k/labeled/valid/', **test_data_kwargs)  # it is same as test dataset.
    test_dataset = WM811K('./data/wm811k/labeled/test/', **test_data_kwargs)

    return train_labeled_dataset, train_unlabeled_dataset, valid_dataset, test_dataset


DATASET_GETTERS = {
                   'wm811k': get_wm811k
                   }


if __name__ == '__main__':
    train_labeled_dataset = WM811K('data/wm811k/labeled/train/', None)