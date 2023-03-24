import torch
import numpy as np
from numpy import random
from torchvision import transforms
from .randaugment import RandAugmentMC
from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.transforms_interface import ImageOnlyTransform
from torch.distributions import Bernoulli
import albumentations as A
import cv2
from utils.common import create_model
from typing import Any, Dict, Tuple, Union
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
import re


class ToWBM(ImageOnlyTransform):
    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super(ToWBM, self).__init__(always_apply, p)

    @property
    def targets(self) -> Dict[str, Callable]:
        return {"image": self.apply}

    def apply(self, img: np.ndarray, **kwargs):  # pylint: disable=unused-argument
        if img.ndim == 2:
            img = img[:, :, None]
        if np.isin(np.unique(img), [0., 127., 255.]).all():
            img = img / 255.
            return np.ceil(img * 2)
        else:
            return img

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}


class MaskedBernoulliNoise(ImageOnlyTransform):
    def __init__(self, noise: float, always_apply: bool = False, p: float = 1.0):
        super(MaskedBernoulliNoise, self).__init__(always_apply, p)
        self.noise = np.random.choice(list(np.arange(0, noise, 0.01)), 1)[0]
        self.min_ = 0
        self.max_ = 1
        self.bernoulli = Bernoulli(probs=noise)

    def apply(self, x: torch.Tensor, **kwargs):  # pylint: disable=unused-argument
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        assert x.ndim == 3
        m = self.bernoulli.sample(x.size()).to(x.device)
        m = m * x.gt(0).float()
        noise_value = 1 + torch.randint_like(x, self.min_, self.max_ + 1).to(x.device)  # 1 or 2
        return (x * (1 - m) + noise_value * m).detach().numpy()

    def get_params(self):
        return {'noise': self.noise}

    @property
    def targets(self) -> Dict[str, Callable]:
        return {"image": self.apply}

    def get_transform_init_args_names(self):
        return []
    def get_params_dependent_on_targets(self, params):
        return {}


class WM811KTransform(object):
    """Transformations for wafer bin maps from WM-811K."""
    def __init__(self,
                 size: tuple = (32, 32),
                 mode: str = 'test',
                 **kwargs):
        assert mode in ['test', 'weak']

        if isinstance(size, int):
            size = (size, size)
        defaults = dict(size=size, mode=mode)
        defaults.update(kwargs)   # Augmentation-specific arguments are configured here.
        self.defaults = defaults  # Falls back to default values if not specified.

        if mode == 'test':
            transform = self.test_transform(**defaults)
        elif mode == 'weak':
            transform = self.weak_transform(**defaults)
        else:
            raise NotImplementedError

        self.transform = A.Compose(transform)

    def __call__(self, img):
        return self.transform(image=img)['image']

    def __repr__(self):
        repr_str = self.__class__.__name__
        for k, v in self.defaults.items():
            repr_str += f"\n{k}: {v}"
        return repr_str

    @staticmethod
    def weak_transform(size: tuple, **kwargs):
        ratio = (0.9, 1.1)  # WaPIRL
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.RandomCrop(height=size[0], width=size[1], p=0.5),     # todo : 적용해봐야하나..
            A.RandomResizedCrop(*size, scale=(0.7, 0.95), ratio=ratio, interpolation=cv2.INTER_NEAREST, p=0.5),  # 230309
            ToWBM()
        ]
        return transform

    @staticmethod
    def test_transform(size: tuple, **kwargs) -> list:  # pylint: disable=unused-argument
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            ToWBM(),
        ]

        return transform


class WM811KTransformMultiple(object):
    """Transformations for wafer bin maps from WM-811K."""
    def __init__(self,
                 args,
                 saliency_map,
                 **kwargs
                 ):
        size = (args.size_xy, args.size_xy)
        _transforms = []
        resize_transform = A.Resize(*size, interpolation=cv2.INTER_NEAREST, always_apply=True)
        _transforms.append(resize_transform)

        self.modes = []
        self.magnitudes = []
        
        
       
        # generate modes and magnitudes
        for i in range(args.n_weaks_combinations):
            mode = random.choice(args.aug_types)
            magnitude = random.rand()
            self.modes.append(mode)
            self.magnitudes.append(magnitude)

        #TODO: cutout 파라미터!!
        cutout_size = 5
        cutout_fill_value = 1  # TODO : 채울 값을 정상값으로 지정. 기본은 0(검정). transform이 일어나는 시점이 0,1,2 인 상황인가?
 
        caption = "["
        ############################################################################################################################################
        # keep-cutout  or cutout
        for i in range(len(self.magnitudes)):
            mode, magnitude = self.modes[i], self.magnitudes[i]
            num_holes: int = int(5 * magnitude) + 1
            if mode == 'cutout':
                _transforms.append(ToWBM())
                if args.keep:
                    _transforms.append(
                        KeepCutout(args=args, saliency_map=saliency_map, num_holes=num_holes, max_h_size=cutout_size, max_w_size=cutout_size, 
                                   fill_value=cutout_fill_value, p=1.0)
                    )
                else:
                    _transforms.append(
                        A.Cutout(num_holes=num_holes, max_h_size=4, max_w_size=4, fill_value=0, p=1.0)
                    )
                caption += f"{mode}({int(magnitude*100)}), " if i != args.n_weaks_combinations-1  else f"{mode}({int(magnitude*100)}), "
      
        # noise
        for i in range(len(self.magnitudes)):
            mode, magnitude = self.modes[i], self.magnitudes[i]
            if mode == 'noise':
                range_magnitude = (0., 0.10) #TODO: noise min, max 지정
                final_magnitude = (range_magnitude[1] - range_magnitude[0]) * magnitude + range_magnitude[0]
                _transforms.append(ToWBM())
                _transforms.append(MaskedBernoulliNoise(noise=final_magnitude))
                caption += f"{mode}({int(magnitude*100)}), " if i != args.n_weaks_combinations-1  else f"{mode}({int(magnitude*100)}), "
        # crop, rotate, shift
        for i in range(args.n_weaks_combinations):
            mode, magnitude = self.modes[i], self.magnitudes[i]
            if mode == 'crop':
                range_magnitude = (0.7, 1.0)  # scale  # cropout min, max 범위 지정
                final_magnitude = (range_magnitude[1] - range_magnitude[0]) * magnitude + range_magnitude[0]
                ratio = (0.9, 1.1)  # WaPIRL
                _transforms.append(A.RandomResizedCrop(*size, scale=(0.7, 0.9), ratio=ratio, interpolation=cv2.INTER_NEAREST, p=1.0),)
                caption += f"{mode}({int(magnitude*100)}), " if i != args.n_weaks_combinations-1  else f"{mode}({int(magnitude*100)}), "
            elif mode == 'cutout' and not args.keep:
                #TODO: cutout 구멍 개수 파라미터
                num_holes: int = 1 + int(5 * magnitude)  # WaPIRL 기본 셋팅 4
                _transforms.append(
                    A.Cutout(num_holes=num_holes, max_h_size=cutout_size, max_w_size=cutout_size, fill_value=cutout_fill_value, p=1.0)
                )
                caption += f"{mode}({int(magnitude*100)}), " if i != args.n_weaks_combinations-1  else f"{mode}({int(magnitude*100)}), "
            elif mode == 'rotate':
                # range_magnitude = (0, 360)
                # final_magnitude = int((range_magnitude[1] - range_magnitude[0]) * magnitude + range_magnitude[0])
                _transforms.append(A.Rotate(limit=(-180, 180), interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),)
                caption += f"{mode}({int(magnitude*100)}), " if i != args.n_weaks_combinations-1  else f"{mode}({int(magnitude*100)}), "
            elif mode == 'shift':
                range_magnitude = (0, 0.25)  #  shift min, max 범위 지정
                final_magnitude = int((range_magnitude[1] - range_magnitude[0]) * magnitude + range_magnitude[0])
                _transforms.append(A.ShiftScaleRotate(
                shift_limit=final_magnitude,
                scale_limit=0,
                rotate_limit=0,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
                ),)
                caption += f"{mode}({int(magnitude*100)}), " if i != args.n_weaks_combinations-1  else f"{mode}({int(magnitude*100)}), "
        ############################################################################################################################################        
        caption = re.sub(', $', '', caption)
        caption += "]"

        _transforms.append(ToWBM())
        self.transform = A.Compose(_transforms)
        self.caption = caption

    def __call__(self, img):
        return self.transform(image=img)['image'], self.caption


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=20,
                                  padding=int(20*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=20,
                                  padding=int(20*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformFixMatchWafer(object):
    def __init__(self, args):
        self.weak = A.Compose([
            A.Resize(width=args.size_xy, height=args.size_xy, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),  #TODO: Ensemble 논문과 동일하게 셋팅
            # A.RandomCrop(height=args.size_xy, width=args.size_xy),  #TODO: 이거 필요없지 않을까..
            ToWBM()
        ])

        self.basic = A.Compose([
            A.Resize(width=args.size_xy, height=args.size_xy, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),  #TODO: Ensemble 논문과 동일하게 셋팅
            # A.RandomCrop(height=args.size_xy, width=args.size_xy),  #TODO: 이거 필요없지 않을까..
        ])

        if args.keep:
            # todo : change specific directory for proportion
            t_prop = args.proportion if args.fix_keep_proportion < 0. else args.fix_keep_proportion
            checkpoint = torch.load(f'results/wm811k-supervised-{t_prop}/model_best.pth.tar')
            print(f"we get sailency map from the pretrained model with accruacy of {checkpoint['acc']} and macro f1 score of {checkpoint['best_f1']} and proportion of {t_prop}")
            args.supervised_model = create_model(args, keep=True)
            args.supervised_model.load_state_dict(checkpoint['state_dict'])
        self.args = args

    def __call__(self, x, saliency_map):
        weak = self.weak(image=x)['image']
        basic = self.basic(image=x)
        strong_trans = WM811KTransformMultiple(self.args, saliency_map)
        strong, caption = strong_trans(basic['image'])
        return weak, strong, caption


class KeepCutout(ImageOnlyTransform):
    def __init__(self,
                 args,
                 saliency_map: np.asarray,
                 num_holes: int = 8,
                 max_h_size: int = 8,
                 max_w_size: int = 8,
                 fill_value: Union[int, float] = 0,
                 always_apply: bool = False,
                 p: float = 0.5):
        super(KeepCutout, self).__init__(always_apply, p)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value
        self.args = args
        self.saliency_map = saliency_map

    # 이 augmentation의 경우에는 데이터가 이미 0,1,2로 처리된 데이터를 학습을 하므로 여러개의 augmetantion을 취할 경우,
    # 현재 적용중인 'crop', 'cutout', 'noise', 'rotate', 'shift'  augmentation을 
    # 아래의 순서대로 정렬해서 조합한다. crop /  rotate / shift가 처리가 된 이후,
    # noise -> cutout 되도록
    def apply(self, img, **params):

        # for param in self.args.supervised_model.parameters():
        #     param.requires_grad = False
        # self.args.supervised_model.eval()
        #
        # img_origin = np.copy(img)

        # if img.ndim == 3:
        #     img = torch.unsqueeze(img, dim=0)
        # assert img.size()[0] == 1
        #
        # images_ = F.one_hot(img.long(), num_classes=3).squeeze().float().unsqueeze(0)
        # images_ = images_.permute(0, 3, 1, 2)  # (c, h, w)
        # images_ = images_.to(self.args.device)
        # images_.requires_grad = True
        # self.args.supervised_model = self.args.supervised_model.to(self.args.device)
        # self.args.supervised_model.eval()  # 고정 안시킴 ㅠㅠ
        # preds = self.args.supervised_model(images_)
        # score, _ = torch.max(preds, 1)
        # score.mean().backward()
        # slc_, _ = torch.max(torch.abs(images_.grad), dim=1)
        # b, h, w = slc_.shape
        # slc_ = slc_.view(slc_.size(0), -1)
        # slc_ -= slc_.min(1, keepdim=True)[0]
        # slc_ /= slc_.max(1, keepdim=True)[0]
        # slc_ = slc_.view(b, h, w)
        slc_ = self.saliency_map
        b, h, w = slc_.shape

        mask = np.ones((h, w), np.float32)
        for i, slc in enumerate(slc_):
            while True:
                y = self.max_h_size//2 + np.random.randint(h - self.max_h_size//2)
                x = self.max_h_size//2 + np.random.randint(w - self.max_h_size//2)
                y1 = np.clip(y - self.max_h_size // 2, 0, h)
                y2 = np.clip(y + self.max_h_size // 2, 0, h)
                x1 = np.clip(x - self.max_h_size // 2, 0, w)
                x2 = np.clip(x + self.max_h_size // 2, 0, w)
                # print(self.max_h_size, x, y, slc[y1: y2, x1: x2].size(),  slc[y1: y2, x1: x2].mean())
                if slc[y1: y2, x1: x2].mean() < self.args.tau:
                    mask[y1: y2, x1: x2] = 0.
                    break
                # print('retrying...')
        img_keep_cutout = img * np.expand_dims(mask, axis=-1)
        return img_keep_cutout

    @property
    def targets(self) -> Dict[str, Callable]:
        return {"image": self.apply}

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}

