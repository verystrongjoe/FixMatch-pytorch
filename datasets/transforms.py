import torch
import numpy as np
from numpy import random
from torchvision import transforms
from .randaugment import RandAugmentMC
from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
from torch.distributions import Bernoulli
import albumentations as A
import cv2
from utils.common import create_model
from typing import Any, Dict, Tuple, Union
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast




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
        self.noise = noise
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

        if mode == 'crop':
            transform = self.crop_transform(**defaults)
        elif mode == 'cutout':
            transform = self.cutout_transform(**defaults)
        elif mode == 'noise':
            transform = self.noise_transform(**defaults)
        elif mode == 'rotate':
            transform = self.rotate_transform(**defaults)
        elif mode == 'shift':
            transform = self.shift_transform(**defaults)
        elif mode == 'test':
            transform = self.test_transform(**defaults)
        elif mode == 'weak':
            transform = self.weak_transform(**defaults)
        elif mode in ['crop+cutout', 'cutout+crop']:
            transform = self.crop_cutout_transform(**defaults)
        elif mode in ['crop+noise', 'noise+crop']:
            transform = self.crop_noise_transform(**defaults)
        elif mode in ['crop+rotate', 'rotate+crop']:
            transform = self.crop_rotate_transform(**defaults)
        elif mode in ['crop+shift', 'shift+crop']:
            transform = self.crop_shift_transform(**defaults)
        elif mode in ['cutout+noise', 'noise+cutout']:
            transform = self.cutout_noise_transform(**defaults)
        elif mode in ['cutout+rotate', 'rotate+cutout']:
            transform = self.cutout_rotate_transform(**defaults)
        elif mode in ['cutout+shift', 'shift+cutout']:
            transform = self.cutout_shift_transform(**defaults)
        elif mode in ['noise+rotate', 'rotate+noise']:
            transform = self.noise_rotate_transform(**defaults)
        elif mode in ['noise+shift', 'shift+noise']:
            transform = self.noise_shift_transform(**defaults)
        elif mode in ['rotate+shift', 'shift+rotate']:
            transform = self.rotate_shift_transform(**defaults)
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
        transform = [
            # transform.RandomHorizontalFlip(),
            # transforms.RandomCrop(size=32,
            #                       padding=int(32 * 0.125),
            #                       padding_mode='reflect'),
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(),
            A.RandomCrop(height=size[0], width=size[1], p=1.0),
            ToWBM()
        ]
        return transform

    @staticmethod
    def crop_transform(size: tuple, scale: tuple = (0.5, 1.0), ratio: tuple = (0.9, 1.1), **kwargs) -> list:  # pylint: disable=unused-argument
        """
        Crop-based augmentation, with `albumentations`.
        Expects a 3D numpy array of shape [H, W, C] as input.
        """
        transform = [
            A.RandomResizedCrop(*size, scale=scale, ratio=ratio, interpolation=cv2.INTER_NEAREST, p=1.0),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def cutout_transform(size: tuple, num_holes: int = 4, cut_ratio: float = 0.2, **kwargs) -> list:  # pylint: disable=unused-argument
        cut_h = int(size[0] * cut_ratio)
        cut_w = int(size[1] * cut_ratio)
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Cutout(num_holes=num_holes, max_h_size=cut_h, max_w_size=cut_w, fill_value=0, p=kwargs.get('cutout_p', 0.5)),
            ToWBM()
        ]

        return transform

    @staticmethod
    def noise_transform(size: tuple, noise: float = 0.05, **kwargs) -> list:  # pylint: disable=unused-argument
        if noise <= 0.:
            raise ValueError("'noise' probability must be larger than zero.")
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            ToWBM(),
            MaskedBernoulliNoise(noise=noise),
        ]

        return transform

    @staticmethod
    def rotate_transform(size: tuple, **kwargs) -> list:  # pylint: disable=unused-argument
        """
        Rotation-based augmentation, with `albumentations`.
        Expects a 3D numpy array of shape [H, W, C] as input.
        """
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Rotate(limit=180, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def shift_transform(size: tuple, shift: float = 0.25, **kwargs) -> list:  # pylint: disable=unused-argument
        transform = [
            A.ShiftScaleRotate(
                shift_limit=shift,
                scale_limit=0,
                rotate_limit=0,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def test_transform(size: tuple, **kwargs) -> list:  # pylint: disable=unused-argument
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def crop_cutout_transform(size: tuple,
                              scale: tuple = (0.5, 1.0), ratio: tuple = (0.9, 1.1),
                              num_holes: int = 4, cut_ratio: float = 0.2,
                              **kwargs) -> list:
        cut_h = int(size[0] * cut_ratio)
        cut_w = int(size[1] * cut_ratio)
        transform = [
            A.RandomResizedCrop(*size, scale=scale, ratio=ratio, interpolation=cv2.INTER_NEAREST, p=1.0),
            A.Cutout(num_holes=num_holes, max_h_size=cut_h, max_w_size=cut_w, fill_value=0, p=kwargs.get('cutout_p', 0.5)),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def crop_noise_transform(size: tuple, # pylint: disable=unused-argument
                             scale: tuple = (0.5, 1.0), ratio: tuple = (0.9, 1.1),
                             noise: float = 0.05,
                             **kwargs) -> list:
        transform = [
            A.RandomResizedCrop(*size, scale=scale, ratio=ratio, interpolation=cv2.INTER_NEAREST, p=1.0),
            ToWBM(),
            MaskedBernoulliNoise(noise=noise),
        ]

        return transform

    @staticmethod
    def crop_rotate_transform(size: tuple, scale: tuple = (0.5, 1.0), ratio: tuple = (0.9, 1.1), **kwargs) -> list: # pylint: disable=unused-argument
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Rotate(limit=180, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.RandomResizedCrop(*size, scale=scale, ratio=ratio, interpolation=cv2.INTER_NEAREST, p=1.0),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def crop_shift_transform(size: tuple,  # pylint: disable=unused-argument
                             scale: tuple = (0.5, 1.0), ratio: tuple = (0.9, 1.1),
                              shift: float = 0.25,
                             **kwargs) -> list:
        transform = [
            A.RandomResizedCrop(*size, scale=scale, ratio=ratio, interpolation=cv2.INTER_NEAREST, p=1.0),
            A.ShiftScaleRotate(
                shift_limit=shift,
                scale_limit=0,
                rotate_limit=0,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def cutout_noise_transform(size: tuple,
                               num_holes: int = 4, cut_ratio: float = 0.2,
                               noise: float =0.05,
                               **kwargs):
        cut_h = int(size[0] * cut_ratio)
        cut_w = int(size[1] * cut_ratio)
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Cutout(num_holes=num_holes, max_h_size=cut_h, max_w_size=cut_w, fill_value=0, p=kwargs.get('cutout_p', 0.5)),
            ToWBM(),
            MaskedBernoulliNoise(noise=noise),
        ]
        return transform

    @staticmethod
    def cutout_rotate_transform(size: tuple,
                                num_holes: int = 4, cut_ratio: float = 0.2,
                                **kwargs):
        cut_h = int(size[0] * cut_ratio)
        cut_w = int(size[1] * cut_ratio)
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Rotate(limit=180, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.Cutout(num_holes=num_holes, max_h_size=cut_h, max_w_size=cut_w, fill_value=0, p=kwargs.get('cutout_p', 0.5)),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def cutout_shift_transform(size: tuple,
                               num_holes: int = 4, cut_ratio: float = 0.2,
                               shift: float = 0.25,
                               **kwargs):
        cut_h = int(size[0] * cut_ratio)
        cut_w = int(size[1] * cut_ratio)
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Cutout(num_holes=num_holes, max_h_size=cut_h, max_w_size=cut_w, fill_value=0, p=kwargs.get('cutout_p', 0.5)),
            A.ShiftScaleRotate(
                shift_limit=shift,
                scale_limit=0,
                rotate_limit=0,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def noise_rotate_transform(size: tuple, noise: float = 0.05, **kwargs):  # pylint: disable=unused-argument
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Rotate(limit=180, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            ToWBM(),
            MaskedBernoulliNoise(noise=noise),
        ]

        return transform

    @staticmethod
    def noise_shift_transform(size: tuple, noise: float = 0.05, shift: float = 0.25, **kwargs):  # pylint: disable=unused-argument
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.ShiftScaleRotate(
                shift_limit=shift,
                scale_limit=0,
                rotate_limit=0,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            ToWBM(),
            MaskedBernoulliNoise(noise=noise),
        ]

        return transform

    @staticmethod
    def rotate_shift_transform(size: tuple, shift: float = 0.25, **kwargs):  # pylint: disable=unused-argument
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Rotate(limit=180, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.ShiftScaleRotate(
                shift_limit=shift,
                scale_limit=0,
                rotate_limit=0,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            ToWBM(),
        ]

        return transform


class WM811KTransformMultiple(object):
    """Transformations for wafer bin maps from WM-811K."""
    def __init__(self,
                 args,
                 **kwargs
                 ):
        _transforms = []
        size = (args.size_xy, args.size_xy)
        resize_transform = A.Resize(*size, interpolation=cv2.INTER_NEAREST)
        _transforms.append(resize_transform)

        modes = []
        magnitudes = []

        for i in range(args.n_weaks_combinations):
            mode = random.choice(args.aug_types)
            magnitude = random.rand()
            # mode = 'cutout'
            # magnitude = 0.1
            modes.append(mode)
            magnitudes.append(magnitude)
            # args.logger.info(f"[{i}] mode: {mode}, magnitude: {magnitude}")
            if mode == 'crop':
                range_magnitude = (0.5, 1.0)  # scale
                final_magnitude = (range_magnitude[1] - range_magnitude[0]) * magnitude + range_magnitude[0]
                ratio = (0.9, 1.1)  # WaPIRL
                # Torchvision's variant of crop a random part of the input and rescale it to some size.
                    # scale	[float, float] :  range of size of the origin size cropped
                    # ratio	[float, float] :  range of aspect ratio of the origin aspect ratio cropped
                    # interpolation	OpenCV flag : flag that is used to specify the interpolation algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.
                    # p	float : probability of applying the transform. Default: 1.
                _transforms.append(A.RandomResizedCrop(*size, scale=(0.5, final_magnitude), ratio=ratio, interpolation=cv2.INTER_NEAREST, p=1.0),)
            elif mode == 'cutout' and not args.keep:
                num_holes: int = int(5 * magnitude)  # WaPIRL 기본 셋팅 4에 대해서 실행 -> 230106 연구미팅 셋팅값 지금 1로 변경(230115)
                _transforms.append(
                    A.Cutout(num_holes=num_holes, max_h_size=4, max_w_size=4, fill_value=0, p=1.0)
                )
            elif mode == 'rotate':
                range_magnitude = (0, 360)
                final_magnitude = int((range_magnitude[1] - range_magnitude[0]) * magnitude + range_magnitude[0])
                _transforms.append(A.Rotate(limit=final_magnitude, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),)
            elif mode == 'shift':
                range_magnitude = (0, 0.5)
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

        for i in range(len(magnitudes)):
            mode, magnitude = modes[i], magnitudes[i]
            if mode == 'cutout':
                num_holes: int = int(5 * magnitude)  # WaPIRL 기본 셋팅 4에 대해서 실행 -> 230106 연구미팅 셋팅값 지금 1로 변경(230115)
                _transforms.append(ToWBM())
                if args.keep:
                    _transforms.append(
                        KeepCutout(args=args, num_holes=num_holes, max_h_size=4, max_w_size=4, fill_value=0, p=1.0)
                    )
                else:
                    _transforms.append(
                        A.Cutout(num_holes=num_holes, max_h_size=4, max_w_size=4, fill_value=0, p=1.0)
                    )

        for i in range(len(magnitudes)):
            mode, magnitude = modes[i], magnitudes[i]
            if mode == 'noise':
                range_magnitude = (0., 0.20)
                final_magnitude = (range_magnitude[1] - range_magnitude[0]) * magnitude + range_magnitude[0]
                _transforms.append(ToWBM())
                _transforms.append(MaskedBernoulliNoise(noise=final_magnitude))

        _transforms.append(ToWBM())

        self.transform = A.Compose(_transforms)

    def __call__(self, img):
        return self.transform(image=img)['image']


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
            A.HorizontalFlip(),
            A.RandomCrop(height=args.size_xy, width=args.size_xy),  # keep!!
            ToWBM()
        ])

        self.basic = A.Compose([
            A.Resize(width=args.size_xy, height=args.size_xy, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(),
            A.RandomCrop(height=args.size_xy, width=args.size_xy),  # keep!!
        ])

        if args.keep:
            # todo : change specific directory for proportion
            checkpoint = torch.load('results/wm811k-supervised/model_best.pth.tar')
            args.supervised_model = create_model(args)
            args.supervised_model.load_state_dict(checkpoint['state_dict'])
        self.args = args

    def __call__(self, x):
        weak = self.weak(image=x)['image']
        basic = self.basic(image=x)
        strong_trans = WM811KTransformMultiple(self.args)
        strong = strong_trans(basic['image'])
        return weak, strong


class KeepCutout(DualTransform):
    def __init__(self,
                 args,
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

    # 이 augmentation의 경우에는 데이터가 이미 0,1,2로 처리된 데이터를 학습을 하므로 여러개의 augmetantion을 취할 경우,
    # 현재 적용중인 'crop', 'cutout', 'noise', 'rotate', 'shift'  augmentation을 
    # 아래의 순서대로 정렬해서 조합한다. crop /  rotate / shift가 처리가 된 이후,
    # noise -> cutout 되도록
    def apply(self, img, **params):
        for param in self.args.supervised_model.parameters():
            param.requires_grad = False
        self.args.supervised_model.eval()

        img_origin = np.copy(img)
        img = torch.from_numpy(img)

        if img.ndim == 3:
            img = torch.unsqueeze(img, dim=0)
        assert img.size()[0] == 1

        images_ = F.one_hot(img.long(), num_classes=3).squeeze().float().unsqueeze(0)
        images_ = images_.permute(0, 3, 1, 2)  # (c, h, w)

        images_.requires_grad = True
        preds = self.args.supervised_model(images_)
        score, _ = torch.max(preds, 1)
        score.mean().backward()
        slc_, _ = torch.max(torch.abs(images_.grad), dim=1)
        b, h, w = slc_.shape
        slc_ = slc_.view(slc_.size(0), -1)
        slc_ -= slc_.min(1, keepdim=True)[0]
        slc_ /= slc_.max(1, keepdim=True)[0]
        slc_ = slc_.view(b, h, w)

        mask = np.ones((h, w), np.float32)
        for i, (img, slc) in enumerate(zip(images_, slc_)):
            while (True):
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
        origin_img = img_origin * np.expand_dims(mask, axis=-1)
        return origin_img

    @property
    def targets(self) -> Dict[str, Callable]:
        return {"image": self.apply}

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}
