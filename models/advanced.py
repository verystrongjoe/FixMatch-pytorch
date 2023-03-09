import torch

from models.alexnet import AlexNetBackbone
from models.head import LinearClassifier
from models.network_configs import ALEXNET_BACKBONE_CONFIGS
from models.network_configs import RESNET_BACKBONE_CONFIGS
from models.network_configs import VGGNET_BACKBONE_CONFIGS
from models.resnet import ResNetBackbone
from models.vggnet import VggNetBackbone

AVAILABLE_MODELS = {
    'alexnet': (ALEXNET_BACKBONE_CONFIGS, AlexNetBackbone),
    'vggnet': (VGGNET_BACKBONE_CONFIGS['16'], VggNetBackbone),
    'vggnet-bn': (VGGNET_BACKBONE_CONFIGS['16.bn'], VggNetBackbone),
    'resnet-18': (RESNET_BACKBONE_CONFIGS['18'], ResNetBackbone),
    'resnet-50': (RESNET_BACKBONE_CONFIGS['50'], ResNetBackbone),
}


class AdvancedCNN:
    def __init__(self, args):
        self.args = args
        in_channels = int(args.decouple_input) + int(args.num_channel)
        BACKBONE_CONFIGS, Backbone = AVAILABLE_MODELS[args.arch]
        self.backbone = Backbone(BACKBONE_CONFIGS, in_channels=in_channels)
        self.classifier = LinearClassifier(in_channels=self.backbone.out_channels, num_classes=args.num_classes)
        self.params = [{'params': self.backbone.parameters()}, {'params': self.classifier.parameters()}]
        self.backbone.to(args.num_gpu)
        self.classifier.to(args.num_gpu)

    def __call__(self, x: torch.Tensor):
        """Make a prediction provided a batch of samples."""
        return self.classifier(self.backbone(x))

    def train(self):
        self.backbone.train()
        self.classifier.train()

    def eval(self):
        self.backbone.eval()
        self.classifier.eval()

    def parameters(self):
        return [{'params': self.backbone.parameters()}, {'params': self.classifier.parameters()}]

    def state_dict(self):
        return {
            'backbone': self.backbone.state_dict(),
            'classifier': self.classifier.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self.backbone.load_state_dict(ckpt['backbone'])
        self.classifier.load_state_dict(ckpt['classifier'])
