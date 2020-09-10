import torch.nn as nn
from efficientnet_pytorch import EfficientNet

import modified_resnet


def make_model(model_name_: str) -> nn.Module:
    model_name = model_name_.lower()

    if model_name.startswith('resnet') or model_name.startswith('resnext'):
        model: nn.Module = getattr(modified_resnet, model_name)(pretrained=False, progress=False, num_classes=10)
    elif model_name.startswith('efficientnet'):
        model: nn.Module = EfficientNet.from_name(model_name, in_channels=2)
    else:
        raise NotImplementedError(f'Unknown model name: {model_name_}')
    return model


def letter_model(model_name_: str):
    model_name = model_name_.lower()

    if model_name.startswith('resnet') or model_name.startswith('resnext'):
        model: nn.Module = getattr(modified_resnet, model_name)(pretrained=False, progress=False, num_classes=26)
    else:
        raise NotImplementedError(f'Unknown model name: {model_name_}')
    return model
