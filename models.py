from pathlib import Path

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

import modified_resnet


def get_digit_model(model_name_: str) -> nn.Module:
    model_name = model_name_.lower()

    if model_name.startswith('resnet') or model_name.startswith('resnext'):
        model: nn.Module = getattr(modified_resnet, model_name)(pretrained=False, progress=False, num_classes=10)
    elif model_name.startswith('efficientnet'):
        model: nn.Module = EfficientNet.from_name(model_name, in_channels=2)
    else:
        raise NotImplementedError(f'Unknown model name: {model_name_}')
    return model


def get_letter_model(model_name_: str) -> nn.Module:
    model_name = model_name_.lower()

    if model_name.startswith('resnet') or model_name.startswith('resnext'):
        model: nn.Module = getattr(modified_resnet, model_name)(pretrained=False, progress=False, num_classes=26)
    else:
        raise NotImplementedError(f'Unknown model name: {model_name_}')
    return model


def composite_model(letter_checkpoint_path: Path) -> nn.Module:
    """
    Composite model only support resnext50_32x4d
    :param letter_checkpoint_path:
    :return:
    """
    letter_model = get_letter_model('resnext50_32x4d')
    print('Load letter-model checkpoint:', letter_checkpoint_path)
    with open(letter_checkpoint_path, 'rb') as f:
        checkpoint = torch.load(f)
    letter_model.load_state_dict(checkpoint['model_state_dict'])
    feat_out = nn.Sequential(*list(letter_model.children())[:5])
    for param in feat_out.parameters():
        param.requires_grad = False

    model = modified_resnet.resnext50_32x4d(pretrained=False, progress=False, letter_model=feat_out)
    return model
