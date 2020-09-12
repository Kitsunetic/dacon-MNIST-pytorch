from pathlib import Path

import torch
import torch.nn as nn

import modified_resnet
from modified_efficientnet import EfficientNet


def from_name(model_name_: str, letter_model=None, num_classes=10) -> nn.Module:
    """
    모델 생성. letter-model은 `letter_model`에 값을 주면 되고, finetune-model은 그냥 생성 후 load_weight를 이용
    :param model_name_:
    :param num_classes: 10 for digit model. 26 for letter model.
    :param in_channels:
    :param letter_model: letter-model의 checkpoint
    :return:
    """
    model_name = model_name_.lower()
    if model_name.startswith('resnet') or model_name.startswith('resnext'):
        if letter_model:
            with open(letter_model, 'rb') as f:
                data = torch.load(f)
            letter_model = modified_resnet.from_name(model_name, False, False,
                                                     in_planes=2, num_classes=26)
            letter_model.load_state_dict(data['model_state_dict'])
            feature_extractor = nn.Sequential(*list(letter_model.children())[:5])
        else:
            feature_extractor = None

        model = modified_resnet.from_name(model_name, pretrained=False, progress=False,
                                          in_planes=2, num_classes=num_classes,
                                          letter_model=feature_extractor)

    elif model_name.startswith('efficientnet'):
        if letter_model:
            with open(letter_model, 'rb') as f:
                data = torch.load(f)
            letter_model = EfficientNet.from_name(model_name, in_channels=2, num_classes=26)
            letter_model.load_state_dict(data['model_state_dict'])
        else:
            letter_model = None

        model = EfficientNet.from_name(model_name, in_channels=2, num_classes=num_classes,
                                       letter_model=letter_model)

    else:
        raise NotImplementedError(f'Unknown model name: {model_name_}')

    return model
