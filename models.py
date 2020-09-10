import torch.nn as nn
import modified_resnet
from efficientnet_pytorch import EfficientNet

def make_model(model_name_: str) -> nn.Module:
    model_name = model_name_.lower()

    if model_name.startswith('resnet'):
        model: nn.Module = getattr(modified_resnet, model_name)(pretrained=False, progress=False, num_classes=10)
    elif model_name.startswith('efficientnet'):
        model: nn.Module = EfficientNet.from_name(model_name, in_channels=2)
    else:
        raise NotImplementedError(f'Unknown model name: {model_name_}')

    # model.fc = nn.Linear(model.fc.in_features, 10)
    return model
