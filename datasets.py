import random
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch.utils.data


def parse_csv(csv: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    X = (csv[[str(i) for i in range(784)]]).values.reshape(-1, 28, 28, 1).astype(np.uint8)
    X = np.stack(X)

    char2num = {chr(i): i - 65 for i in range(65, 91)}
    # num2char = {i - 65: chr(i) for i in range(65, 91)}

    if 'digit' in csv.columns:
        Y = np.array([v for v in csv['digit'].values], dtype=np.int64)
    else:
        Y = None

    Z = np.array([char2num[v] for v in csv['letter'].values], dtype=np.int64)

    return X, Y, Z


class RandomAffine:
    def __init__(self, threshold=0., degrees=0, scale=0., horizontal_shift=0., vertical_shift=0.):
        self.threshold = threshold
        self.degrees = degrees
        self.scale = scale
        self.horizontal_shift = horizontal_shift
        self.vertical_shift = vertical_shift

    def __call__(self, img: np.ndarray):
        if self.threshold <= 0:
            return img

        w, h = img.shape[1], img.shape[0]

        degrees, scale, h_shift, v_shift = 0, 0, 0, 0
        if self.threshold <= random.random():
            degrees = self.degrees * (2 * random.random() - 1)
        if self.threshold <= random.random():
            scale = self.scale * (2 * random.random() - 1)
        if self.threshold <= random.random():
            h_shift = self.horizontal_shift * (2 * random.random() - 1) * w
        if self.threshold <= random.random():
            v_shift = self.vertical_shift * (2 * random.random() - 1) * h

        mat = cv2.getRotationMatrix2D((w // 2, h // 2), degrees, 1 + scale)
        # combine rotation matrix with transformatio|n matrix
        mat[0, 2] += h_shift
        mat[1, 2] += v_shift
        img = cv2.warpAffine(img, mat, (w, h))

        return img


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y=None, is_train=False):
        super(BaseDataset, self).__init__()

        self.X = X
        self.Y = torch.tensor(Y) if Y is not None else None
        self.is_train = is_train

        self.transform = RandomAffine(threshold=0.4, degrees=30, scale=0.25, horizontal_shift=0.25, vertical_shift=0.25)

    def _imresize(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, (280, 280))
        img = np.squeeze(img).astype(np.float32) / 255.
        img2 = img * (img >= 0.549)  # 140 / 255

        img_comb = np.stack([img, img2], axis=2)
        return img_comb

    def _totensor(self, img: np.ndarray) -> torch.Tensor:
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute([2, 0, 1])
        return img

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        x = self._imresize(x)
        if self.is_train:
            x = self.transform(x)
        x = self._totensor(x)

        if self.Y is None:
            return x
        else:
            y = self.Y[idx]
            return x, y
