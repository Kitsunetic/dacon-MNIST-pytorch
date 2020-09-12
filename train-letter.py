import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_optimizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # hyper parameters
    experiment_name = args.name
    checkpoint_dir = Path(args.checkpoint_dir) / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    minimum_checkpoint_epoch = 10  # the minimum epoch to save checkpoint
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    train_valid_ratio = args.train_valid_ratio
    early_stopping_patience = 30
    lr_decay_patience = args.lr_decay_patience
    lr = args.lr
    lr_decay_rate = 0.5
    image_resize = args.image_resize
    cpus = args.cpus
    gpus = args.gpus
    model_name = args.model_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load train data
    train_csv = pd.read_csv('data/train.csv')
    X, Y, Z = datasets.parse_csv(train_csv)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Z, test_size=train_valid_ratio)
    train_ds = datasets.BaseDataset(X_train, Y_train, is_train=True, image_resize=image_resize)
    valid_ds = datasets.BaseDataset(X_valid, Y_valid, is_train=False, image_resize=image_resize)
    train_loader = DataLoader(train_ds, batch_size, num_workers=cpus, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size, num_workers=cpus, shuffle=False)
    print(f'Dataset: train[{len(train_ds)}] validation[{len(valid_ds)}]')

    # make model
    model = models.from_name(model_name, num_classes=26)
    print(model)

    if gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch_optimizer.RAdam(model.parameters(), lr=lr)

    min_loss = math.inf
    early_stopping_counter = 0
    lr_decay_counter = 0
    for epoch in range(1, num_epochs + 1):
        # train
        model.train()
        losses = []
        correct, total = 0, 0
        with tqdm(total=len(train_loader), ncols=100, desc=f'[{epoch:03d}/{num_epochs:03d}] Train') as t:
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update progress
                _, preds = torch.max(output.data, 1)
                total += y.size(0)
                correct += (preds == y).sum().item()
                accuracy = correct / total
                losses.append(loss.item())
                mean_loss = sum(losses) / len(losses)
                t.set_postfix_str(f'loss: {mean_loss:.4f} acc: {accuracy:.4f}', refresh=False)
                t.update()

        # validation
        with torch.no_grad():
            model.eval()
            losses = []
            correct, total = 0, 0
            with tqdm(total=len(valid_loader), ncols=100, desc=f'[{epoch:03d}/{num_epochs:03d}] Validation') as t:
                for x, y in valid_loader:
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    loss = criterion(output, y)

                    # update progress
                    _, preds = torch.max(output.data, 1)
                    total += y.size(0)
                    correct += (preds == y).sum().item()
                    accuracy = correct / total
                    losses.append(loss.item())
                    mean_loss = sum(losses) / len(losses)
                    t.set_postfix_str(f'val_loss: {mean_loss:.4f} val_acc: {accuracy:.4f}', refresh=False)
                    t.update()

        # save checkpoint only when val_loss decreased
        if mean_loss < min_loss:
            if epoch > minimum_checkpoint_epoch:
                time.sleep(0.25)
                checkpoint_path = str(
                    checkpoint_dir / f'ckpt-epoch{epoch:03d}-val_loss{mean_loss:.4f}-val_acc{accuracy:.4f}.pth')
                print(f'val_loss decreased from', min_loss, 'to', mean_loss)
                print('Save checkpoint to', checkpoint_path)
                with open(checkpoint_path, 'wb') as f:
                    torch.save({
                        'model_state_dict': model.state_dict() if gpus <= 1 else model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, f)

                min_loss = mean_loss
                early_stopping_counter = 0
                lr_decay_counter = 0
        else:
            # early stopping
            early_stopping_counter += 1
            if early_stopping_counter > early_stopping_patience:
                print('Early stopping: val_loss didn\'t decreased for', early_stopping_counter, 'epochs.')
                break

            # lr decay
            lr_decay_counter += 1
            if lr_decay_counter > lr_decay_patience:
                print('LR decaying from', lr, 'to', lr * lr_decay_rate)
                lr_decay_counter = 0
                lr *= lr_decay_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-epochs', type=int, default=200)
    p.add_argument('--cpus', type=str, default=12)
    p.add_argument('--gpus', type=str, default=1)
    p.add_argument('--train-valid-ratio', type=float, default=0.2)
    p.add_argument('--checkpoint-dir', type=str, default='checkpoint')
    p.add_argument('--seed', type=int, default=867243624)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr-decay-patience', type=int, default=5)
    p.add_argument('--image-resize', type=int, default=224)
    p.add_argument('name', type=str)
    p.add_argument('model_name', type=str)

    args = p.parse_args(sys.argv[1:])
    main(args)
