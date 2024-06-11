import random
import time
import math
from argparse import ArgumentParser
from collections import defaultdict
from itertools import islice
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision

from collections import deque
from typing import Dict, Optional, Literal
import torch.nn as nn
from grokfast import *
import wandb


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def compute_accuracy(network, dataset, device, N=2000, batch_size=50):
    """Computes accuracy of `network` on `dataset`.
    """
    with torch.no_grad():
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        correct = 0
        total = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            logits = network(x.to(device))
            predicted_labels = torch.argmax(logits, dim=1)
            correct += torch.sum(predicted_labels == labels.to(device))
            total += x.size(0)
        return (correct / total).item()


def compute_loss(network, dataset, loss_function, device, N=2000, batch_size=50):
    """Computes mean loss of `network` on `dataset`.
    """
    with torch.no_grad():
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss_fn = loss_function_dict[loss_function](reduction='sum')
        one_hots = torch.eye(10, 10).to(device)
        total = 0
        points = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            y = network(x.to(device))
            if loss_function == 'CrossEntropy':
                total += loss_fn(y, labels.to(device)).item()
            elif loss_function == 'MSE':
                total += loss_fn(y, one_hots[labels]).item()
            points += len(labels)
        return total / points


optimizer_dict = {
    'AdamW': torch.optim.AdamW,
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD
}

activation_dict = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'GELU': nn.GELU
}

loss_function_dict = {
    'MSE': nn.MSELoss,
    'CrossEntropy': nn.CrossEntropyLoss
}


def main(args):
    # Initialize wandb
    wandb.init(project="grokfast_mnist", config=args)
    
    log_freq = math.ceil(args.optimization_steps / 150)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    torch.set_default_dtype(dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load dataset
    train = torchvision.datasets.MNIST(root=args.download_directory, train=True, 
        transform=torchvision.transforms.ToTensor(), download=True)
    test = torchvision.datasets.MNIST(root=args.download_directory, train=False, 
        transform=torchvision.transforms.ToTensor(), download=True)
    train = torch.utils.data.Subset(train, range(args.train_points))
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)

    assert args.activation in activation_dict, f"Unsupported activation function: {args.activation}"
    activation_fn = activation_dict[args.activation]

    # create model
    layers = [nn.Flatten()]
    for i in range(args.depth):
        if i == 0:
            layers.append(nn.Linear(784, args.width))
            layers.append(activation_fn())
        elif i == args.depth - 1:
            layers.append(nn.Linear(args.width, 10))
        else:
            layers.append(nn.Linear(args.width, args.width))
            layers.append(activation_fn())
    mlp = nn.Sequential(*layers).to(device)
    with torch.no_grad():
        for p in mlp.parameters():
            p.data = args.initialization_scale * p.data
    nparams = sum([p.numel() for p in mlp.parameters() if p.requires_grad])
    print(f'Number of parameters: {nparams}')
    wandb.config.update({"nparams": nparams})

    # create optimizer
    assert args.optimizer in optimizer_dict, f"Unsupported optimizer choice: {args.optimizer}"
    optimizer = optimizer_dict[args.optimizer](mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # define loss function
    assert args.loss_function in loss_function_dict
    loss_fn = loss_function_dict[args.loss_function]()

    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
    norms, last_layer_norms, log_steps = [], [], []
    grads = None

    steps = 0
    one_hots = torch.eye(10, 10).to(device)
    with tqdm(total=args.optimization_steps, dynamic_ncols=True) as pbar:
        for x, labels in islice(cycle(train_loader), args.optimization_steps):
            do_log = (steps < 30) or (steps < 150 and steps % 10 == 0) or steps % log_freq == 0
            if do_log:
                train_loss = compute_loss(mlp, train, args.loss_function, device, N=len(train))
                train_acc = compute_accuracy(mlp, train, device, N=len(train))
                test_loss = compute_loss(mlp, test, args.loss_function, device, N=len(test))
                test_acc = compute_accuracy(mlp, test, device, N=len(test))
                
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                test_losses.append(test_loss)
                test_accuracies.append(test_acc)
                log_steps.append(steps)

                pbar.set_description(
                    "L: {0:1.1e}|{1:1.1e}. A: {2:2.1f}%|{3:2.1f}%".format(
                        train_loss,
                        test_loss,
                        train_acc * 100, 
                        test_acc * 100,
                    )
                )

                # Log to wandb
                wandb.log({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "step": steps
                })

            y = mlp(x.to(device))
            if args.loss_function == 'CrossEntropy':
                loss = loss_fn(y, labels.to(device))
            elif args.loss_function == 'MSE':
                loss = loss_fn(y, one_hots[labels])

            optimizer.zero_grad()
            loss.backward()

            #######

            trigger = False

            if args.filter == "none":
                pass
            elif args.filter == "ma":
                grads = gradfilter_ma(mlp, grads=grads, window_size=args.window_size, lamb=args.lamb, trigger=trigger)
            elif args.filter == "ema":
                grads = gradfilter_ema(mlp, grads=grads, alpha=args.alpha, lamb=args.lamb)
            else:
                raise ValueError(f"Invalid gradient filter type `{args.filter}`")

            #######

            optimizer.step()

            steps += 1
            pbar.update(1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--train_points", type=int, default=1000)
    parser.add_argument("--optimization_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--loss_function", type=str, default="MSE")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--initialization_scale", type=float, default=8.0)
    parser.add_argument("--download_directory", type=str, default=".")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--activation", type=str, default="ReLU")

    # Grokfast
    parser.add_argument("--filter", type=str, choices=["none", "ma", "ema", "fir"], default="none")
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--lamb", type=float, default=5.0)
    args = parser.parse_args()

    filter_str = ('_' if args.label != '' else '') + args.filter
    window_size_str = f'_w{args.window_size}'
    alpha_str = f'_a{args.alpha:.3f}'.replace('.', '')
    lamb_str = f'_l{args.lamb:.2f}'.replace('.', '')

    if args.filter == 'none':
        filter_suffix = ''
    elif args.filter == 'ma':
        filter_suffix = window_size_str + lamb_str
    elif args.filter == 'ema':
        filter_suffix = alpha_str + lamb_str
    else:
        raise ValueError(f"Unrecognized filter type {args.filter}")

    optim_suffix = ''
    if args.weight_decay != 0:
        optim_suffix = optim_suffix + f'_wd{args.weight_decay:.1e}'.replace('.', '')
    if args.lr != 1e-3:
        optim_suffix = optim_suffix + f'_lrx{int(args.lr / 1e-3)}'

    args.label = args.label + filter_str + filter_suffix + optim_suffix
    print(f'Experiment results saved under name: {args.label}')

    main(args)
