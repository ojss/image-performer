import torch
import torch.nn as nn
import math
import pickle
import os, os.path
import sys
import time
import logging
import yaml
import shutil
import numpy as np
import tensorboardX
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import tensorboardX
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
from vit_pytorch.efficient import ViT
from performer_pytorch import Performer
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset,TensorDataset, DataLoader, ConcatDataset, random_split
from pathlib import Path


matplotlib.use('Agg')

sns.set()


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parse_args_and_config():
    """
    :return args, config: namespace objects that stores information in args and config files.
    """
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument(
        '--config', type=str, default='vip.yml', help='Path to the config file')
    parser.add_argument('--doc', type=str, default='0',
                        help='A string for documentation purpose')
    parser.add_argument('--verbose', type=str, default='info',
                        help='Verbose level: info | debug | warning | critical')
    args = parser.parse_args()

    # print(args.img64)

    args.log = os.path.join('transformer_logs', args.doc)
    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    new_config = dict2namespace({**config, **vars(args)})

    if os.path.exists(args.log):
        shutil.rmtree(args.log)

    os.makedirs(args.log)

    with open(os.path.join(args.log, 'config.yml'), 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)

    # setup logger
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    # add device information to args
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(new_config.seed)
    torch.cuda.manual_seed_all(new_config.seed)
    np.random.seed(new_config.seed)
    logging.info("Run name: {}".format(args.doc))

    return args, new_config


def main():
    args, config = parse_args_and_config()
    tb_logger = tensorboardX.SummaryWriter(log_dir=os.path.join('vit_logs', args.doc))
    device = config.device
    batch_size = config.train.batch_size
    lr = float(config.optim.lr)
    epochs = config.train.epochs
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    cifar_train = datasets.CIFAR10(root="CIFAR10/",
                                 train=True,
                                 download=True,
                                 transform=transform)

    cifar_test = datasets.CIFAR10(root="CIFAR10/",
                                 train=False,
                                 download=True,
                                 transform=transform)

    data_train = DataLoader(dataset=cifar_train,
                            batch_size=config.train.batch_size,
                            shuffle=True)

    data_test= DataLoader(dataset=cifar_test,
                            batch_size=config.train.batch_size//4,
                            shuffle=False)
    
    torch.manual_seed(43)
    val_size = 5000
    train_size = len(cifar_train) - val_size
    
    train_ds, val_ds = random_split(cifar_train, [train_size, val_size])
    print(len(train_ds), len(val_ds))

    efficient_transformer =  Performer(
    dim_head = 64,
    dim = config.model.p_dim,
    depth = config.model.p_depth,
    heads = config.model.p_heads,
    causal = True
    )

    model = ViT(
        dim = config.model.dim,
        image_size = config.model.image_size,
        patch_size = config.model.patch_size,
        num_classes = config.model.num_classes,
        transformer = efficient_transformer
    )

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size//4, num_workers=4, pin_memory=True)
    test_loader = DataLoader(data_test, batch_size//4, num_workers=4, pin_memory=True)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    
    step = 0
    for epoch in range(1, epochs + 1):
        acc = 0
        tot_loss = 0
        train_cnt = 0
        test_cnt = 0
        model.train()
        pbar = tqdm(train_loader)
        acc_tr = 0
        for x, y in pbar:
    #         print(x.shape)
    #         break
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            opt.zero_grad()
            loss = criterion(y_pred, y)
            loss.backward()
            opt.step()
            tot_loss += loss.item()*x.shape[0]
            train_cnt += x.shape[0]
            
            acc_tr = accuracy(y_pred, y)
            if step % config.train.log_iter == 0:
                tb_logger.add_scalar('loss', tot_loss/train_cnt, global_step=step)
                tb_logger.add_scalar('train_accuracy', acc_tr, global_step=step)
            if step % 100 == 0:
                imgs_grid = torchvision.utils.make_grid(x[:8, ...], 3)
                tb_logger.add_image('imgs', imgs_grid, global_step=step)
            pbar.set_description(f"Loss : {tot_loss/train_cnt:.4f}, Acc: {acc_tr}")
            step += 1

        model.eval()

        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            
            y_argmax = y_pred.argmax(dim = 1)
#             acc += (y == y_argmax).sum()
            acc += accuracy(y_pred, y)
            test_cnt += x.shape[0]

        print(f'epoch {epoch} : Average loss : {tot_loss/train_cnt:.4f}, test_acc : {acc.item()/test_cnt:.4f}')
        average_loss = tot_loss/train_cnt

        logging.info(f'epoch {epoch} : average_val_loss : {average_loss:.4f}, test_acc : {acc.item()/test_cnt}')
        tb_logger.add_scalar('average_val_loss', tot_loss/train_cnt, global_step=epoch)
        tb_logger.add_scalar(f'val_acc', acc.item()/test_cnt, global_step=epoch)


        logging.info("Sampling from model: {}".format(args.doc))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

if __name__ == '__main__':
    sys.exit(main())
