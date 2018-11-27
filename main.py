# -*- coding: utf-8 -*-
import os
import gc
import sys
import copy
import pickle
import logging
import argparse

from utils.trainer_verbose import train_with_ignite
from utils import check_mkdir

import torch

logger = logging.getLogger('hair segmentation project')

# WIP

def get_args():
    model_names = [] # from networks
    scheduler_names = [] # default torch or others
    optimizer_names = [] # default torch or others
    pretrained_names = [] # from models or torchvision

    parser = argparse.ArgumentParser(description='Hair Segmentation')
    parser.add_argument('--networks', default='resnet101')
    parser.add_argument('--dataset', default='figaro')
    parser.add_argument('--data_dir', default='./data/Figaro1k')
    parser.add_argument('--scheduler', default='ReduceLROnPlateau')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--description', type=str, default='binary_class')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--use_pretrained', type=str, default='ImageNet')

    args = parser.parse_args()

    return args

def main():
    args = get_args()
    
    check_mkdir('./logs')
    
    logging_name = './logs/{}_{}_lr_{}.txt'.format(args.networks,
                                                args.optimizer,
                                                args.lr)
    
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(asctime)10s][%(levelname)s] %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S'
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(logging_name)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info('arguments:{}'.format(" ".join(sys.argv)))

    train_with_ignite(networks=args.networks,
                      dataset=args.dataset,
                      data_dir=args.data_dir,
                      scheduler=args.scheduler,
                      batch_size=args.batch_size,
                      description=args.description,
                      epochs=args.epochs,
                      lr=args.lr,
                      num_workers=args.num_workers,
                      optimizer=args.optimizer,
                      use_pretrained=args.use_pretrained,
                      momentum=args.momentum,
                      img_size=args.img_size,
                      logger=logger)


if __name__ == '__main__':
    main()
