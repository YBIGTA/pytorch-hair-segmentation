# -*- coding: utf-8 -*-
import os
import gc
import sys
import copy
import pickle
import logging
import argparse

from utils.trainer_verbose import train_with_ignite, train_without_ignite, get_optimizer
from utils import check_mkdir

import torch

from networks import mobile_hair

logger = logging.getLogger('hair segmentation project')

def str2bool(s):
    return s.lower() in ('t', 'true', '1')


def get_args():
    parser = argparse.ArgumentParser(description='Hair Segmentation')
    parser.add_argument('--networks', default='mobilenet')
    parser.add_argument('--scheduler', default='ReduceLROnPlateau')
    parser.add_argument('--dataset', default='figaro')
    parser.add_argument('--data_dir', default='./data/Figaro1k')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--img_size',type=int, default=256)
    parser.add_argument('--use_pretrained', type=str, default='ImageNet')
    parser.add_argument('--ignite', type=str2bool, default=True)
    parser.add_argument('--visdom', type=str2bool, default=False)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--momentum', type=float, default=0.9)

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
    if args.ignite is False:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = mobile_hair.MobileMattingFCN()
        
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print('multi gpu')
                model = torch.nn.DataParallel(model)
        
        model.to(device)
        
        loss = mobile_hair.HairMattingLoss()
        
        optimizer = get_optimizer(args.optimizer, model, args.lr, args.momentum)
        # torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=0.0001, betas=(0.9, 0.999))
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        
        train_without_ignite(model, 
                             loss,
                             batch_size=args.batch_size,
                             img_size=args.img_size,
                             epochs=args.epochs,
                             lr=args.lr,
                             num_workers=args.num_workers,
                             optimizer=optimizer,
                             logger=logger,
                             gray_image=True,
                             scheduler=scheduler,
                             viz=args.visdom)
    
    else: train_with_ignite(networks=args.networks,
                      dataset=args.dataset,
                      data_dir=args.data_dir,
                      batch_size=args.batch_size,
                      epochs=args.epochs,
                      lr=args.lr,
                      num_workers=args.num_workers,
                      optimizer=args.optimizer,
                      momentum=args.momentum,
                      img_size=args.img_size,
                      logger=logger)


if __name__ == '__main__':
    main()
