import os
import torch


def check_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def update_state(weight, train_loss, val_pix_acc, val_loss, val_miu):
    state = {
        'weight': weight,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_pix_acc': val_pix_acc,
        'val_miu': val_miu
    }
    return state


def save_ckpt_file(ckpt_path, state):

    check_mkdir(os.path.split(ckpt_path)[0])
    with open(ckpt_path, 'wb') as fout:
        torch.save(state, fout)
