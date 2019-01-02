import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from data import get_loader
from utils import update_state, save_ckpt_file
from utils import joint_transforms as jnt_trnsf
from utils import summarize_model
from networks import get_network

import torch
import torchvision.transforms as std_trnsf

from tqdm import tqdm

def get_optimizer(string, model, lr, momentum):
    string = string.lower()
    if string == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(momentum, 0.999))
    elif string == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    raise ValueError


def train_with_ignite(networks, dataset, data_dir, batch_size, img_size,
                      epochs, lr, momentum, num_workers, optimizer, logger):

    from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
    from ignite.metrics import Loss
    from utils.metrics import MultiThresholdMeasures, Accuracy, IoU, F1score

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # build model
    model = get_network(networks)

    # log model summary
    input_size = (3, img_size, img_size)
    summarize_model(model.to(device), input_size, logger, batch_size, device)

    # build loss
    loss = torch.nn.BCEWithLogitsLoss()

    # build optimizer and scheduler
    model_optimizer = get_optimizer(optimizer, model, lr, momentum)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer)

    # transforms on both image and mask
    train_joint_transforms = jnt_trnsf.Compose([
        jnt_trnsf.RandomCrop(img_size),
        jnt_trnsf.RandomRotate(5),
        jnt_trnsf.RandomHorizontallyFlip()
    ])

    # transforms only on images
    train_image_transforms = std_trnsf.Compose([
        std_trnsf.ColorJitter(0.05, 0.05, 0.05, 0.05),
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    test_joint_transforms = jnt_trnsf.Compose([
        jnt_trnsf.Safe32Padding()
    ])

    test_image_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # transforms only on mask
    mask_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor()
        ])

    # build train / test loader
    train_loader = get_loader(dataset=dataset,
                              data_dir=data_dir,
                              train=True,
                              joint_transforms=train_joint_transforms,
                              image_transforms=train_image_transforms,
                              mask_transforms=mask_transforms,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers)

    test_loader = get_loader(dataset=dataset,
                             data_dir=data_dir,
                             train=False,
                             joint_transforms=test_joint_transforms,
                             image_transforms=test_image_transforms,
                             mask_transforms=mask_transforms,
                             batch_size=1,
                             shuffle=False,
                             num_workers=num_workers)

    # build trainer / evaluator with ignite
    trainer = create_supervised_trainer(model, model_optimizer, loss, device=device)
    measure = MultiThresholdMeasures()
    evaluator = create_supervised_evaluator(model,
                                            metrics={
                                                '': measure,
                                                'pix-acc': Accuracy(measure),
                                                'iou': IoU(measure),
                                                'loss': Loss(loss),
                                                'f1': F1score(measure),
                                                },
                                            device=device)

    # initialize state variable for checkpoint
    state = update_state(model.state_dict(), 0, 0, 0, 0, 0)

    # make ckpt path
    ckpt_root = './ckpt/'
    filename = '{network}_{optimizer}_lr_{lr}_epoch_{epoch}.pth'
    ckpt_path = os.path.join(ckpt_root, filename)

    # execution after every training iteration
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        num_iter = (trainer.state.iteration - 1) % len(train_loader) + 1
        if num_iter % 20 == 0:
            logger.info("Epoch[{}] Iter[{:03d}] Loss: {:.2f}".format(
                trainer.state.epoch, num_iter, trainer.state.output))

    # execution after every training epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        # evaluate on training set
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        logger.info("Training Results - Epoch: {} Avg-loss: {:.3f}\n Pix-acc: {}\n IoU: {}\n F1: {}\n".format(
            trainer.state.epoch, metrics['loss'], str(metrics['pix-acc']), str(metrics['iou']), str(metrics['f1'])))

        # update state
        update_state(weight=model.state_dict(),
                     train_loss=metrics['loss'],
                     val_loss=state['val_loss'],
                     val_pix_acc=state['val_pix_acc'],
                     val_iou=state['val_iou'],
                     val_f1=state['val_f1'])

    # execution after every epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        # evaluate test(validation) set
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        logger.info("Validation Results - Epoch: {} Avg-loss: {:.3f}\n Pix-acc: {}\n IoU: {}\n F1: {}\n".format(
            trainer.state.epoch, metrics['loss'], str(metrics['pix-acc']), str(metrics['iou']), str(metrics['f1'])))

        # update scheduler
        lr_scheduler.step(metrics['loss'])

        # update and save state
        update_state(weight=model.state_dict(),
                     train_loss=state['train_loss'],
                     val_loss=metrics['loss'],
                     val_pix_acc=metrics['pix-acc'],
                     val_iou=metrics['iou'],
                     val_f1=metrics['f1'])

        path = ckpt_path.format(network=networks,
                                optimizer=optimizer,
                                lr=lr,
                                epoch=trainer.state.epoch)
        save_ckpt_file(path, state)

    trainer.run(train_loader, max_epochs=epochs)

def train_without_ignite(model, loss, batch_size, img_size,
        epochs, lr, num_workers, optimizer, logger, gray_image=False, scheduler=None, viz=True):
    import visdom
    from utils.metrics import Accuracy, IoU

    DEFAULT_PORT = 8097
    DEFAULT_HOSTNAME = "http://localhost"
    
    if viz:
        vis = visdom.Visdom(port=DEFAULT_PORT, server=DEFAULT_HOSTNAME)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_loader = {}
    
    joint_transforms = jnt_trnsf.Compose([
        jnt_trnsf.RandomCrop(img_size),
        jnt_trnsf.RandomRotate(5),
        jnt_trnsf.RandomHorizontallyFlip()
    ])

    train_image_transforms = std_trnsf.Compose([
        std_trnsf.ColorJitter(0.05, 0.05, 0.05, 0.05),
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    test_joint_transforms = jnt_trnsf.Compose([
        jnt_trnsf.Safe32Padding()
    ])

    test_image_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    mask_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor()
        ])
    
    data_loader['train'] = get_loader(dataset='figaro',
                              train=True,
                              joint_transforms=joint_transforms,
                              image_transforms=train_image_transforms,
                              mask_transforms=mask_transforms,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              gray_image=gray_image)

    data_loader['test'] = get_loader(dataset='figaro',
                             train=False,
                             joint_transforms=test_joint_transforms,
                             image_transforms=test_image_transforms,
                             mask_transforms=mask_transforms,
                             batch_size=1,
                             shuffle=True,
                             num_workers=num_workers,
                             gray_image=gray_image)
    
    for epoch in range(epochs):
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train(True)
            else:
                prev_grad_state = torch.is_grad_enabled()
                torch.set_grad_enabled(False)
                model.train(False)
            
            running_loss = 0.0
            
            for i, data in enumerate(tqdm(data_loader[phase], file=sys.stdout)):
                if i == len(data_loader[phase]) - 1: break
                data_ = [t.to(device) if isinstance(t, torch.Tensor) else t for t in data]
                
                if gray_image:
                    img, mask, gray = data_
                else:
                    img, mask = data_
                
                model.zero_grad()
                
                pred_mask = model(img)
                
                if gray_image:
                    l = loss(pred_mask, mask, gray)
                else:
                    l = loss(pred_mask, mask)
                
                if phase == 'train':
                    l.backward()
                    optimizer.step()
                
                running_loss += l.item()
            
            epoch_loss = running_loss / len(data_loader[phase])
            
            if phase == 'train':
                logger.info(f"Training Results - Epoch: {epoch} Avg-loss: {epoch_loss:.3f}")
                if viz:
                    vis.images([
                        np.clip(pred_mask.detach().cpu().numpy()[0],0,1),
                        mask.detach().cpu().numpy()[0]
                    ], opts=dict(title=f'pred img for {epoch}-th iter'))
            
            if phase == 'test':
                if viz:
                    vis.images([
                        np.clip(pred_mask.detach().cpu().numpy()[0],0,1),
                        mask.detach().cpu().numpy()[0]
                    ], opts=dict(title=f'pred img for {epoch}-th iter'))
                logger.info(f"Test Results - Epoch: {epoch} Avg-loss: {epoch_loss:.3f}")
                
                if scheduler: scheduler.step(epoch_loss)
            
                torch.set_grad_enabled(prev_grad_state)
  
