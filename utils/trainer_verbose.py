import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data import get_loader
from utils import update_state, save_ckpt_file
from utils import joint_transforms as jnt_trnsf
from utils import summarize_model
from networks import get_network

import torch
import torchvision.transforms as std_trnsf


def description2num_class(d):
    if d == 'binary_class':
        return 1
    return 7


def get_optimizer(string, model, lr, momentum):
    string = string.lower()
    if string == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(momentum, 0.999))
    elif string == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    raise ValueError


def get_scheduler(string, optimizer):
    string = string.lower()
    if string == 'reducelronplateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    raise ValueError


# for torch-igniter
def train_with_ignite(networks, scheduler, batch_size, description, img_size,
        epochs, lr, momentum,  num_workers, optimizer, use_pretrained, logger):

    from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
    from ignite.metrics import Loss
    from utils.metrics import Accuracy, MeanIU

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_class = description2num_class(description)

    # building model
    model = get_network(networks, num_class)
    
    # log model summary
    input_size = (3, img_size, img_size)
    summarize_model(model.to(device), input_size, logger, batch_size, device)

    # TODO: these should be selectable
    loss = torch.nn.BCEWithLogitsLoss()

    model_optimizer = get_optimizer(optimizer, model, lr, momentum)
    lr_scheduler = get_scheduler(scheduler, model_optimizer)

    joint_transforms = jnt_trnsf.Compose([
        jnt_trnsf.Resize(img_size),
        jnt_trnsf.RandomRotate(5),
        jnt_trnsf.RandomHorizontallyFlip()
    ])

    train_image_transforms = std_trnsf.Compose([
        std_trnsf.ColorJitter(0.05, 0.05, 0.05, 0.05),
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_image_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    mask_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor()
        ])

    train_loader = get_loader(dataset='figaro',
                              train=True,
                              joint_transforms=joint_transforms,
                              image_transforms=train_image_transforms,
                              mask_transforms=mask_transforms,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers)

    test_loader = get_loader(dataset='figaro',
                             train=False,
                             joint_transforms=joint_transforms,
                             image_transforms=test_image_transforms,
                             mask_transforms=mask_transforms,
                             batch_size=1,
                             shuffle=False,
                             num_workers=num_workers)

    trainer = create_supervised_trainer(model, model_optimizer, loss, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={
                                                'pix-acc': Accuracy(),
                                                'mean-iu': MeanIU(0.5),
                                                'loss': Loss(loss)
                                                },
                                            device=device)

    # initialize state variable
    state = update_state(model.state_dict(), 0, 0, 0, 0)

    # make ckpt path
    ckpt_root = './ckpt/'
    filename = '{network}_{optimizer}_lr_{lr}_epoch_{epoch}.pth'
    ckpt_path = os.path.join(ckpt_root, filename)


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        num_iter = (trainer.state.iteration - 1) % len(train_loader) + 1
        if num_iter % 20 == 0:
            logger.info("Epoch[{}] Iter[{:03d}] Loss: {:.2f}".format(
                trainer.state.epoch, num_iter, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        # evaluate training set
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        logger.info("Training Results - Epoch: {}  Pix-acc: {:.3f} MeanIU: {:.3f} Avg-loss: {:.3f}".format(
            trainer.state.epoch, metrics['pix-acc'], metrics['mean-iu'], metrics['loss']))

        # update state
        update_state(model.state_dict(), metrics['loss'], state['val_loss'], state['val_pix_acc'], state['val_miu'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        # evaluate test(validation) set
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        logger.info("Validation Results - Epoch: {}  Pix-acc: {:.2f} MeanIU: {:.3f} Avg-loss: {:.2f}".format(
            trainer.state.epoch, metrics['pix-acc'], metrics['mean-iu'], metrics['loss']))

        # update scheduler
        lr_scheduler.step(metrics['loss'])

        # update and save state
        update_state(model.state_dict(), state['train_loss'], metrics['loss'], metrics['pix-acc'], metrics['mean-iu'])
        save_ckpt_file(ckpt_path.format(network=network, 
                                        optimizer=model_optimizer, 
                                        lr=lr, 
                                        epoch=trainer.state.epoch),
                       state)

    trainer.run(train_loader, max_epochs=epochs)


# 

## pytorch-torchsummary
## pytorch-visdom
## etc
