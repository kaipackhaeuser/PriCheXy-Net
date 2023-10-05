from __future__ import print_function, division

import os
import csv
import time
from shutil import rmtree

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import transforms
from torch.autograd import Variable

import chexnet.eval_model as E
import chexnet.cxr_dataset as CXR


use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
print("Available GPU count:" + str(gpu_count))


def checkpoint(model, best_loss, epoch, LR, save_path):
    """Saves checkpoint of torchvision model during training.

    :param model: torch.nn.Module 
        Classification model to be saved.
    :param best_loss: float 
        Best validation loss achieved so far during training.
    :param epoch: int 
        Current training epoch.
    :param LR: float
        Current learning rate.
    :param save_path: str
        Path to the folder where the model file is stored.
    """

    print('saving')
    state = {
        'model': model,
        'best_loss': best_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR
    }

    torch.save(state, save_path + 'checkpoint')


def train_model(model, criterion, optimizer, LR, num_epochs, dataloaders, dataset_sizes, weight_decay, SAVE_PATH):
    """Fine tunes torchvision model to NIH CXR data.

    :param model: torch.nn.Module 
        Classification model to be trained (DenseNet-121).
    :param criterion: torch.nn.Loss 
        The loss criterion (binary cross entropy loss, BCELoss).
    :param optimizer: torch.optim.Optimizer 
        The optimizer to use during training (SGD).
    :param LR: float 
        Current learning rate.
    :param num_epochs: int
        Continue training up to this many epochs.
    :param dataloaders: dict (torch.utils.data.DataLoader)
        Train and val dataloaders.
    :param dataset_sizes: int
        The size of train and val datasets.
    :param weight_decay: float
        The weight decay parameter for SGD.
    :return SAVE_PATH: str
        The folder where the model checkpoint is stored.
    :return model: torch.nn.Module
        Trained classification model.
    :return best_epoch: int
        Epoch on which best model val loss was obtained.
    """
    
    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or val
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader
            for data in dataloaders[phase]:
                i += 1
                inputs, labels, _ = data
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda()).float()

                outputs = model(inputs)

                # calculate gradient and update parameters in train phase
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data * batch_size

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(epoch, epoch_loss, dataset_sizes[phase]))

            # decay learning rate if no val loss improvement in this epoch
            if phase == 'val' and epoch_loss > best_loss:
                print("decay loss from " + str(LR) + " to " + str(LR / 10) + " as not seeing improvement in val loss")
                LR = LR / 10
                # create new optimizer with lower learning rate
                optimizer = optim.SGD(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=LR,
                    momentum=0.9,
                    weight_decay=weight_decay)
                print("created new optimizer with LR " + str(LR))

            # checkpoint model if has best val loss yet
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                checkpoint(model, best_loss, epoch, LR, SAVE_PATH)

            # log training and validation loss over each epoch
            if phase == 'val':
                with open(SAVE_PATH + "log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            print("no improvement in 3 epochs, break")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    checkpoint_best = torch.load(SAVE_PATH + 'checkpoint')
    model = checkpoint_best['model']

    return model, best_epoch


def train_cnn(PATH_TO_IMAGES, SAVE_PATH, LR, WEIGHT_DECAY):
    """Train torchvision model to NIH data given high level hyperparameters.

    :param PATH_TO_IMAGES: str
        Path to the folder where the images are stored.
    :param SAVE_PATH: str
        Path to the folder where the model file and the results are stored.
    :param LR: float 
        Current learning rate.
    :param WEIGHT_DECAY: float 
        The weight decay parameter for SGD.
    :return preds: pandas.DataFrame 
        Model predictions on test fold with ground truth for comparison.
    :return aucs: pandas.DataFrame
        Resulting AUC values for each abnormality.
    """

    NUM_EPOCHS = 100
    BATCH_SIZE = 16

    try:
        rmtree(SAVE_PATH)
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs(SAVE_PATH)

    # use imagenet mean and std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    N_LABELS = 14  # we are predicting 14 labels

    # define torchvision transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            # because resize doesn't always give 224 x 224, this ensures 224 x 224
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    # create train/val dataloaders
    transformed_datasets = {}
    transformed_datasets['train'] = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='train',
        transform=data_transforms['train'])
    transformed_datasets['val'] = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='val',
        transform=data_transforms['val'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8)

    if not use_gpu:
        raise ValueError("Error, requires GPU")
    
    model = models.densenet121(weights='DEFAULT')
    num_ftrs = model.classifier.in_features
    # add final layer with # outputs in same dimension of labels with sigmoid activation
    model.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    # put model on GPU
    model = model.cuda()

    # define criterion, optimizer for training
    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    # train model
    model, best_epoch = train_model(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS, dataloaders=dataloaders,
                                    dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY, SAVE_PATH=SAVE_PATH)

    # get preds and AUCs on test fold
    preds, aucs = E.make_pred_multilabel(data_transforms, model, PATH_TO_IMAGES, SAVE_PATH)

    return preds, aucs
