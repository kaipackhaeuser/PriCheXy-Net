import os
import random
import pickle
import zipfile
import numpy as np
import pandas as pd
from sklearn import metrics
from statistics import mean
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from datasets.Dataset import Dataset
from datasets.SiameseDataset import SiameseDataset
from datasets.DatasetPretrain import DatasetPretrain


def seed_all(seed):
    """Seeding.

    :param seed: int
        The seed that is used.
    """

    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def get_data_loader(phase='training', experimental_step='pretrain', image_size=256, n_channels=1, batch_size=32, 
                    shuffle=True, num_workers=8, pin_memory=True, b=None, m=None, eps=None, image_path='./'):
    """This function returns a data loader either for the 'training', 'validation', or 'testing' phase for the chest
    X-ray anonymization experiments.

    :param phase: str
        Specifies the phase.
        Training: phase='training'
        Validation: phase='validation'
        Testing: phase='testing'
    :param experimental_step: str
        A string that indicates the experimental step. Options are:
        experimental_step='pretrain'
        experimental_step='anonymization'
        experimental_step='retrainSNN'
    :param image_size: int
        The image size to be used for the respective experiment.
    :param n_channels: int
        Specifies the number of channels for the input images (n_channels=1 or n_channels=3).
    :param batch_size: int
        The batch size that is used for loading and processing the data.
    :param shuffle: bool
        A boolean value that represents whether the data will be shuffled or not.
    :param num_workers: int
        How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    :param pin_memory: bool
        If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    :param b: int
        Specifies the size of the grid cells for DP pixelization.
    :param m: int
        Specifies the m-neighborhood (DP sensitivity factor).
    :param eps: int
        Specifies the DP privacy budget (smaller values indicate greater privacy).
    :param image_path: str
        The path to the folder where the images are stored.
    :return data_loader: torch.utils.data.DataLoader
        The respective data loader that will be used for the planned experiments.
    """

    if experimental_step == 'pretrain':
        data_set = DatasetPretrain(phase=phase, image_size=image_size, n_channels=n_channels, image_path=image_path)
    elif experimental_step == 'anonymization':
        data_set = Dataset(phase=phase, image_size=image_size, n_channels=n_channels, image_path=image_path)
    elif experimental_step == 'retrainSNN':
        data_set = SiameseDataset(phase=phase, n_channels=n_channels, image_size=image_size, b=b, m=m, eps=eps, 
                                  image_path=image_path)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                             pin_memory=pin_memory)
    return data_loader


def pretrain(generator, training_loader, gauss_filter, grid_identity, mu, mse_loss, optimizer_g, epoch, n_epochs,
             show_every_n_epochs, show_every_n_iters, save_path):
    """This function is used to pre-train the flow field generator of the anonymization model.

    :param generator: torch.nn.Module
        The flow field generator that will be pre-trained to obtain the identity.
    :param training_loader: torch.utils.data.DataLoader
        The data loader that is used during training.
    :param gauss_filter: torch.nn.Module (GaussianSmoothing)
        The gaussian filter that is applied on the learned flow field to guarantee smooth image deformations.
    :param grid_identity: torch.Tensor
        The identity grid that would result in the exact same image when using torch's grid_sample() function.
    :param mu: float
        This factor controls the degree of deformation. Larger values allow for more deformation.
        For mu=0, the images will not be deformed and the operation would result in the original images.
    :param mse_loss: torch.nn.Loss
        The MSELoss is used to pre-train the flow field generator.
    :param optimizer_g: torch.optim.Optimizer
        The chosen optimizer used to pre-train the flow field generator.
    :param epoch: int
        The current epoch. Only needed for printing purposes.
    :param n_epochs: int
        The maximum number of epochs for the training/validation loop. Only needed for printing purposes.
    :param show_every_n_epochs: int
        An integer value indicating in which epochs the original and reconstructed images are added to the tensorboard.
    :param show_every_n_iters: int
        An integer value indicating in which iterations the original and reconstructed images are added to the
        tensorboard.
    :param save_path: str
        Path to the folder where the tensorboard files are stored.
    :return loss: float
        Epoch-wise reconstruction loss.
    """

    generator.train()

    list_rec_loss = []

    print('Training----->')
    for i, batch in enumerate(training_loader):
        inputs = batch.cuda()

        if grid_identity is not None:
            # Generate grids
            grids = generator(inputs)

            # Constraints on grids
            grids = grid_identity - mu * grids
            grids = gauss_filter(grids)
            grids = grids.permute(0, 2, 3, 1)

            # Compute reconstructed images by using the original input in conjunction with pixel locations from grids
            outputs = torch.nn.functional.grid_sample(inputs, grids, padding_mode='border', align_corners=True)
        else:
            outputs = generator(inputs)

        if epoch % show_every_n_epochs == 0 and i % show_every_n_iters == 0:
            # Write images to tensorboard
            writer = SummaryWriter(save_path + 'runs/epoch' + str(epoch) + '/')
            img_grid_original = torchvision.utils.make_grid(inputs)
            writer.add_image('training/original_images', img_grid_original, i)
            img_grid_reconstructed = torchvision.utils.make_grid(outputs)
            writer.add_image('training/reconstructed_images', img_grid_reconstructed, i)
            writer.close()

        # Compute reconstruction loss for the generator
        rec_loss = mse_loss(outputs, inputs)

        # Append loss values to respective list
        list_rec_loss.append(rec_loss.item())

        # Optimize the flow field generator
        optimizer_g.zero_grad()
        rec_loss.backward()
        optimizer_g.step()

        # Print information
        print('Epoch [%d/%d], Iteration [%d/%d], MSE Loss (rec_loss): %.4f' % (epoch + 1, n_epochs, i + 1,
                                                                               len(training_loader), rec_loss.item()))
    return mean(list_rec_loss)


def preval(generator, validation_loader, gauss_filter, grid_identity, mu, mse_loss, epoch, n_epochs, 
           show_every_n_epochs, show_every_n_iters, save_path):
    """This function is used to validate the flow field generator of the anonymization model while pre-training.

    :param generator: torch.nn.Module
        The flow field generator that will be pre-trained to obtain the identity.
    :param validation_loader: torch.utils.data.DataLoader
        The data loader that is used for validation.
    :param gauss_filter: torch.nn.Module (GaussianSmoothing)
        The gaussian filter that is applied on the learned flow field to guarantee smooth image deformations.
    :param grid_identity: torch.Tensor
        The identity grid that would result in the exact same image when using torch's grid_sample() function.
    :param mu: float
        This factor controls the degree of deformation. Larger values allow for more deformation.
        For mu=0, the images will not be deformed and the operation would result in the original images.
    :param mse_loss: torch.nn.Loss
        The MSELoss is used to validate the flow field generator while pre-training.
    :param epoch: int
        The current epoch. Only needed for printing purposes.
    :param n_epochs: int
        The maximum number of epochs for the training/validation loop. Only needed for printing purposes.
    :param show_every_n_epochs: int
        An integer value indicating in which epochs the original and reconstructed images are added to the tensorboard.
    :param show_every_n_iters: int
        An integer value indicating in which iterations the original and reconstructed images are added to the
        tensorboard.
    :param save_path: str
        Path to the folder where the tensorboard files are stored.
    :return loss: float
        Epoch-wise reconstruction loss.
    """

    generator.eval()

    list_rec_loss = []

    print('Validation----->')
    with torch.no_grad():
        for i, batch in enumerate(validation_loader):
            inputs = batch.cuda()

            if grid_identity is not None:
                # Generate grids
                grids = generator(inputs)

                # Constraints on grids
                grids = grid_identity - mu * grids
                grids = gauss_filter(grids)
                grids = grids.permute(0, 2, 3, 1)

                # Compute reconstructed images by using the original input in conjunction with pixel locations from grids
                outputs = torch.nn.functional.grid_sample(inputs, grids, padding_mode='border', align_corners=True)
            else:
                outputs = generator(inputs)

            if epoch % show_every_n_epochs == 0 and i % show_every_n_iters == 0:
                # Write images to tensorboard
                writer = SummaryWriter(save_path + 'runs/epoch' + str(epoch) + '/')
                img_grid_original = torchvision.utils.make_grid(inputs)
                writer.add_image('validation/original_images', img_grid_original, i)
                img_grid_reconstructed = torchvision.utils.make_grid(outputs)
                writer.add_image('validation/reconstructed_images', img_grid_reconstructed, i)
                writer.close()

            # Compute reconstruction loss for the generator
            rec_loss = mse_loss(outputs, inputs)

            # Append loss values to respective list
            list_rec_loss.append(rec_loss.item())

            # Print information
            print('Epoch [%d/%d], Iteration [%d/%d], MSE Loss (rec_loss): %.4f' % (epoch + 1, n_epochs, i + 1,
                                                                                   len(validation_loader),
                                                                                   rec_loss.item()))
    return mean(list_rec_loss)


def train(generator, training_loader, gauss_filter, grid_identity, mu, ac_loss, verification_loss, ac_loss_weight, 
          ver_loss_weight, optimizer_g, optimizer_ac, optimizer_ver, criterion_ac, criterion_ver, epoch, n_epochs, 
          show_every_n_epochs, show_every_n_iters, save_path):
    """This function is used to train the entire anonymization model.

    :param generator: torch.nn.Module
        The flow field generator that is used to targetedly deform the chest radiographs.
    :param training_loader: torch.utils.data.DataLoader
        The data loader that is used during training.
    :param gauss_filter: torch.nn.Module (GaussianSmoothing)
        The gaussian filter that is applied on the learned flow field to guarantee smooth image deformations.
    :param grid_identity: torch.Tensor
        The identity grid that would result in the exact same image when using torch's grid_sample() function.
    :param mu: float
        This factor controls the degree of deformation. Larger values allow for more deformation.
        For mu=0, the images will not be deformed and the operation would result in the original images.
    :param ac_loss: torch.nn.Module (ACLoss)
        The auxiliary classifier loss which is intended to be used to ensure that underlying abnormality patterns are
        preserved during the anonymization process.
    :param verification_loss: torch.nn.Module (VerificationLoss)
        The verification loss is intended to be used for learning a flow field that targetedly deforms a chest
        radiograph and thereby obfuscates the underlying biometric information.
    :param ac_loss_weight: int, float
        Weighting factor for the auxiliary classifier loss.
    :param ver_loss_weight: int, float
        Weighting factor for the verification loss.
    :param optimizer_g: torch.optim.Optimizer
        The chosen optimizer used to train the anonymization architecture.
    :param optimizer_ac: torch.optim.Optimizer
        The chosen optimizer used to train the auxiliary classifier.
    :param optimizer_ver: torch.optim.Optimizer
        The chosen optimizer used to train the patient verification model.
    :param criterion_ac: torch.nn.Loss
        The loss to update the auxiliary classifier model.
    :param criterion_ver: torch.nn.Loss
        The loss to update the patient verification model.
    :param epoch: int
        The current epoch. Only needed for printing purposes.
    :param n_epochs: int
        The maximum number of epochs for the training/validation loop. Only needed for printing purposes.
    :param show_every_n_epochs: int
        An integer value indicating in which epochs the original and deformed images are added to the tensorboard.
    :param show_every_n_iters: int
        An integer value indicating in which iterations the original and deformed images are added to the tensorboard.
    :param save_path: str
        Path to the folder where the tensorboard files are stored.
    :return loss_values: list
        Epoch-wise loss values.
    """

    generator.train()

    list_ac_loss = []
    list_ver_loss = []
    list_log_likelihood_ver_loss = []
    list_total_loss = []

    print('Training----->')
    for i, batch in enumerate(training_loader):
        inputs1, inputs2, labels, labels_id = batch
        inputs1, inputs2, labels, labels_id = inputs1.cuda(), inputs2.cuda(), labels.cuda(), labels_id.cuda()

        if grid_identity is not None:
            # Generate grids
            grids = generator(inputs1)

            # Constraints on grids
            grids = grid_identity - mu * grids
            grids = gauss_filter(grids)
            grids = grids.permute(0, 2, 3, 1)

            # Compute deformed images by using original input values in conjunction with pixel locations from grids
            fakes_1 = torch.nn.functional.grid_sample(inputs1, grids, padding_mode='border', align_corners=True)
        else:
            fakes_1 = generator(inputs1)

        if epoch % show_every_n_epochs == 0 and i % show_every_n_iters == 0:
            # Write images to tensorboard
            writer = SummaryWriter(save_path + 'runs/epoch' + str(epoch) + '/')
            img_grid_original = torchvision.utils.make_grid(inputs1)
            writer.add_image('training/original_images', img_grid_original, i)
            img_grid_deformed = torchvision.utils.make_grid(fakes_1)
            writer.add_image('training/deformed_images', img_grid_deformed, i)
            writer.close()

        # Compute AC loss
        ac_loss_value = ac_loss(fakes_1, labels)

        # Compute verification loss
        ver_loss = verification_loss(fakes_1, inputs2)
        log_likelihood_ver_loss = - torch.log(1 - ver_loss)
        ver_loss = ver_loss.mean()
        log_likelihood_ver_loss = log_likelihood_ver_loss.mean()

        # Put auxiliary classifier loss and verification loss together
        total_loss = ac_loss_weight * ac_loss_value + ver_loss_weight * log_likelihood_ver_loss

        # Append loss values to respective lists
        list_ac_loss.append(ac_loss_value.item())
        list_ver_loss.append(ver_loss.item())
        list_log_likelihood_ver_loss.append(log_likelihood_ver_loss.item())
        list_total_loss.append(total_loss.item())

        # Optimize the flow field generator
        optimizer_g.zero_grad()
        total_loss.backward()
        optimizer_g.step()

        # Set loss models to train mode
        verification_loss.verification_model.train()
        ac_loss.ac_model.train()

        inputs1_snn = fakes_1.detach().expand(-1, 3, -1, -1)
        inputs2_snn = inputs2.expand(-1, 3, -1, -1)

        # Apply the ImageNet transform
        resize = transforms.Resize((224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        inputs1_snn = normalize(inputs1_snn)
        inputs2_snn = normalize(inputs2_snn)

        # Zero the parameter gradients
        optimizer_ver.zero_grad()
        optimizer_ac.zero_grad()

        # Optimize patient verification model
        outputs_snn = verification_loss.verification_model(inputs1_snn, inputs2_snn)
        outputs_snn = outputs_snn.squeeze()
        labels_id = labels_id.type_as(outputs_snn)
        loss_ver = criterion_ver(outputs_snn, labels_id)
        loss_ver.backward()
        optimizer_ver.step()
        verification_loss.verification_model.eval()

        # Optimize auxiliary classifier
        inputs_ac = fakes_1.detach().expand(-1, 3, -1, -1)
        inputs_ac = normalize(resize(inputs_ac))
        outputs_ac = ac_loss.ac_model(inputs_ac)
        loss_ac = criterion_ac(outputs_ac, labels)
        loss_ac.backward()
        optimizer_ac.step()
        ac_loss.ac_model.eval()

        # Print information
        print('Epoch [%d/%d], Iteration [%d/%d], Verification Loss (ver_loss): %.4f' % (epoch + 1, n_epochs, i + 1,
                                                                                        len(training_loader),
                                                                                        ver_loss.item()))

    return [mean(list_ac_loss), mean(list_ver_loss), mean(list_log_likelihood_ver_loss), mean(list_total_loss)]


def validate(generator, validation_loader, gauss_filter, grid_identity, mu, ac_loss, verification_loss, ac_loss_weight, 
             ver_loss_weight, epoch, n_epochs, show_every_n_epochs, show_every_n_iters, save_path):
    """This function is used to validate the entire anonymization model.

    :param generator: torch.nn.Module
        The flow field generator that is used to targetedly deform the chest radiographs.
    :param validation_loader: torch.utils.data.DataLoader
        The data loader that is used for validation.
    :param gauss_filter: torch.nn.Module (GaussianSmoothing)
        The gaussian filter that is applied on the learned flow field to guarantee smooth image deformations.
    :param grid_identity: torch.Tensor
        The identity grid that would result in the exact same image when using torch's grid_sample() function.
    :param mu: float
        This factor controls the degree of deformation. Larger values allow for more deformation.
        For mu=0, the images will not be deformed and the operation would result in the original images.
    :param ac_loss: torch.nn.Module (ACLoss)
        The auxiliary classifier loss which is intended to be used to ensure that underlying abnormality patterns are
        preserved during the anonymization process.
    :param verification_loss: torch.nn.Module (VerificationLoss)
        The verification loss is intended to be used for learning a flow field that targetedly deforms a chest
        radiograph and thereby obfuscates the underlying biometric information.
    :param ac_loss_weight: int, float
        Weighting factor for the auxiliary classifier loss.
    :param ver_loss_weight: int, float
        Weighting factor for the verification loss.
    :param epoch: int
        The current epoch. Only needed for printing purposes.
    :param n_epochs: int
        The maximum number of epochs for the training/validation loop. Only needed for printing purposes.
    :param show_every_n_epochs: int
        An integer value indicating in which epochs the original and deformed images are added to the tensorboard.
    :param show_every_n_iters: int
        An integer value indicating in which iterations the original and deformed images are added to the tensorboard.
    :param save_path: str
        Path to the directory where the tensorboard files are stored.
    :return loss_values: list
        Epoch-wise loss values.
    """

    generator.eval()

    list_ac_loss = []
    list_ver_loss = []
    list_log_likelihood_ver_loss = []
    list_total_loss = []

    print('Validation----->')
    with torch.no_grad():
        for i, batch in enumerate(validation_loader):
            inputs1, inputs2, labels, _ = batch
            inputs1, inputs2, labels = inputs1.cuda(), inputs2.cuda(), labels.cuda()

            if grid_identity is not None:
                # Generate grids
                grids = generator(inputs1)

                # Constraints on grids
                grids = grid_identity - mu * grids
                grids = gauss_filter(grids)
                grids = grids.permute(0, 2, 3, 1)

                # Compute deformed images by using original input values in conjunction with pixel locations from grids
                fakes_1 = torch.nn.functional.grid_sample(inputs1, grids, padding_mode='border', align_corners=True)
            else:
                fakes_1 = generator(inputs1)

            if epoch % show_every_n_epochs == 0 and i % show_every_n_iters == 0:
                # Write images to tensorboard
                writer = SummaryWriter(save_path + 'runs/epoch' + str(epoch) + '/')
                img_grid_original = torchvision.utils.make_grid(inputs1)
                writer.add_image('validation/original_images', img_grid_original, i)
                img_grid_deformed = torchvision.utils.make_grid(fakes_1)
                writer.add_image('validation/deformed_images', img_grid_deformed, i)
                writer.close()

            # Compute AC loss
            ac_loss_value = ac_loss(fakes_1, labels)

            # Compute verification loss
            ver_loss = verification_loss(fakes_1, inputs2)
            log_likelihood_ver_loss = - torch.log(1 - ver_loss)
            ver_loss = ver_loss.mean()
            log_likelihood_ver_loss = log_likelihood_ver_loss.mean()

            # Put auxiliary classifier loss and verification loss together
            total_loss = ac_loss_weight * ac_loss_value + ver_loss_weight * log_likelihood_ver_loss

            # Append loss values to respective lists
            list_ac_loss.append(ac_loss_value.item())
            list_ver_loss.append(ver_loss.item())
            list_log_likelihood_ver_loss.append(log_likelihood_ver_loss.item())
            list_total_loss.append(total_loss.item())

            # Print information
            print('Epoch [%d/%d], Iteration [%d/%d], Verification Loss (ver_loss): %.4f' % (epoch + 1, n_epochs, i + 1,
                                                                                            len(validation_loader),
                                                                                            ver_loss.item()))

    return [mean(list_ac_loss), mean(list_ver_loss), mean(list_log_likelihood_ver_loss), mean(list_total_loss)]


def train_snn(perturbation_type, net, perturbation_net, grid_identity, gauss_filter, mu, training_loader, criterion,
              optimizer, epoch, n_epochs):
    """This function is used to re-train the incorporated patient verification architecture with 
    deformed/perturbed/anonymized images.

    :param perturbation_type: str
        Defines the type of perturbation. Options: 'flow_field', 'privacy_net', 'dp_pix', 'none'
    :param net: torch.nn.Module
        The SNN architecture to train.
    :param perturbation_net: torch.nn.Module
        The flow field generator that is used to targetedly deform the chest radiographs.
    :param grid_identity: torch.Tensor
        The identity grid that would result in the exact same image when using torch's grid_sample() function.
    :param gauss_filter: torch.nn.Module (GaussianSmoothing)
        The gaussian filter that is applied on the learned flow field to guarantee smooth image deformations.
    :param mu: float
        This factor controls the degree of deformation. Larger values allow for more deformation.
        For mu=0, the images will not be deformed and the operation would result in the original images.
    :param training_loader: torch.utils.data.DataLoader
        The data loader that is used during training.
    :param criterion: torch.nn.Loss
        The loss criterion that is used to compare the predictions with the ground truth.
    :param optimizer: torch.optim.Optimizer
        The chosen optimizer used to train the SNN architecture.
    :param epoch: int
        The current epoch. Only needed for printing purposes.
    :param n_epochs: int
        The maximum number of epochs for the training/validation loop. Only needed for printing purposes.
    :return training_loss: float
        Epoch-wise training loss for the patient verification task.
    """

    net.train()
    if perturbation_type in ['flow_field', 'privacy_net']:
        perturbation_net.eval()
    running_loss = 0.0

    print('Training----->')
    for i, batch in enumerate(training_loader):
        inputs1, inputs2, labels = batch
        inputs1, inputs2, labels = inputs1.cuda(), inputs2.cuda(), labels.cuda()

        if perturbation_type == 'flow_field':
            # Generate grids, impose constraints, and compute the deformed images
            grid1 = perturbation_net(inputs1)
            grid1 = grid_identity - mu * grid1
            grid1 = gauss_filter(grid1)
            grid1 = grid1.permute(0, 2, 3, 1)
            inputs1 = torch.nn.functional.grid_sample(inputs1, grid1, padding_mode='border', align_corners=True)

            # Generate grids, impose constraints, and compute the deformed images
            grid2 = perturbation_net(inputs2)
            grid2 = grid_identity - mu * grid2
            grid2 = gauss_filter(grid2)
            grid2 = grid2.permute(0, 2, 3, 1)
            inputs2 = torch.nn.functional.grid_sample(inputs2, grid2, padding_mode='border', align_corners=True)

        if perturbation_type == 'privacy_net':
            # Compute Privacy-Net output
            inputs1 = perturbation_net(inputs1)
            inputs2 = perturbation_net(inputs2)

        if perturbation_type in ['flow_field', 'privacy_net', 'dp_pix']:
            # Expand the input tensors
            inputs1 = inputs1.expand(-1, 3, -1, -1)
            inputs2 = inputs2.expand(-1, 3, -1, -1)

            # Apply the ImageNet transform
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            inputs1 = normalize(inputs1)
            inputs2 = normalize(inputs2)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(inputs1, inputs2)
        outputs = outputs.squeeze()
        labels = labels.type_as(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print('Epoch [%d/%d], Iteration [%d/%d], Loss: %.4f' % (epoch + 1, n_epochs, i + 1, len(training_loader),
                                                                loss.item()))

    # Compute the average loss per epoch
    training_loss = running_loss / len(training_loader)
    return training_loss


def validate_snn(perturbation_type, net, perturbation_net, grid_identity, gauss_filter, mu, validation_loader,
                 criterion, epoch, n_epochs):
    """This function is used to validate the incorporated patient verification architecture with
    deformed/perturbed/anonymized images.

    :param perturbation_type: str
        Defines the type of perturbation. Options: 'flow_field', 'privacy_net', 'dp_pix', 'none'
    :param net: torch.nn.Module
        The SNN architecture to validate.
    :param perturbation_net: torch.nn.Module
        The flow field generator that is used to targetedly deform the chest radiographs.
    :param grid_identity: torch.Tensor
        The identity grid that would result in the exact same image when using torch's grid_sample() function.
    :param gauss_filter: torch.nn.Module (GaussianSmoothing)
        The gaussian filter that is applied on the learned flow field to guarantee smooth image deformations.
    :param mu: float
        This factor controls the degree of deformation. Larger values allow for more deformation.
        For mu=0, the images will not be deformed and the operation would result in the original images.
    :param validation_loader: torch.utils.data.DataLoader
        The data loader that is used for validation.
    :param criterion: torch.nn.Loss
        The loss criterion that is used to compare the predictions with the ground truth.
    :param epoch: int
        The current epoch. Only needed for printing purposes.
    :param n_epochs: int
        The maximum number of epochs for the training/validation loop. Only needed for printing purposes.
    :return validation_loss: float
        Epoch-wise validation loss for the patient verification task.
    """

    net.eval()
    if perturbation_type in ['flow_field', 'privacy_net']:
        perturbation_net.eval()
    running_loss = 0.0

    print('Validating----->')
    with torch.no_grad():
        for i, batch in enumerate(validation_loader):
            inputs1, inputs2, labels = batch
            inputs1, inputs2, labels = inputs1.cuda(), inputs2.cuda(), labels.cuda()

            if perturbation_type == 'flow_field':
                # Generate grids, impose constraints, and compute the deformed images
                grid1 = perturbation_net(inputs1)
                grid1 = grid_identity - mu * grid1
                grid1 = gauss_filter(grid1)
                grid1 = grid1.permute(0, 2, 3, 1)
                inputs1 = torch.nn.functional.grid_sample(inputs1, grid1, padding_mode='border', align_corners=True)

                # Generate grids, impose constraints, and compute the deformed images
                grid2 = perturbation_net(inputs2)
                grid2 = grid_identity - mu * grid2
                grid2 = gauss_filter(grid2)
                grid2 = grid2.permute(0, 2, 3, 1)
                inputs2 = torch.nn.functional.grid_sample(inputs2, grid2, padding_mode='border', align_corners=True)

            if perturbation_type == 'privacy_net':
                # Compute Privacy-Net output
                inputs1 = perturbation_net(inputs1)
                inputs2 = perturbation_net(inputs2)

            if perturbation_type in ['flow_field', 'privacy_net', 'dp_pix']:
                # Expand the input tensors
                inputs1 = inputs1.expand(-1, 3, -1, -1)
                inputs2 = inputs2.expand(-1, 3, -1, -1)

                # Apply the ImageNet transform
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                inputs1 = normalize(inputs1)
                inputs2 = normalize(inputs2)

            # Forward
            outputs = net(inputs1, inputs2)
            outputs = outputs.squeeze()
            labels = labels.type_as(outputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            print('Epoch [%d/%d], Iteration [%d/%d], Loss: %.4f' % (epoch + 1, n_epochs, i + 1, len(validation_loader),
                                                                    loss.item()))

    # Compute the average loss per epoch
    validation_loss = running_loss / len(validation_loader)
    return validation_loss


def test_snn(perturbation_type, net, perturbation_net, grid_identity, gauss_filter, mu, test_loader):
    """This function is used to test the incorporated patient verification architecture after re-training with
    deformed/perturbed/anonymized images. This function represents a realistic attack scenario where a real image is
    attempted to be linked to an anonymized image.

    :param perturbation_type: str
        Defines the type of perturbation. Options: 'flow_field', 'privacy_net', 'dp_pix', 'none'
    :param net: torch.nn.Module
        The SNN architecture to test.
    :param perturbation_net: torch.nn.Module
        The flow field generator that is used to targetedly deform the chest radiographs.
    :param grid_identity: torch.Tensor
        The identity grid that would result in the exact same image when using torch's grid_sample() function.
    :param gauss_filter: torch.nn.Module (GaussianSmoothing)
        The gaussian filter that is applied on the learned flow field to guarantee smooth image deformations.
    :param mu: float
        This factor controls the degree of deformation. Larger values allow for more deformation.
        For mu=0, the images will not be deformed and the operation would result in the original images.
    :param test_loader: torch.utils.data.DataLoader
        The data loader that is used for testing.
    :return y_true: torch.Tensor
        The true labels indicating whether the two input images belong to the same patient.
        y_true=0: input images belong to two different patients
        y_true=1: input images belong to the same patient
    :return y_pred: torch.Tensor
        The predicted similarity output score which can take values between 0 and 1. The higher the similarity value is,
        the more likely it is that both images belong to the same patient.
    """

    net.eval()
    if perturbation_type in ['flow_field', 'privacy_net']:
        perturbation_net.eval()
    y_true = None
    y_pred = None

    print('Testing----->')
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs1, inputs2, labels = batch

            if y_true is None:
                y_true = labels
            else:
                y_true = torch.cat((y_true, labels), 0)

            inputs1, inputs2, labels = inputs1.cuda(), inputs2.cuda(), labels.cuda()

            if perturbation_type == 'flow_field':
                # Generate grids, impose constraints, and compute the deformed images
                grid1 = perturbation_net(inputs1)
                grid1 = grid_identity - mu * grid1
                grid1 = gauss_filter(grid1)
                grid1 = grid1.permute(0, 2, 3, 1)
                inputs1 = torch.nn.functional.grid_sample(inputs1, grid1, padding_mode='border', align_corners=True)

            if perturbation_type == 'privacy_net':
                # Compute Privacy-Net output
                inputs1 = perturbation_net(inputs1)

            if perturbation_type in ['flow_field', 'privacy_net', 'dp_pix']:
                # Expand the input tensors
                inputs1 = inputs1.expand(-1, 3, -1, -1)
                inputs2 = inputs2.expand(-1, 3, -1, -1)

                # Apply the ImageNet transform
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                inputs1 = normalize(inputs1)
                inputs2 = normalize(inputs2)

            # Forward
            outputs = net(inputs1, inputs2)
            outputs = torch.sigmoid(outputs)

            if y_pred is None:
                y_pred = outputs.cpu()
            else:
                y_pred = torch.cat((y_pred, outputs.cpu()), 0)

    y_pred = y_pred.squeeze()
    return y_true, y_pred


def make_zip(output_filename, source_dir, config_file):
    """This function creates a .zip folder that contains the complete project structure and all important/necessary
    files that are used for a specific experiment. This may be useful to investigate the code corresponding to a
    conducted experiment at a later point in time again.

    :param output_filename: str
        The name of the file to be produced. Make sure to also provide the correct path.
    :param source_dir: str
        The source directory to be compressed.
    :param config_file: str
        The config file used for the respective experiment. The folder './config_files/' may contain several config
        files. However, only save the one which was actually used.
    """

    rel_root = os.path.abspath(os.path.join(source_dir, os.pardir))
    # exclude specific folders from .zip file generation
    exclude_dirs = {'.git', '.idea', '__pycache__'}
    # exclude specific files from .zip file generation
    exclude_files = {output_filename, '.gitignore'}
    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zip_f:
        for root, dirs, files in os.walk(source_dir):
            # change dirs and files lists in-place according to the above specified exclude_dirs and exclude_files
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            files[:] = [f for f in files if f not in exclude_files]
            # add directory (needed for empty dirs)
            zip_f.write(root, os.path.relpath(root, rel_root))
            # do not save the contents of the archive folder
            if os.path.relpath(root, source_dir) == 'archive':
                dirs[:] = []
                files[:] = []
            # save only the config file that was used for the respective experiment
            if os.path.relpath(root, source_dir) == 'config_files':
                files[:] = [config_file]
            # write the files to the zip folder
            for file in files:
                filename = os.path.join(root, file)
                arc_name = os.path.join(os.path.relpath(root, rel_root), file)
                zip_f.write(filename, arc_name)


def apply_threshold(input_tensor, threshold):
    """This function is used to apply a threshold to a tensor.

    :param input_tensor: numpy.ndarray or torch.Tensor
        The input tensor to be thresholded.
    :param threshold: float
        The threshold which is applied to the input tensor.
    :return output: numpy.ndarray
        The binary output which only contains zeros and ones.
    """

    output = np.where(input_tensor > threshold, np.ones(input_tensor.shape), np.zeros(input_tensor.shape))
    return output


def bootstrap(n_bootstraps, y_true, y_pred, path, experiment_description):
    """This function represents bootstrapping which can be used to get the mean AUC value and the 95% confidence
    interval.

    :param n_bootstraps: int
        The number of bootstrap runs.
    :param y_true: numpy.ndarray
        The true labels which can take the values 0 or 1.
        y_true=0: Input images belong to two different patients.
        y_true=1: Input images belong to the same patient.
    :param y_pred: numpy.ndarray
        The predicted similarity output scores that can take values between 0 and 1. High values indicate a higher
        probability that the two input images belong to the same patient.
    :param path: str
        The path where the bootstrapping results should be saved.
    :param experiment_description: str
        The experiment description tag.
    :return auc_mean: float
        The mean AUC value.
    :return confidence_lower: float
        The lower bound of the 95% confidence interval.
    :return confidence_upper: float
        The upper bound of the 95% confidence interval.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    bootstrapped_scores = []

    f = open(path + experiment_description + '_AUC_bootstrapped.txt', "w+")
    f.write('AUC_bootstrapped\n')

    for i in range(n_bootstraps):
        indices = np.random.randint(0, len(y_pred) - 1, len(y_pred))
        auc = metrics.roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(auc)
        f.write(str(auc) + '\n')
    f.close()
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    auc_mean = np.mean(sorted_scores)
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    f = open(path + experiment_description + '_AUC_confidence.txt', "w+")
    f.write('AUC_mean: %s\n' % auc_mean)
    f.write('Confidence interval for the AUC score: ' + str(confidence_lower) + ' - ' + str(confidence_upper))
    f.close()
    return auc_mean, confidence_lower, confidence_upper


def bootstrap_abnormalities(path, n_bootstraps=1000, labels='./chexnet/nih_labels.csv', predictions='preds.csv'):
    """This function is used to compute the 95% confidence intervals for the abnormality classification experiments.

    :param path: str
        The path where the prediction file is stored.
    :param n_bootstraps: int
        The number of bootstrap runs.
    :param labels: str
        The .csv file storing all the labels.
    :param predictions: str
        The .csv file storing all the predictions.
    :return auc_mean: float
        The mean AUC value.
    :return confidence_lower: float
        The lower bound of the 95% confidence interval.
    :return confidence_upper: float
        The upper bound of the 95% confidence interval.
    """

    predictions = pd.read_csv(path + predictions).values
    labels = pd.read_csv(labels).values
    labels = labels[labels[:, 20] == 'test']
    labels = np.delete(labels, [1, 2, 3, 4, 5, 20], 1)

    i = [0, 8, 1, 3, 5, 6, 7, 11, 9, 14, 13, 2, 12, 10, 4]
    labels = labels[:, i]

    np.random.seed(42)
    scores = []

    for i in range(n_bootstraps):
        indices = np.random.randint(0, len(labels) - 1, len(labels))
        auc1 = metrics.roc_auc_score(labels[indices, 1].astype(float), predictions[indices, 1].astype(float))
        auc2 = metrics.roc_auc_score(labels[indices, 2].astype(float), predictions[indices, 2].astype(float))
        auc3 = metrics.roc_auc_score(labels[indices, 3].astype(float), predictions[indices, 3].astype(float))
        auc4 = metrics.roc_auc_score(labels[indices, 4].astype(float), predictions[indices, 4].astype(float))
        auc5 = metrics.roc_auc_score(labels[indices, 5].astype(float), predictions[indices, 5].astype(float))
        auc6 = metrics.roc_auc_score(labels[indices, 6].astype(float), predictions[indices, 6].astype(float))
        auc7 = metrics.roc_auc_score(labels[indices, 7].astype(float), predictions[indices, 7].astype(float))
        auc8 = metrics.roc_auc_score(labels[indices, 8].astype(float), predictions[indices, 8].astype(float))
        auc9 = metrics.roc_auc_score(labels[indices, 9].astype(float), predictions[indices, 9].astype(float))
        auc10 = metrics.roc_auc_score(labels[indices, 10].astype(float), predictions[indices, 10].astype(float))
        auc11 = metrics.roc_auc_score(labels[indices, 11].astype(float), predictions[indices, 11].astype(float))
        auc12 = metrics.roc_auc_score(labels[indices, 12].astype(float), predictions[indices, 12].astype(float))
        auc13 = metrics.roc_auc_score(labels[indices, 13].astype(float), predictions[indices, 13].astype(float))
        auc14 = metrics.roc_auc_score(labels[indices, 14].astype(float), predictions[indices, 14].astype(float))

        aucs = np.array([auc1, auc2, auc3, auc4, auc5, auc6, auc7, auc8, auc9, auc10, auc11, auc12, auc13, auc14])
        mean_auc = np.mean(aucs)
        scores.append(mean_auc)

    scores = np.array(scores)
    mean_score = np.mean(scores)

    scores.sort()
    confidence_lower = scores[int(0.025 * len(scores))]
    confidence_upper = scores[int(0.975 * len(scores))]

    return mean_score, confidence_lower, confidence_upper


def get_evaluation_metrics(y_true, y_pred):
    """Given the true labels and the predicted values, this function computes some standard evaluation metrics for the
    patient verification task.

    :param y_true: numpy.ndarray
        The true labels which can take the values 0 or 1.
        y_true=0: Input images belong to two different patients.
        y_true=1: Input images belong to the same patient.
    :param y_pred: numpy.ndarray
        The predicted similarity output scores that can take values between 0 and 1. High values indicate a higher
        probability that the two input images belong to the same patient.
    :return accuracy: float
        The classification accuracy: (TP + TN) / (TP + TN + FP + FN)
    :return f1_score: float
        The F1- score: F1 = 2 * (precision * recall) / (precision + recall)
    :return precision: float
        The precision value: TP / (TP + FP)
    :return recall: float
        The recall value: TP / (TP + FN)
    :return report: string / dict
        The classification report showing the main classification metrics.
    :return confusion_matrix: numpy.ndarray
        The resulting confusion matrix containing the total amount of TPs, TNs, FPs and FNs.
    """

    accuracy = metrics.accuracy_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    report = metrics.classification_report(y_true, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    return accuracy, f1_score, precision, recall, report, confusion_matrix


def compute_meanAUC(path):
    """This function computes the AUC (mean +- std) over multiple training/testing runs.

    :param path: str
        A folder that should contain the result .txt files for the individual training/testing runs.
    :return mean_result: float
        The resulting mean value.
    :return std_result: float
        The resulting std value.
    """

    list_dir = os.listdir(path)
    results = []

    for file in list_dir:
        result = np.genfromtxt(path + file, dtype=str, skip_footer=12)
        results.append(result[1].astype(np.float64))

    results = np.array(results)
    mean_result = round(np.mean(results) * 100, 1)
    std_result = round(np.std(results) * 100, 1)
    return mean_result, std_result


def save_labels_predictions(y_true, y_pred, y_pred_thresh, path, experiment_description):
    """This function saves the true labels, the predicted values, and the thresholded values to a .txt file.

    :param y_true: numpy.ndarray
        The true labels which can take the values 0 or 1.
        y_true=0: Input images belong to two different patients.
        y_true=1: Input images belong to the same patient.
    :param y_pred: numpy.ndarray
        The predicted similarity output scores that can take values between 0 and 1. High values indicate a higher
        probability that the two input images belong to the same patient.
    :param y_pred_thresh: numpy.ndarray
        The thresholded prediction scores. Values above a chosen threshold are set to 1, otherwise to 0.
    :param path: str
        The path to the directory where the .txt file should be saved.
    :param experiment_description: str
        The experiment description tag.
    """

    f = open(path + experiment_description + '_labels_predictions.txt', "w+")
    f.write('Label\tPrediction\tPredictionThresholded\n')
    for i in range(len(y_true)):
        f.write(str(y_true[i]) + '\t' + str(y_pred[i]) + '\t' + str(y_pred_thresh[i]) + '\n')
    f.close()


def save_results_to_file(auc, accuracy, f1_score, precision, recall, report, confusion_matrix, path, 
                         experiment_description):
    """This is a function that saves the evaluation metrics to a .txt file.

    :param auc: float
        The area under the receiver operating characteristic (ROC) curve.
    :param accuracy: float
        The classification accuracy: (TP + TN) / (TP + TN + FP + FN)
    :param f1_score: float
        The F1- score: F1 = 2 * (precision * recall) / (precision + recall)
    :param precision: float
        The precision value: TP / (TP + FP)
    :param recall: float
        The recall value: TP / (TP + FN)
    :param report: string / dict
        The classification report showing the main classification metrics.
    :param confusion_matrix: numpy.ndarray
        The confusion matrix containing the total amount of TPs, TNs, FPs and FNs.
    :param path: str
        The path to the directory where the .txt file should be saved.
    :param experiment_description: str
        The experiment description tag.
    """

    f = open(path + experiment_description + '_results.txt', "w+")
    if auc is not None:
        f.write('AUC: %s\n' % auc)
    f.write('Accuracy: %s\n' % accuracy)
    f.write('F1-Score: %s\n' % f1_score)
    f.write('Precision: %s\n' % precision)
    f.write('Recall: %s\n' % recall)
    f.write('Classification report: %s\n' % report)
    f.write('Confusion matrix: %s\n' % confusion_matrix)
    f.close()


def save_roc_metrics_to_file(fp_rates, tp_rates, thresholds, path, experiment_description):
    """This function saves the ROC metrics to a .txt file. Used in experimental step 3.

    :param fp_rates: numpy.ndarray
        The FP rates.
    :param tp_rates: numpy.ndarray
        The TP rates.
    :param thresholds: numpy.ndarray
        The corresponding threshold values.
    :param path: str
        The path to the directory where the .txt file should be saved.
    :param experiment_description: str
        The description of the corresponding experiment.
    """

    f = open(path + experiment_description + '_ROC_metrics.txt', "w+")
    f.write('FP_rate\tTP_rate\tThreshold\n')
    for i in range(len(fp_rates)):
        f.write(str(fp_rates[i]) + '\t' + str(tp_rates[i]) + '\t' + str(thresholds[i]) + '\n')
    f.close()


def plot_roc_curve(fp_rates, tp_rates, path, experiment_description):
    """This is a function that plots and saves the ROC curve. Used in experimental step 3.

    :param fp_rates: numpy.ndarray
        The FP rates.
    :param tp_rates: numpy.ndarray
        The TP rates.
    :param path: str
        The path to the directory where the ROC curve should be saved.
    :param experiment_description: str
        The description of the corresponding experiment.
    """

    plt.figure()
    plt.plot(fp_rates, tp_rates, label='ROC Curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(path + experiment_description + '_ROC_curve.png')


def append_losses_to_dict(loss_dict, phase, losses):
    """This function appends loss values in a specific phase to the corresponding loss dict.

    :param loss_dict: dict
        The dictionary that stores all the loss values.
    :param phase: str
        The current phase (e.g. 'training' or 'validation').
    :param losses: list
        The list which contains the different loss values. Each value in the list is the mean over all iterations of one
        epoch.
    :return loss_dict: dict
        The dictionary that stores all the loss values.
    """

    loss_dict[phase]['ac_loss'].append(losses[0])
    loss_dict[phase]['ver_loss'].append(losses[1])
    loss_dict[phase]['log_likelihood_ver_loss'].append(losses[2])
    loss_dict[phase]['total_loss'].append(losses[3])
    return loss_dict


def show_loss_curves(loss_dict, pre_train=True, save_fig=True, show_fig=False, path='./'):
    """This function is used to plot specific loss curves.

    :param loss_dict: dict
        The loss dict that stores the loss values.
    :param pre_train: bool
        Boolean value to indicate whether or not the loss dict belongs to a pre-training experiment.
    :param save_fig: bool
        Boolean value to indicate whether or not to save the plot.
    :param show_fig: bool
        Boolean value to indicate whether or not to show the plot.
    :param path: str
        The path to the directory where the plot will be saved.
    """

    if pre_train is True:
        plt.figure()
        plt.plot(range(1, len(loss_dict['training']) + 1), loss_dict['training'], label='training')
        plt.plot(range(1, len(loss_dict['validation']) + 1), loss_dict['validation'], label='validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and validation loss curves')
        plt.legend()
        if save_fig is True:
            plt.savefig(path + 'loss_curves.png')
        if show_fig is True:
            plt.show()
        plt.close()
    else:
        for phase in ['training', 'validation']:
            plt.figure()
            plt.plot(range(1, len(loss_dict[phase]['ac_loss']) + 1), loss_dict[phase]['ac_loss'], label='ac_loss')
            plt.plot(range(1, len(loss_dict[phase]['ver_loss']) + 1), loss_dict[phase]['ver_loss'], label='ver_loss')
            plt.plot(range(1, len(loss_dict[phase]['log_likelihood_ver_loss']) + 1),
                     loss_dict[phase]['log_likelihood_ver_loss'], label='log_likelihood_ver_loss')
            plt.plot(range(1, len(loss_dict[phase]['total_loss']) + 1),
                     loss_dict[phase]['total_loss'], label='total_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            if phase == 'training':
                plt.title('Training loss curves')
            elif phase == 'validation':
                plt.title('Validation loss curves')
            plt.legend()
            if save_fig is True:
                plt.savefig(path + 'loss_curves_' + phase + '.png')
            if show_fig is True:
                plt.show()
            plt.close()


def save_loss_dict(loss_dict, save_path):
    """This function saves a loss dict as a pickle file.

    :param loss_dict: dict
        The loss dictionary that is to be saved.
    :param save_path: str
        The path to the directory where the .pkl file should be stored.
    """

    file = open(save_path + "loss_dict.pkl", "wb")
    pickle.dump(loss_dict, file)
    file.close()


def load_loss_dict(file_name):
    """This function loads a loss dict from a pickle file.

    :param file_name: str
        The filename of the pickle file that stores the loss values.
    :return output: dict
        The loss dict extracted from the pickle file.
    """

    file = open(file_name, "rb")
    output = pickle.load(file)
    file.close()
    return output


def save_loss_curves_snn(loss_dict, path, experiment_description):
    """This function saves the training and validation loss values to a .txt file. Used in experimental step 3.

    :param loss_dict: dict
        The loss dict that stores the training loss values as well as the validation loss values.
        loss_dict['training'] contains the loss values for training.
        loss_dict['validation'] contains the loss values for validation.
    :param path: str
        The path to the directory where the .txt file should be saved.
    :param experiment_description: str
        The experiment description tag.
    """

    f = open(path + experiment_description + '_loss_values.txt', "w+")
    f.write('TrainingLoss\tValidationLoss\n')
    for i in range(len(loss_dict['training'])):
        f.write(str(loss_dict['training'][i]) + '\t' + str(loss_dict['validation'][i]) + '\n')
    f.close()


def plot_loss_curves_snn(loss_dict, path, experiment_description):
    """This is a function that plots the loss curves and saves them to a .png file.

    :param loss_dict: dict
        The loss dict that stores the training loss values as well as the validation loss values.
        loss_dict['training'] contains the loss values for training.
        loss_dict['validation'] contains the loss values for validation.
    :param path: str
        The path to the directory where the .png image should be saved.
    :param experiment_description: str
        The experiment description tag.
    """

    plt.figure()
    plt.plot(range(1, len(loss_dict['training']) + 1), loss_dict['training'], label='Training Loss')
    plt.plot(range(1, len(loss_dict['validation']) + 1), loss_dict['validation'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss curves')
    plt.legend()
    plt.savefig(path + experiment_description + '_loss_curves.png')


def dp_pix(image_tensor, b, m, eps, plot=False):
    """This function applies differentially private pixelization to an input tensor.
    Implemented according to: Liyue Fan - "Image Pixelization with Differential Privacy" (2018)
    https://hal.inria.fr/hal-01954420/document

    :param image_tensor: torch.Tensor
        The input image tensor that will be pixelized using a differentially private mechanism.
        This function expects image_tensor to be in range [0, 1] (no ImageNet normalization applied).
    :param b: int
        Specifies the size of the grid cells for DP pixelization.
    :param m: int
        Specifies the m-neighborhood (DP sensitivity factor).
    :param eps: int
        Specifies the DP privacy budget (smaller values indicate greater privacy).
    :param plot: bool
        If true, the pixelized image(s) will be plotted.
    :return patches_orig: torch.Tensor
        The image tensor after applying DP-Pix.
    """

    kc, kh, kw = image_tensor.shape[1], b, b  # kernel size
    dc, dh, dw = image_tensor.shape[1], b, b  # stride

    # create patches
    patches = image_tensor.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = patches.shape
    patches = patches.contiguous().view(-1, kc, kh, kw)

    # compute mean for each patch
    patches = torch.mean(patches, dim=(2, 3), keepdim=True)

    # add laplace noise
    laplace = torch.distributions.laplace.Laplace(0, m / (b * b * eps))
    noise = laplace.sample((patches.shape[0], kc, 1, 1)).cuda()
    patches = torch.add(patches, noise).expand(patches.shape[0], kc, b, b)
    patches = torch.clip(patches, min=0, max=1)

    # reshape back
    patches_orig = patches.view(unfold_shape)
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    patches_orig = patches_orig.view(image_tensor.shape[0], output_c, output_h, output_w)

    if plot:
        for i in range(patches_orig.shape[0]):
            out = patches_orig[i].squeeze(0).squeeze(0).numpy()
            plt.imshow(out, cmap='gray')
            plt.show()

    return patches_orig
