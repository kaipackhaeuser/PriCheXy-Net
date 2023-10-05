import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils import utils
from utils.ACLoss import ACLoss
from utils.VerificationLoss import VerificationLoss
from utils.GaussianSmoothing import GaussianSmoothing

from networks.UNet_PriCheXyNet import UNet
from networks.UNet_PrivacyNet import Unet2D_encoder
from networks.SiameseNetwork import SiameseNetwork


class Agent:
    def __init__(self, config):
        """This is the agent that provides code for training and validating the anonymization model.

        :param config: dict
            A dictionary that stores the hyper-parameter configuration and some other important variables.
        """

        self.config = config

        # Set path used to save experiment-related files and results
        self.SAVINGS_PATH = './archive/' + self.config['experiment_description'] + '/'
        self.IMAGE_PATH = self.config['image_path']

        # Reproducibility
        utils.seed_all(42)

        # Set all the important variables
        self.ac_loss_weight = self.config['ac_loss_weight']
        self.ver_loss_weight = self.config['ver_loss_weight']

        self.generator_type = self.config['generator_type']
        self.mu = self.config['mu']

        self.image_size = self.config['image_size']
        self.batch_size = self.config['batch_size']
        self.learning_rate = self.config['learning_rate']
        self.max_epochs = self.config['max_epochs']

        self.num_workers = 8
        self.pin_memory = True

        self.show_every_n_epochs = self.config['show_every_n_epochs']
        self.show_every_n_iterations = self.config['show_every_n_iterations']

        self.writer = SummaryWriter(self.SAVINGS_PATH + 'runs/')

        if self.generator_type == 'flow_field':
            # Define the identity grid
            d = torch.linspace(-1, 1, self.image_size)
            mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
            grid_identity = torch.stack((mesh_y, mesh_x), 2)
            self.grid_identity = grid_identity.unsqueeze(0).permute(0, 3, 1, 2).cuda()
            # Define the Gauss filter which is used for smoothing the resulting flow field
            self.gauss_filter = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).cuda()
            # Define the flow field generator
            self.generator = UNet(1, 2, 32).cuda()
            self.generator.load_state_dict(torch.load('./networks/pretrained_generator_prichexy_net.pth'))
        elif self.generator_type == 'privacy_net':
            # Set identity grid and gauss filter to None
            self.grid_identity = None
            self.gauss_filter = None
            # Define PrivacyNet encoder
            self.generator = Unet2D_encoder(1, 1, 16).cuda()
            self.generator.load_state_dict(torch.load('./networks/pretrained_generator_privacy_net.pth'))
        else:
            raise Exception('Invalid argument: ' + self.generator_type)

        self.start_epoch = 0
        self.lowest_ac_loss = 10000
        self.lowest_ver_loss = 10000
        self.lowest_total_loss = 10000
        self.loss_dict = {
            'training': {
                'ac_loss': [],
                'ver_loss': [],
                'log_likelihood_ver_loss': [],
                'total_loss': []
            },
            'validation': {
                'ac_loss': [],
                'ver_loss': [],
                'log_likelihood_ver_loss': [],
                'total_loss': []
            }
        }

        # Define the auxiliary classifier (self.ac_model is already on GPU)
        self.ac_model = torch.load('./networks/pretrained_classifier.pth')['model']
        self.ac_loss = ACLoss(ac_model=self.ac_model).cuda()

        # Define the verification model and the verification loss
        self.verification_model = SiameseNetwork().cuda()
        self.verification_model.load_state_dict(torch.load('./networks/pretrained_verification_model.pth'))
        self.verification_loss = VerificationLoss(verification_model=self.verification_model, reduction='none').cuda()

        # Loss functions for the auxiliary classifier and the verification model
        self.criterion_ac = nn.BCELoss().cuda()
        self.criterion_ver = nn.BCEWithLogitsLoss().cuda()

        # Set the optimizer functions
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.optimizer_ver = optim.Adam(self.verification_loss.verification_model.parameters(), lr=self.learning_rate)
        self.optimizer_ac = optim.SGD(filter(lambda p: p.requires_grad, self.ac_loss.ac_model.parameters()),
                                      lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)

        # Initialize data loaders
        self.training_loader = utils.get_data_loader(phase='training', experimental_step='anonymization', 
                                                     image_size=self.image_size, n_channels=1, 
                                                     batch_size=self.batch_size, shuffle=True, 
                                                     num_workers=self.num_workers, pin_memory=self.pin_memory, 
                                                     image_path=self.IMAGE_PATH)
        self.validation_loader = utils.get_data_loader(phase='validation', experimental_step='anonymization', 
                                                       image_size=self.image_size, n_channels=1, 
                                                       batch_size=self.batch_size, shuffle=False, 
                                                       num_workers=self.num_workers, pin_memory=self.pin_memory,
                                                       image_path=self.IMAGE_PATH)

    def training_validation(self):
        # Training and validation loop
        for epoch in range(self.start_epoch, self.max_epochs):
            start_time = time.time()

            # Train the anonymization model
            train_losses = utils.train(self.generator, self.training_loader, self.gauss_filter, self.grid_identity, 
                                       self.mu, self.ac_loss, self.verification_loss, self.ac_loss_weight, 
                                       self.ver_loss_weight, self.optimizer_g, self.optimizer_ac, self.optimizer_ver, 
                                       self.criterion_ac, self.criterion_ver, epoch, self.max_epochs, 
                                       self.show_every_n_epochs, self.show_every_n_iterations, self.SAVINGS_PATH)

            # Validate the anonymization model
            val_losses = utils.validate(self.generator, self.validation_loader, self.gauss_filter, self.grid_identity, 
                                        self.mu, self.ac_loss, self.verification_loss, self.ac_loss_weight, 
                                        self.ver_loss_weight, epoch, self.max_epochs, self.show_every_n_epochs, 
                                        self.show_every_n_iterations, self.SAVINGS_PATH)

            end_time = time.time()
            print('Time elapsed for epoch ' + str(epoch + 1) + ': ' + str(
                round((end_time - start_time) / 60, 2)) + ' minutes')

            # Append losses to dict
            utils.append_losses_to_dict(self.loss_dict, 'training', train_losses)
            utils.append_losses_to_dict(self.loss_dict, 'validation', val_losses)

            # Plot loss curves
            utils.show_loss_curves(self.loss_dict, pre_train=False, save_fig=True, show_fig=False, path=self.SAVINGS_PATH)

            # Save loss dict
            utils.save_loss_dict(self.loss_dict, self.SAVINGS_PATH)

            # Save latest network components
            torch.save(self.generator.state_dict(), self.SAVINGS_PATH + 'generator_latest.pth')
            torch.save(self.ac_loss.ac_model.state_dict(), self.SAVINGS_PATH + 'ac_model_trained_latest.pth')
            torch.save(self.verification_loss.verification_model.state_dict(), self.SAVINGS_PATH + 'ver_model_trained_latest.pth')

            # Save flow field generator that produces the lowest ac_loss
            if val_losses[0] < self.lowest_ac_loss:
                self.lowest_ac_loss = val_losses[0]
                torch.save(self.generator.state_dict(), self.SAVINGS_PATH + 'generator_lowest_ac_loss.pth')
                print('Current generator with lowest ac_loss: epoch ' + str(epoch))

            # Save flow field generator that produces the lowest ver_loss
            if val_losses[1] < self.lowest_ver_loss:
                self.lowest_ver_loss = val_losses[1]
                torch.save(self.generator.state_dict(), self.SAVINGS_PATH + 'generator_lowest_ver_loss.pth')
                print('Current generator with lowest ver_loss: epoch ' + str(epoch))

            # Save flow field generator that produces the lowest total_loss
            if val_losses[3] < self.lowest_total_loss:
                self.lowest_total_loss = val_losses[3]
                torch.save(self.generator.state_dict(), self.SAVINGS_PATH + 'generator_lowest_total_loss.pth')
                print('Current generator with lowest total_loss: epoch ' + str(epoch))
                torch.save(self.ac_loss.ac_model.state_dict(), self.SAVINGS_PATH + 'ac_model_trained_lowest_total_loss.pth')
                torch.save(self.verification_loss.verification_model.state_dict(), self.SAVINGS_PATH + 'ver_model_trained_lowest_total_loss.pth')

        print('Finished Training!')

    def run(self):
        # Call training/validation loop
        self.training_validation()
