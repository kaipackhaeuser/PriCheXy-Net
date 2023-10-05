import time
import copy
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim

from utils import utils
from utils.EarlyStopping import EarlyStopping
from utils.GaussianSmoothing import GaussianSmoothing

from networks.UNet_PriCheXyNet import UNet
from networks.UNet_PrivacyNet import Unet2D_encoder
from networks.SiameseNetwork import SiameseNetwork


class AgentSiameseNetwork:
    def __init__(self, config):
        """This is the agent that provides code for re-training, validating and testing the patient verification model.

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
        self.perturbation_type = self.config['perturbation_type']
        self.perturbation_model_file = self.config['perturbation_model_file']
        self.mu = self.config['mu']

        self.b = self.config['b']
        self.m = self.config['m']
        self.eps = self.config['eps']

        self.image_size = self.config['image_size']
        self.batch_size = self.config['batch_size']
        self.learning_rate = self.config['learning_rate']
        self.max_epochs = self.config['max_epochs']
        self.early_stopping = self.config['early_stopping']

        self.num_workers = 16
        self.pin_memory = True

        # Define the identity grid and the 
        if self.perturbation_type == 'flow_field':
            d = torch.linspace(-1, 1, 256)
            mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
            grid_identity = torch.stack((mesh_y, mesh_x), 2)
            self.grid_identity = grid_identity.unsqueeze(0).permute(0, 3, 1, 2).cuda()
            self.gauss_filter = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).cuda()
        elif self.perturbation_type in ['privacy_net', 'dp_pix', 'none']:
            self.grid_identity = None
            self.gauss_filter = None
        else:
            raise Exception('Invalid argument: ' + self.perturbation_type)

        self.start_epoch = 0
        self.es = EarlyStopping(patience=self.early_stopping)
        self.best_loss = 100000
        self.loss_dict = {'training': [],
                          'validation': []}

        # Initialize the perturbation network
        if self.perturbation_type == 'none':
            self.n_channels = 3
            self.perturbation_net = None
        else:
            self.n_channels = 1
            if self.perturbation_type == 'flow_field':
                self.perturbation_net = UNet(1, 2, 32).cuda()
                self.perturbation_net.load_state_dict(torch.load(self.perturbation_model_file))
            elif self.perturbation_type == 'privacy_net':
                self.perturbation_net = Unet2D_encoder(self.n_channels, self.n_channels, 16).cuda()
                self.perturbation_net.load_state_dict(torch.load(self.perturbation_model_file))
            elif self.perturbation_type == 'dp_pix':
                self.perturbation_net = None
            else:
                raise Exception('Invalid argument: ' + self.perturbation_type)
            

        # Define the siamese neural network architecture
        self.net = SiameseNetwork().cuda()
        self.best_net = copy.deepcopy(self.net)

        # Choose loss function
        self.loss = nn.BCEWithLogitsLoss().cuda()

        # Set the optimizer function
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # Initialize data loaders
        self.training_loader = utils.get_data_loader(phase='training', experimental_step='retrainSNN', 
                                                     image_size=self.image_size, n_channels=self.n_channels, 
                                                     batch_size=self.batch_size, shuffle=True, 
                                                     num_workers=self.num_workers, pin_memory=self.pin_memory, 
                                                     b=self.b, m=self.m, eps=self.eps, image_path=self.IMAGE_PATH)
        self.validation_loader = utils.get_data_loader(phase='validation', experimental_step='retrainSNN',
                                                       image_size=self.image_size, n_channels=self.n_channels, 
                                                       batch_size=self.batch_size, shuffle=False, 
                                                       num_workers=self.num_workers, pin_memory=self.pin_memory, 
                                                       b=self.b, m=self.m, eps=self.eps, image_path=self.IMAGE_PATH)
        self.test_loader = utils.get_data_loader(phase='testing', experimental_step='retrainSNN', 
                                                 image_size=self.image_size, n_channels=self.n_channels, 
                                                 batch_size=self.batch_size, shuffle=False, 
                                                 num_workers=self.num_workers, pin_memory=self.pin_memory,
                                                 b=self.b, m=self.m, eps=self.eps, image_path=self.IMAGE_PATH)

    def training_validation(self):
        # Training and validation loop
        for epoch in range(self.start_epoch, self.max_epochs):
            start_time = time.time()

            training_loss = utils.train_snn(self.perturbation_type, self.net, self.perturbation_net, self.grid_identity, 
                                            self.gauss_filter, self.mu, self.training_loader, self.loss, self.optimizer, 
                                            epoch, self.max_epochs)
            validation_loss = utils.validate_snn(self.perturbation_type, self.net, self.perturbation_net, 
                                                 self.grid_identity, self.gauss_filter, self.mu, self.validation_loader, 
                                                 self.loss, epoch, self.max_epochs)

            self.loss_dict['training'].append(training_loss)
            self.loss_dict['validation'].append(validation_loss)

            end_time = time.time()
            print('Time elapsed for epoch ' + str(epoch + 1) + ': ' + str(
                round((end_time - start_time) / 60, 2)) + ' minutes')

            if validation_loss < self.best_loss:
                self.best_loss = validation_loss
                self.best_net = copy.deepcopy(self.net)

            torch.save(self.best_net.state_dict(), self.SAVINGS_PATH + self.config[
                'experiment_description'] + '_best_network.pth')

            utils.save_loss_curves_snn(self.loss_dict, self.SAVINGS_PATH, self.config['experiment_description'])
            utils.plot_loss_curves_snn(self.loss_dict, self.SAVINGS_PATH, self.config['experiment_description'])

            if self.es.step(validation_loss):
                break

        print('Finished Training!')
    
    def testing_evaluation(self):
        # Testing phase
        y_true, y_pred = utils.test_snn(self.perturbation_type, self.best_net, self.perturbation_net, 
                                        self.grid_identity, self.gauss_filter, self.mu, self.test_loader)

        y_true, y_pred = [y_true.numpy(), y_pred.numpy()]

        # Compute the evaluation metrics
        fp_rates, tp_rates, thresholds = metrics.roc_curve(y_true, y_pred)
        auc = metrics.roc_auc_score(y_true, y_pred)
        y_pred_thresh = utils.apply_threshold(y_pred, 0.5)
        accuracy, f1_score, precision, recall, report, confusion_matrix = utils.get_evaluation_metrics(y_true,
                                                                                                       y_pred_thresh)
        auc_mean, confidence_lower, confidence_upper = utils.bootstrap(1000,
                                                                       y_true,
                                                                       y_pred,
                                                                       self.SAVINGS_PATH,
                                                                       self.config['experiment_description'])

        # Plot ROC curve
        utils.plot_roc_curve(fp_rates, tp_rates, self.SAVINGS_PATH, self.config['experiment_description'])

        # Save all the results to files
        utils.save_labels_predictions(y_true, y_pred, y_pred_thresh, self.SAVINGS_PATH,
                                      self.config['experiment_description'])

        utils.save_results_to_file(auc, accuracy, f1_score, precision, recall, report, confusion_matrix,
                                   self.SAVINGS_PATH, self.config['experiment_description'])

        utils.save_roc_metrics_to_file(fp_rates, tp_rates, thresholds, self.SAVINGS_PATH,
                                       self.config['experiment_description'])

        # Print the evaluation metrics
        print('EVALUATION METRICS:')
        print('AUC: ' + str(auc))
        print('Accuracy: ' + str(accuracy))
        print('F1-Score: ' + str(f1_score))
        print('Precision: ' + str(precision))
        print('Recall: ' + str(recall))
        print('Report: ' + str(report))
        print('Confusion matrix: ' + str(confusion_matrix))

        print('BOOTSTRAPPING: ')
        print('AUC Mean: ' + str(auc_mean))
        print('Confidence interval for the AUC score: ' + str(confidence_lower) + ' - ' + str(confidence_upper))
    
    def run(self):
        # Call training/validation and testing loop successively
        self.training_validation()
        self.testing_evaluation()
