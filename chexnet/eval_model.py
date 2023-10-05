import os
import numpy as np
import pandas as pd
import sklearn.metrics as sklm

import torch
from torch.autograd import Variable
from torchvision import transforms, utils

from networks.UNet_PriCheXyNet import UNet
from networks.UNet_PrivacyNet import Unet2D_encoder

from utils import utils
from utils.GaussianSmoothing import GaussianSmoothing

import chexnet.cxr_dataset as CXR


def make_pred_multilabel(data_transforms, model, image_path, save_path, perturbation_type='none', 
                         perturbation_checkpoint=None, mu=None, b=None, m=None, eps=None):
    """Gives predictions for test fold and calculates AUCs using pre-trained classifier model.

    :param data_transforms: torchvision.transforms
        To preprocess raw images; same as validation transformation.
    :param model: torch.nn.Module
        Pre-trained Densenet-121.
    :param image_path: str
        The path to the folder where the images are stored.
    :param save_path: str
        Path to the folder where the results are stored.
    :param perturbation_type: str
        Defines the type of perturbation. Options: 'flow_field', 'privacy_net', 'dp_pix', 'none'
    :param perturbation_checkpoint: torch.nn.Module
        The flow field generator that is used to targetedly deform the chest radiographs.
    :param mu: float
        This factor controls the degree of deformation. Larger values allow for more deformation.
        For mu=0, the images will not be deformed and the operation would result in the original images.
    :param b: int
        Specifies the size of the grid cells for DP pixelization.
    :param m: int
        Specifies the m-neighborhood (DP sensitivity factor).
    :param eps: int
        Specifies the DP privacy budget (smaller values indicate greater privacy).
    :return pred_df: pandas.DataFrame 
        Contains individual predictions and ground truth for each test image.
    :return auc_df: pandas.DataFrame
        Contains computed AUCs values.
    """

    try:
        os.makedirs(save_path)
    except BaseException:
        pass

    BATCH_SIZE = 16

    # Set model to eval mode; required for proper predictions given use of batchnorm
    model.train(False)

    # Create dataloader
    dataset = CXR.CXRDataset(
        path_to_images=image_path,
        fold="test",
        transform=data_transforms['val'],
        perturbation_type=perturbation_type)
    dataloader = torch.utils.data.DataLoader(
        dataset, BATCH_SIZE, shuffle=False, num_workers=8)

    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])

    if perturbation_type == 'flow_field':
        perturbation_model = UNet(1, 2, 32).cuda()
        perturbation_model.load_state_dict(perturbation_checkpoint)
        perturbation_model.eval()

        d = torch.linspace(-1, 1, 256)
        mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
        grid_identity = torch.stack((mesh_y, mesh_x), 2)
        grid_identity = grid_identity.unsqueeze(0).permute(0, 3, 1, 2).cuda()
        gauss_filter = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).cuda()

    if perturbation_type == 'privacy_net':
        perturbation_model = Unet2D_encoder(1, 1, 16).cuda()
        perturbation_model.load_state_dict(perturbation_checkpoint)
        perturbation_model.eval()

    if perturbation_type in ['flow_field', 'privacy_net', 'dp_pix']:
        trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Iterate over dataloader
    for i, data in enumerate(dataloader):

        inputs, labels, _ = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        if perturbation_type == 'flow_field':
            grid = perturbation_model(inputs)
            grid = grid_identity - mu * grid
            grid = gauss_filter(grid)
            grid = grid.permute(0, 2, 3, 1)
            inputs = torch.nn.functional.grid_sample(inputs, grid, padding_mode='border', align_corners=True)

        if perturbation_type == 'privacy_net':
            inputs = perturbation_model(inputs)

        if perturbation_type == 'dp_pix':
            inputs = utils.dp_pix(inputs, b, m, eps, plot=False)

        if perturbation_type in ['flow_field', 'privacy_net', 'dp_pix']:
            inputs = inputs.expand(-1, 3, -1, -1)
            inputs = trans(inputs)

        true_labels = labels.cpu().data.numpy()
        batch_size = true_labels.shape

        outputs = model(inputs)
        probs = outputs.cpu().data.numpy()

        # Get predictions and true values for each item in batch
        for j in range(0, batch_size[0]):
            thisrow = {}
            truerow = {}
            thisrow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]
            truerow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]

            # Iterate over each entry in prediction vector; each corresponds to individual label
            for k in range(len(dataset.PRED_LABEL)):
                thisrow["prob_" + dataset.PRED_LABEL[k]] = probs[j, k]
                truerow[dataset.PRED_LABEL[k]] = true_labels[j, k]

            pred_df = pd.concat([pred_df, pd.DataFrame(thisrow, index=[0])], ignore_index=True)
            true_df = pd.concat([true_df, pd.DataFrame(truerow, index=[0])], ignore_index=True)

        if(i % 10 == 0):
            print(str(i * BATCH_SIZE))

    auc_df = pd.DataFrame(columns=["label", "auc"])

    column_list = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 
                   'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    # Calculate AUCs
    for column in true_df:
        if column not in column_list:
            continue
        actual = true_df[column]
        pred = pred_df["prob_" + column]
        thisrow = {}
        thisrow['label'] = column
        thisrow['auc'] = np.nan
        try:
            thisrow['auc'] = sklm.roc_auc_score(actual.values.astype(int), pred.values)
        except BaseException:
            print("can't calculate auc for " + str(column))
        auc_df = pd.concat([auc_df, pd.DataFrame(thisrow, index=[0])], ignore_index=True)

    pred_df.to_csv(save_path + "preds.csv", index=False)
    auc_df.to_csv(save_path + "aucs.csv", index=False)
    return pred_df, auc_df
