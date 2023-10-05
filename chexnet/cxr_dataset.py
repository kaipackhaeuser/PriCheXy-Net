import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class CXRDataset(Dataset):
    def __init__(self, path_to_images, fold, transform=None, sample=0, finding="any", perturbation_type='none'):
        '''Description.

        :param path_to_images: str
            The path to the folder where the images are stored.
        :param fold: str
            Specifies the subset. Options: 'train', 'val', 'test'
        :param transform: torchvision.transforms
            To preprocess raw images.
        :param sample: int
            Specifies the size of the dataset. If one wants to use a small data proportion only.
        :param finding: str
            Specifies the abnormality finding. If one wants to evaluate only for one specific class.
        :param perturbation_type: str
            Defines the type of perturbation. Options: 'flow_field', 'privacy_net', 'dp_pix', 'none'
        '''

        self.path_to_images = path_to_images
        self.df = pd.read_csv('./chexnet/nih_labels.csv')
        self.df = self.df[self.df['fold'] == fold]
        self.transform = transform
        self.perturbation_type = perturbation_type

        if self.perturbation_type in ['flow_field', 'privacy_net', 'dp_pix']:
            self.trans = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])

        if sample > 0 and sample < len(self.df):
            self.df = self.df.sample(sample)

        if not finding == "any":
            if finding in self.df.columns:
                if len(self.df[self.df[finding] == 1]) > 0:
                    self.df = self.df[self.df[finding] == 1]
                else:
                    print("No positive cases exist for " + finding + ", returning all unfiltered cases")
            else:
                print("cannot filter on finding " + finding + " as not in data - please check spelling")

        self.df = self.df.set_index("Image Index")

        self.PRED_LABEL = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 
                           'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 
                           'Hernia']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.path_to_images, self.df.index[idx]))
        image = image.convert('L')

        label = np.zeros(len(self.PRED_LABEL), dtype=int)

        for i in range(0, len(self.PRED_LABEL)):
            if self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0:
                label[i] = self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int')

        if self.perturbation_type in ['flow_field', 'privacy_net', 'dp_pix']:
            image = self.trans(image)
        else:
            image = image.convert('RGB')
            
            if self.transform:
                image = self.transform(image)

        return image, label, self.df.index[idx]
