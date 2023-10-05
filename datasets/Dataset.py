import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision import transforms


class Dataset(data.Dataset):
    def __init__(self, phase='training', image_size=256, n_channels=1, image_path='./'):
        """The dataset that is used for the chest X-ray anonymization experiments.

        :param phase: str
            Specifies the phase.
            Training: phase='training'
            Validation: phase='validation'
            Testing: phase='testing'
        :param image_size: int
            The image size to be used. Default: image_size=256
        :param n_channels: int
            Specifies the number of channels for the input images. Default: n_channels=1
        :param image_path: str
            The path to the folder where the images are stored.
        """

        self.phase = phase
        self.image_size = image_size
        self.n_channels = n_channels
        self.image_path = image_path

        # Load the image pairs (filenames)
        if self.phase == 'training':
            self.image_pairs = np.loadtxt('./image_pairs/image_pairs_training_10000.txt', dtype=str)
        elif self.phase == 'validation':
            self.image_pairs = np.loadtxt('./image_pairs/image_pairs_validation_2000.txt', dtype=str)
        elif self.phase == 'testing':
            self.image_pairs = np.loadtxt('./image_pairs/image_pairs_testing_5000.txt', dtype=str)
        else:
            raise Exception('Invalid argument for parameter phase!')
        
        # Define transformations
        if self.n_channels == 1:
            self.transform = transforms.ToTensor()
        elif self.n_channels == 3:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        self.resize = transforms.Resize((self.image_size, self.image_size))

        self.images_1 = []
        self.images_2 = []
        self.ac_labels_1 = []
        self.labels_id = []

        self.PRED_LABEL = [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia']

        # Load meta-information
        meta_data = pd.read_csv("Data_Entry_2017_v2020.csv").values
        filenames = meta_data[:, 0]
        findings = meta_data[:, 1]

        for i in range(len(self.image_pairs)):
            # Get abnormality labels of image1 and provide them as one-hot encoded binary vectors
            idx = np.where(filenames == self.image_pairs[i, 0])
            finding = findings[idx]
            label = np.zeros(14)

            if finding == 'No Finding':
                self.ac_labels_1.append(label)
            else:
                finding = finding[0].split('|')
                for d in finding:
                    idx_d = self.PRED_LABEL.index(d)
                    label[idx_d] = 1
                self.ac_labels_1.append(label)

            # Load images
            image1 = pil_loader(self.image_path + self.image_pairs[i][0], self.n_channels)
            image2 = pil_loader(self.image_path + self.image_pairs[i][1], self.n_channels)

            # Resize images
            image1 = self.resize(image1)
            image2 = self.resize(image2)

            # Append images to respective lists
            self.images_1.append(image1)
            self.images_2.append(image2)
            self.labels_id.append(float(self.image_pairs[i][2]))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        # Get image1 and image2
        image1 = self.images_1[index]
        image2 = self.images_2[index]

        # Apply the transformation
        image1 = self.transform(image1)
        image2 = self.transform(image2)

        return image1, image2, self.ac_labels_1[index].astype(np.float32), self.labels_id[index]


def pil_loader(path, n_channels):
    with open(path, 'rb') as f:
        img = Image.open(f)
        if n_channels == 1:
            return img.convert('L')
        elif n_channels == 3:
            return img.convert('RGB')
        else:
            raise ValueError('Invalid value for parameter n_channels!')
