import numpy as np
from PIL import Image
from utils import utils
from torch.utils import data
import torchvision.transforms as transforms


class SiameseDataset(data.Dataset):
    def __init__(self, phase='training', n_channels=3, image_size=256, b=None, m=None, eps=None, image_path='./'):
        """The dataset that is used for the patient verification experiments.

        :param phase: str
            Specifies the phase.
            Training: phase='training'
            Validation: phase='validation'
            Testing: phase='testing'
        :param n_channels: int
            Specifies the number of channels for the input images. Default: n_channels=3
        :param image_size: int
            The image size to be used. Default: image_size=256
        :param b: int
            Specifies the size of the grid cells for DP pixelization.
        :param m: int
            Specifies the m-neighborhood (DP sensitivity factor).
        :param eps: int
            Specifies the DP privacy budget (smaller values indicate greater privacy).
        :param image_path: str
            The path to the folder where the images are stored.
        """

        self.phase = phase
        self.n_channels = n_channels
        self.image_size = image_size
        self.PATH = image_path

        # Load the image pairs (filenames)
        if self.phase == 'training':
            self.image_pairs = np.loadtxt('./image_pairs/image_pairs_training_10000.txt', dtype=str)
        elif self.phase == 'validation':
            self.image_pairs = np.loadtxt('./image_pairs/image_pairs_validation_2000.txt', dtype=str)
        elif self.phase == 'testing':
            self.image_pairs = np.loadtxt('./image_pairs/image_pairs_testing_5000.txt', dtype=str)
        else:
            raise Exception('Invalid argument for parameter phase!')

        # Parameters that are used for the DP-Pix experiments (if self.use_dp_pix is True)
        self.use_dp_pix = b is not None and m is not None and eps is not None
        self.b = b
        self.m = m
        self.eps = eps

        # Define transformations
        if self.n_channels == 1:
            self.transform = transforms.ToTensor()
        elif self.n_channels == 3:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.resize = transforms.Resize((self.image_size, self.image_size))
        self.to_pil = transforms.ToPILImage()

        self.images_1 = []
        self.images_2 = []
        self.labels = []

        for i in range(len(self.image_pairs)):
            # Load images
            image1 = pil_loader(self.PATH + self.image_pairs[i][0], self.n_channels)
            image2 = pil_loader(self.PATH + self.image_pairs[i][1], self.n_channels)

            # Resize images
            image1 = self.resize(image1)
            image2 = self.resize(image2)

            # Apply DP-Pix if respective flag is activated
            if self.use_dp_pix:
                image1 = self.transform(image1).unsqueeze(0)
                image1 = utils.dp_pix(image_tensor=image1, b=self.b, m=self.m, eps=self.eps, plot=False).squeeze(0)
                image1 = self.to_pil(image1)

                if self.phase == 'training' or self.phase == 'validation':
                    image2 = self.transform(image2).unsqueeze(0)
                    image2 = utils.dp_pix(image_tensor=image2, b=self.b, m=self.m, eps=self.eps, plot=False).squeeze(0)
                    image2 = self.to_pil(image2)

            # Append images to respective lists
            self.images_1.append(image1)
            self.images_2.append(image2)
            self.labels.append(float(self.image_pairs[i][2]))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        # Get image1 and image2
        image1 = self.images_1[index]
        image2 = self.images_2[index]

        # Apply the transformation
        image1 = self.transform(image1)
        image2 = self.transform(image2)

        return image1, image2, self.labels[index]


def pil_loader(path, n_channels):
    with open(path, 'rb') as f:
        img = Image.open(f)
        if n_channels == 1:
            return img.convert('L')
        elif n_channels == 3:
            return img.convert('RGB')
        else:
            raise ValueError('Invalid value for parameter n_channels!')
