import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms


class DatasetPretrain(data.Dataset):
    def __init__(self, phase='training', image_size=256, n_channels=1, image_path='./'):
        """The dataset that is used for pre-training the flow field generator of PriCheXy-Net.

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

        # Load the image filenames
        if self.phase == 'training':
            self.filenames = np.loadtxt('./image_pairs/train_val_list.txt', dtype=str)[:75708]
        elif self.phase == 'validation':
            self.filenames = np.loadtxt('./image_pairs/train_val_list.txt', dtype=str)[75708:]
        elif self.phase == 'testing':
            self.filenames = np.loadtxt('./image_pairs/test_list.txt', dtype=str)
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

        self.images = []

        for file in self.filenames:
            # Load image
            image = pil_loader(self.image_path + file, self.n_channels)
            # Resize image
            image = self.resize(image)
            # Append image to list
            self.images.append(image)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        # Get image
        image = self.images[index]

        # Apply the transformation
        image = self.transform(image)

        return image


def pil_loader(path, n_channels):
    with open(path, 'rb') as f:
        img = Image.open(f)
        if n_channels == 1:
            return img.convert('L')
        elif n_channels == 3:
            return img.convert('RGB')
        else:
            raise ValueError('Invalid value for parameter n_channels!')
