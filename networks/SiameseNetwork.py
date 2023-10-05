import torch
import torch.nn as nn
import torchvision.models as models


class SiameseNetwork(nn.Module):
    def __init__(self):
        """
        This class represents the Siamese Neural Network (patient verification model) incorporated into PriCheXy-Net. 
        It takes two chest X-ray images and yields a similarity score, indicating the probability of whether or not the 
        two scans belong to the same patient.
        """

        super(SiameseNetwork, self).__init__()

        # Model: Use ResNet-50 architecture in both network branches
        self.model = models.resnet50(pretrained=True)

        # Adjust the ResNet classification layer to produce 128-dimensional feature vectors
        self.model.fc = nn.Linear(in_features=2048, out_features=128, bias=True)

        # Final FC layer to produce a single output score
        self.fc_end = nn.Linear(128, 1)

    def forward_once(self, x):
        # Forward function for a single branch to get the feature vector before merging
        output = self.model(x)
        output = torch.sigmoid(output)
        return output

    def forward(self, input1, input2):
        # Forward
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # Compute the absolute difference between the feature vectors and pass it to the last FC-Layer
        difference = torch.abs(output1 - output2)
        output = self.fc_end(difference)

        return output
