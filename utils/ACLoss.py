import copy
import torch.nn as nn
import torchvision.transforms as transforms


class ACLoss(nn.Module):
    def __init__(self, ac_model, reduction: str = 'mean'):
        """The auxiliary classifier loss which is intended to be used to ensure that underlying abnormality patterns are
        preserved during the anonymization process.

        :param ac_model: nn.Module
            A pre-trained abnormality classifier (DenseNet-121).
        :param reduction: str
            Loss reduction method: default value is 'mean'; other options are 'sum' or 'none'.
        """

        super().__init__()
        self.ac_model = ac_model

        # Set model to evaluation mode
        self.ac_model.eval()

        # Turn on gradient computation
        for param in self.ac_model.parameters():
            param.requires_grad = True

        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=self.reduction).cuda()

    def forward(self, deformed_image, target_labels):
        # The abnormality classification model was trained with 3-channel inputs
        # --> expand tensors to have 3 identical channels
        deformed_image = deformed_image.expand(-1, 3, -1, -1)

        # Apply the ImageNet transform (since the classifier was trained with the ImageNet transform as well)
        resize = transforms.Resize(224)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        deformed_image = normalize(resize(deformed_image))

        # Cut the last layer for the actual loss model
        loss_model = copy.deepcopy(self.ac_model)
        loss_model.classifier = nn.Sequential(*list(loss_model.classifier.children())[:-1])

        # Compute the classification output
        ac_predictions = loss_model(deformed_image)
        loss = self.bce_loss(ac_predictions, target_labels)

        return loss
