import torch
import torch.nn as nn
import torchvision.transforms as transforms


class VerificationLoss(nn.Module):
    def __init__(self, verification_model, reduction: str = 'mean'):
        """The verification loss is intended to be used for learning a flow field that targetedly deforms a chest
        radiograph and thereby obfuscates the underlying biometric information.

        :param verification_model: nn.Module
            A pre-trained verification model (Siamese Neural Network).
        :param reduction: str
            Loss reduction method: default value is 'mean'; other options are 'sum' or 'none'.
        """

        super().__init__()
        self.verification_model = verification_model

        # Set model to evaluation mode
        self.verification_model.eval()

        # Turn on gradient computation
        for param in self.verification_model.parameters():
            param.requires_grad = True

        self.reduction = reduction

    def forward(self, output1, output2):
        # The verification model was trained with 3-channel inputs --> expand tensors to have 3 identical channels
        output1 = output1.expand(-1, 3, -1, -1)
        output2 = output2.expand(-1, 3, -1, -1)

        # Apply the ImageNet transform (since the verification model was trained with the ImageNet transform as well)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        output1 = normalize(output1)
        output2 = normalize(output2)

        # Compute SNN output followed by a sigmoid activation function
        verification_loss = torch.sigmoid(self.verification_model(output1, output2).to(dtype=torch.float64))

        return self._reduce(verification_loss)

    def _reduce(self, x):
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x
