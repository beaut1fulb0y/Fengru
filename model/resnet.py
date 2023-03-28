import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18


class CustomResNet18(nn.Module):
    def __init__(self, num_classes, pretrain):
        super(CustomResNet18, self).__init__()

        # Load the pre-trained ResNet18 model
        if pretrain:
            self.resnet18 = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.resnet18 = resnet18()

        # Modify the last layer to match the number of classes in your dataset
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, num_classes)

        # You can add more custom layers or modify the existing layers here
        # Example: self.custom_layer = nn.Linear(128, 64)

    def forward(self, x):
        x = self.resnet18(x)

        return x


if __name__ == "__main__":
    model = CustomResNet18()
