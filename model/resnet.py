import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, resnet101


class CustomResNet18(nn.Module):
    def __init__(self, num_classes, pretrain, dropout=0.1):
        super(CustomResNet18, self).__init__()

        # Load the pre-trained ResNet18 model
        if pretrain:
            if torch.__version__ == "2.0.0":
                self.resnet = resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                self.resnet = resnet18(pretrained=True)
        else:
            self.resnet = resnet18()
        self.dropout = nn.Dropout(p=dropout)

        # Modify the last layer to match the number of classes in your dataset
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            self.dropout,
            nn.Linear(num_features, num_classes),
        )

        # You can add more custom layers or modify the existing layers here
        # Example: self.custom_layer = nn.Linear(128, 64)

    def forward(self, x):
        x = self.resnet(x)
        return x


class CustomResNet101(nn.Module):
    def __init__(self, num_classes, pretrain, dropout=0.1):
        super(CustomResNet101, self).__init__()

        # Load the pre-trained ResNet18 model
        if pretrain:
            if torch.__version__ == "2.0.0":
                self.resnet = resnet101(weights=models.ResNet101_Weights.DEFAULT)
            else:
                self.resnet = resnet101(pretrained=True)
        else:
            self.resnet = resnet101()
        self.dropout = nn.Dropout(p=dropout)

        # Modify the last layer to match the number of classes in your dataset
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            self.dropout,
            nn.Linear(num_features, num_classes),
        )

        # You can add more custom layers or modify the existing layers here
        # Example: self.custom_layer = nn.Linear(128, 64)

    def forward(self, x):
        x = self.resnet(x)
        return x


if __name__ == "__main__":
    num_classes = 10  # Set this to the number of classes in your dataset
    dropout_p = 0.5  # Set the desired dropout probability
    model = CustomResNet18(num_classes, dropout_p)
    print(model)
