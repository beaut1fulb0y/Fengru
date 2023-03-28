import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18


class CustomResNet18(nn.Module):
    def __init__(self, num_classes, pretrain, dropout=0.1):
        super(CustomResNet18, self).__init__()

        # Load the pre-trained ResNet18 model
        if pretrain:
            self.resnet18 = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.resnet18 = resnet18()
        self.dropout = nn.Dropout(p=dropout)

        # Modify the last layer to match the number of classes in your dataset
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, num_classes)

        # You can add more custom layers or modify the existing layers here
        # Example: self.custom_layer = nn.Linear(128, 64)

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Apply dropout before the fully connected layer
        x = self.resnet18.fc(x)
        return x


if __name__ == "__main__":
    num_classes = 10  # Set this to the number of classes in your dataset
    dropout_p = 0.5  # Set the desired dropout probability
    model = CustomResNet18(num_classes, dropout_p)
    print(model)


