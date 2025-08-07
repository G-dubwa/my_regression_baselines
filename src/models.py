import torch 
import numpy as np 
from sklearn import metrics 
from torchvision.models import resnet18, efficientnet_b0, mobilenet_v2, squeezenet1_0

class LinearRegression(torch.nn.Module):
    """

    Simple linear regression
    
    P: input dimension

    """
    def __init__(self,P=128):
        super().__init__()
        self.linear = torch.nn.Linear(P,1)
    
    def forward(self, x):
        """
        Returns continuous value (batch_size,)
        """
        z = self.linear(x).squeeze(-1)
        return z

class TinyCNNRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # (3,224,224) → (16,224,224)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # → (16,112,112)

            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # → (32,56,56)

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1,1))  # → (64,1,1)
        )
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x).squeeze(-1)

class SqueezeNetRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = squeezenet1_0()
        self.base_model.classifier[1] = torch.nn.Conv2d(512, 1, kernel_size=1)
    
    def forward(self, x):
        x = self.base_model(x)
        return x.squeeze(-1).squeeze(-1)

class MobileNetRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = mobilenet_v2()
        self.base_model.classifier[1] = torch.nn.Linear(1280, 1)
    
    def forward(self, x):
        return self.base_model(x).squeeze(-1)

class ResNet18Regression(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.base_model = resnet18()
        self.base_model.fc = torch.nn.Linear(512, 1)
    
    def forward(self, x):
        z = self.base_model(x).squeeze(-1)
        return z


class EfficientNetRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = efficientnet_b0()
        self.base_model.classifier[1] = torch.nn.Linear(1280, 1)

    def forward(self, x):
        return self.base_model(x).squeeze(-1)


