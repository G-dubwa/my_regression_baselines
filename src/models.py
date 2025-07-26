import torch 
import numpy as np 
from sklearn import metrics 
from torchvision.models import resnet18

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

class ResNet18Regression(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.base_model = resnet18()
        self.base_model.fc = torch.nn.Linear(512, 1)
    
    def forward(self, x):
        z = self.base_model(x).squeeze(-1)
        return z

    
 

    
