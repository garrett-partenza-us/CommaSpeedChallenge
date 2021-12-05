import torch
import torch.nn.functional as F

class SFF(torch.nn.Module):
    def __init__(self, n_feature):
        super(SFF, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_feature//4**1)
        self.hidden2 = torch.nn.Linear(n_feature//4**1, n_feature//4**2)
        self.hidden3 = torch.nn.Linear(n_feature//4**2, n_feature//4**3)
        self.predict = torch.nn.Linear(n_feature//4**3, 1)
                                     
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.predict(x)             
        return x
