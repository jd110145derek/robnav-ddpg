import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO(Lab-02): Complete the network model.
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.linear1 = torch.nn.Linear(23,512)
        self.linear2 = torch.nn.Linear(512,512)
        self.linear3 = torch.nn.Linear(512,512)
        self.linear4 = torch.nn.Linear(512,2)
    def forward(self, s):
        output = self.linear1(s)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.relu(output)
        output = self.linear3(output)
        output = self.relu(output)
        output = self.linear4(output)
        a = self.tanh(output)
        return a
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.linear1 = torch.nn.Linear(23,512)
        self.linear2 = torch.nn.Linear(514,512)
        self.linear3 = torch.nn.Linear(512,512)
        self.linear4 = torch.nn.Linear(512,1)
    def forward(self, s, a):
        output = self.linear1(s)
        output = self.relu(output)
        output = torch.cat((output,a),1)
        output = self.linear2(output)
        output = self.relu(output)
        output = self.linear3(output)
        output = self.relu(output)
        output = self.linear4(output)
        return output
