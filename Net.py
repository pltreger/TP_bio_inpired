import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, nb_actions):
        super(Net, self).__init__()
        # prend en entrée un état
        # sa taille de sortie est de même taille que le nombre d’actions possibles
        self.fc1 = nn.Linear(4, 50)
        self.fc2 = nn.Linear(50, nb_actions)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
