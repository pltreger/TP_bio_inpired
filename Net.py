import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, nb_actions, ENVIRONNEMENT, sizeBuffer):
        super(Net, self).__init__()
        self.ENVIRONNEMENT = ENVIRONNEMENT
        self.sizeBuffer = sizeBuffer
        # prend en entrée un état
        # sa taille de sortie est de même taille que le nombre d’actions possibles
        if ENVIRONNEMENT == "CartPole":
            self.fc1 = nn.Linear(4, 100)
            self.fc2 = nn.Linear(100, nb_actions)

        if ENVIRONNEMENT == "Vizdoom":
            ### DEBUT TESTS ###
            '''
            self.features = nn.Sequential(
                nn.Conv2d(4, 32, 8, 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear(512, 3)
            )
            '''
            '''
            self.conv1 = nn.Conv2d(100, 100, 100, 100)
            self.conv2 = nn.Conv2d(6, 16, 3)
            self.fc1 = nn.Linear(5824, 200)
            self.fc2 = nn.Linear(200, 100)
            self.fc3 = nn.Linear(100, 3)
            '''
            '''
            self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
            self.bn3 = nn.BatchNorm2d(32)
            '''
            ### FIN TESTS ###


    def forward(self, x):
        if self.ENVIRONNEMENT == "CartPole":
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

        if self.ENVIRONNEMENT == "Vizdoom":
            ### DEBUT TESTS ###
            '''
            x = F.relu(self.bn1(self.conv1(x.reshape([1, 1, 112, 64]))))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            return self.head(x.view(x.size(0), -1))
            '''
            '''
            x = F.max_pool2d(F.relu(self.conv1(x)), (1,1,112,64))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
            '''
            '''
            x = self.features(x)
            x = self.classifier(x.view(x.size(0), -1))
            return x
            '''
            ### FIN TESTS ###

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


