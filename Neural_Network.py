import numpy
from numpy.random import random, randint, choice
import torch.nn.functional as F

import torch
from Net import Net
from Buffer import Buffer


class Neural_Network:
    def __init__(self, nb_actions, sizeBuffer):
        #initialisation du model
        self.device = torch.device("cpu")
        print(nb_actions)
        self.model = Net(nb_actions).to(self.device)

        #On essaie de charger les models si ils existent déjà
        try:
            self.model = torch.load("Save/models.data")
            print("Models loaded from Memory ! ")
        except:
            pass

        self.nb_actions = nb_actions

        self.buffer = Buffer(sizeBuffer)



    def get_action(self, state, strategie,epsilon=0,tau=0):
        #renvoie la meilleure action selon l'état actuel
        self.model.eval()
        state = torch.FloatTensor(state).to(self.device)
        Qvaleur = self.model(state)

        if strategie == "aleatoire":
            action = randint(0, self.nb_actions)

        elif strategie == "e-greedy":
            #probabilité epsilon : choix aléatoire de l'action
            if random() < epsilon:
                action = randint(0, self.nb_actions)
            #probabilité e-1 : renvoie la meilleure action
            else:
                action = torch.argmax(Qvaleur).item()

        elif strategie == "boltzmann":
            proba = F.softmax(Qvaleur / tau, dim=0).detach()
            proba = proba.cpu().detach().numpy()
            actions_possibles = numpy.arange(2)
            action = choice(actions_possibles, p=proba)

        self.model.train()
        return action

    def add_memoire(self, ob, action, ob_next, reward, done):
        self.buffer.add(ob, action, ob_next, reward, done)