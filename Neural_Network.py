from random import randint
import random
import numpy

import torch.nn.functional as F

import torch
from numpy.random.mtrand import choice

from Net import Net
from Buffer import Buffer


class Neural_Network:
    def __init__(self, nb_actions, sizeBuffer):
        #initialisation du model_anticipation
        self.device = torch.device("cpu")
        self.model_anticipation = Net(nb_actions).to(self.device)
        self.model_apprentissage = Net(nb_actions).to(self.device)

        self.nb_actions = nb_actions

        self.buffer = Buffer(sizeBuffer)

        self.optimizer = torch.optim.Adam(self.model_anticipation.parameters())
        self.erreur = torch.nn.MSELoss().to(self.device)

        self.compteur_apprentissage = 0

        self.memory = []


    "renvoie la meilleure action selon l'état actuel"
    def get_action(self, state, strategie,epsilon=0,tau=0):
        #self.model_anticipation.eval()
        state = torch.FloatTensor(state).to(self.device)
        Qvaleur = self.model_anticipation(state)

        if strategie == "aleatoire":
            action = randint(0, self.nb_actions)

        elif strategie == "e-greedy":
            #probabilité epsilon : choix aléatoire de l'action
            if random.random() < epsilon:
                action = randint(0, self.nb_actions-1)
            #probabilité e-1 : renvoie la meilleure action
            else:
                action = torch.argmax(Qvaleur).item()

        elif strategie == "boltzmann":
            proba = F.softmax(Qvaleur / tau, dim=0).detach()
            proba = proba.cpu().detach().numpy()
            actions_possibles = numpy.arange(2)
            action = choice(actions_possibles, p=proba)

        #self.model_anticipation.train()
        return action


    "Apprentissage de l'agent"
    def learn(self, sizeBatch, nb_actions, gamma):
        self.compteur_apprentissage += 1

        #On commence l'apprentissage lorsqu'il y a assez de données en mémoire pour constituer un minibatch
        if self.buffer.index+self.buffer.index*self.buffer.init < sizeBatch:
            return

        # On récupère un minibatch
        states, actions, nextstates, rewards, done = self.buffer.get_minibatch(sizeBatch)

        # On transforme les array en tensor
        state_batch = torch.from_numpy(states)
        action_batch = torch.from_numpy(actions)
        nextstate_batch = torch.from_numpy(nextstates)
        reward_batch = torch.from_numpy(rewards)
        done_batch = torch.from_numpy(done)

        Q_next = self.model_anticipation(nextstate_batch.float()).detach()

        # application de l'équation de Bellman : à r si l'épisode se termine, à r + gamma*max(Q) sinon
        Q_apprentissage = torch.stack(tuple(reward_batch[i] if i == sizeBatch - 1
                                            else reward_batch[i] + gamma * torch.max(Q_next[i])
                                            for i in range(sizeBatch)))

        Q_anticipation = self.model_anticipation(state_batch.float()).gather(1, action_batch.long().unsqueeze(1))
        Q_anticipation = Q_anticipation.reshape([sizeBatch])

        # On optimise l'apprentissage selon l'erreur obtenue entre les deux modèles
        self.optimizer.zero_grad()
        perte = self.erreur(Q_anticipation.float(), Q_apprentissage.float())
        perte.backward()
        self.optimizer.step()


        #Mise à jour des poids
        dict_anticipation = self.model_anticipation.state_dict()
        dict_apprentissate = self.model_apprentissage.state_dict()
        for weights in dict_apprentissate:
            dict_anticipation[weights] = (1 - 0.01) * dict_anticipation[weights] + 0.01 * dict_apprentissate[weights]
            self.model_anticipation.load_state_dict(dict_anticipation)

    "sauvegarde l'expérience en mémoire"
    def add_memoire(self, ob, action, ob_next, reward, done):
        self.buffer.add(ob, action, ob_next, reward, done)

