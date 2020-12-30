import numpy
from numpy.random import random, randint, choice
import torch.nn.functional as F

import torch
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


    "renvoie la meilleure action selon l'état actuel"
    def get_action(self, state, strategie,epsilon=0,tau=0):
        self.model_anticipation.eval()
        state = torch.FloatTensor(state).to(self.device)
        Qvaleur = self.model_anticipation(state)

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

        self.model_anticipation.train()
        return action


    "Apprentissage de l'agent"
    def learn(self, sizeBatch, gamma):
        self.compteur_apprentissage += 1

        #On commence l'apprentissage lorsqu'il y a assez de données en mémoire pour constituer un minibatch
        if self.buffer.index+self.buffer.index*self.buffer.init < sizeBatch:
            return

        #On récupère un minibatch
        states, actions, nextstates, rewards, done = self.buffer.get_minibatch(sizeBatch)

        # On transforme les array en tensor
        state_batch = torch.from_numpy(states)
        action_batch = torch.from_numpy(actions)
        state_1_batch = torch.from_numpy(nextstates)
        reward_batch = torch.from_numpy(rewards)
        done = torch.from_numpy(done)

        # get output for the next state
        output_1_batch = self.model_anticipation(state_1_batch.float())

        #print(reward_batch)
        #print(output_1_batch)
        # application de l'équation de Bellman : à r si l'épisode se termine, à r + gamma*max(Q) sinon
        y_batch = torch.stack(tuple(reward_batch[i] if i == sizeBatch-1
                                  else reward_batch[i] + gamma * torch.max(output_1_batch[i])
                                  for i in range(sizeBatch)))

        #On extrait la Qvaleur
        #q_value = torch.sum(self.model_anticipation(state_batch.float()) * action_batch, dim=1)
        #Q_anticipation = self.model_anticipation(states.float()).gather(1, actions.long().unsqueeze(1))

        q_value = torch.sum(self.model_anticipation(state_batch.float()) * action_batch.long().unsqueeze(1), dim =1)


        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        self.optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        perte = self.erreur(q_value.float(), y_batch.float())

        # do backward pass
        perte.backward()
        self.optimizer.step()




        """""

        #On récupère un minibatch
        states, actions, nextstates, rewards, done = self.buffer.get_minibatch(sizeBatch)

        #On transforme les array en tensor
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        nextstates = torch.from_numpy(nextstates)
        rewards = torch.from_numpy(rewards)
        done = torch.from_numpy(done)
        Q_anticipation = self.model_anticipation(states.float()).gather(1, actions.long().unsqueeze(1))

       


        Q_anticipation = Q_anticipation.reshape([sizeBatch])
        Q_next = self.model_anticipation(nextstates.float()).detach()
        Q_apprentissage = rewards + 0.9 * Q_next.max(1)[0].reshape([sizeBatch])

        #On optimise l'apprentissage selon l'erreur obtenue entre les deux modèles
        self.optimizer.zero_grad()
        perte = self.erreur(Q_anticipation.float(), Q_apprentissage.float())
        perte.backward()
        self.optimizer.step()
        """

        #Mise à jour des poids
        dict_anticipation = self.model_anticipation.state_dict()
        dict_apprentissate = self.model_apprentissage.state_dict()
        for weights in dict_apprentissate:
            dict_anticipation[weights] = (1 - 0.01) * dict_anticipation[weights] + 0.01 * dict_apprentissate[weights]
            self.model_anticipation.load_state_dict(dict_anticipation)


    "sauvegarde l'expérience en mémoire"
    def add_memoire(self, ob, action, ob_next, reward, done):
        self.buffer.add(ob, action, ob_next, reward, done)

