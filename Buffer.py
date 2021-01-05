from random import *
import numpy

class Buffer:
    def __init__(self, sizeBuffer, ENVIRONNEMENT):
        #On initialise la mémoire
        if ENVIRONNEMENT == "CartPole":
            self.states = numpy.zeros((sizeBuffer,4))
            self.nextstates = numpy.zeros((sizeBuffer,4))
        if ENVIRONNEMENT == "Vizdoom":
            self.states = numpy.zeros((sizeBuffer, 1, 1, 112, 64))
            self.nextstates = numpy.zeros((sizeBuffer, 1, 1, 112, 64))


        self.actions = numpy.zeros(sizeBuffer)
        self.rewards = numpy.zeros(sizeBuffer)
        self.done = numpy.zeros(sizeBuffer)

        self.sizeBuffer = sizeBuffer

        #Index correspondra à l'endroit où on se trouve dans la mémoire pour la remplir
        self.index = 0

    def add(self, state, action, next_state, reward, done):
        #On ajoute les nouvelles données dans la mémoire
        self.states[self.index % self.sizeBuffer] = state
        self.actions[self.index % self.sizeBuffer] = action
        self.nextstates[self.index % self.sizeBuffer] = next_state
        self.rewards[self.index % self.sizeBuffer] = reward
        self.done[self.index % self.sizeBuffer] = done

        #On modifie l'index
        self.index = self.index + 1

    def get_minibatch(self, sizeBatch, ENVIRONNEMENT):

        if ENVIRONNEMENT == "CartPole":
            states = numpy.zeros((sizeBatch,4))
            nextstates = numpy.zeros((sizeBatch,4))
        if ENVIRONNEMENT == "Vizdoom":
            states = numpy.zeros((sizeBatch, 1, 1, 112, 64))
            nextstates = numpy.zeros((sizeBatch, 1, 1, 112, 64))
        actions = numpy.zeros(sizeBatch)
        rewards = numpy.zeros(sizeBatch)
        done = numpy.zeros(sizeBatch)

        listIndex = []

        if self.index < self.sizeBuffer:
            init = self.index
        else:
            init = self.sizeBuffer

        for k in range(sizeBatch):
            index = randint(0,init-1)
            #On fait en sorte qu'il n'y ait pas la même donnée dans ke mini batch
            while index in listIndex:
                index = randint(0,init-1)
            listIndex.append(index)

            #On rempli le minibatch
            states[k] = self.states[index]
            actions[k] = self.actions[index]
            nextstates[k] = self.nextstates[index]
            rewards[k] = self.rewards[index]
            done[k] = self.done[index]

        return states, actions, nextstates, rewards, done