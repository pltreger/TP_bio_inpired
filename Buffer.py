from random import *
import numpy

class Buffer:
    def __init__(self, sizeBuffer):
        #On initialise la mémoire
        self.states = numpy.zeros((sizeBuffer,4))
        self.actions = numpy.zeros(sizeBuffer)
        self.nextstates = numpy.zeros((sizeBuffer,4))
        self.rewards = numpy.zeros(sizeBuffer)
        self.done = numpy.zeros(sizeBuffer)

        self.sizeBuffer = sizeBuffer

        #Index correspondra à l'endroit où on se trouve dans la mémoire pour la remplir
        self.index = 0
        #Init nous indiquera si la mémoire a déjà été entièrement remplie
        self.init = 0

    def add(self, state, action, next_state, reward, done):
        #On ajoute les nouvelles données dans la mémoire
        self.states[self.index] = state
        self.actions[self.index] = action
        self.nextstates[self.index] = next_state
        self.rewards[self.index] = reward
        self.done[self.index] = done

        #On modifie l'index
        self.index = self.index + 1
        if self.index == self.sizeBuffer:
            self.init = 1
            self.index = 0

    def get_minibatch(self, sizeBatch):
        states = numpy.zeros((sizeBatch, 4))
        actions = numpy.zeros(sizeBatch)
        nextstates = numpy.zeros((sizeBatch, 4))
        rewards = numpy.zeros(sizeBatch)
        done = numpy.zeros(sizeBatch)

        listIndex = []

        #Si c'est la premiere fois que l'on parcourt le remplissage de la mémoire
        if self.init == 0:
            init = self.index
        #Si la mémoire est déja remplie
        else:
            init = self.sizeBuffer


        for k in range(sizeBatch):
            index = randint(0,init)
            #On fait en sorte qu'il n'y ait pas la même donnée dans ke mini batch
            while index in listIndex:
                index = randint(0,init)
            listIndex.append(index)

            #On rempli le minibatch
            states[k] = self.states[index]
            actions[k] = self.actions[index]
            nextstates[k] = self.nextstates[index]
            rewards[k] = self.rewards[index]
            done[k] = self.done[index]

        return states, actions, nextstates, rewards, done