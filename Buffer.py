from random import *

class Buffer:
    def __init__(self, sizeBuffer):
        self.statess = []
        self.actions = []
        self.nextstates = []
        self.rewards = []
        self.done = []

        self.sizeBuffer = sizeBuffer

        self.index = 0
        self.init = 0

    def add(self, data):
        self.statess[self.index] = data[0]
        self.actions[self.index] = [data[1]]
        self.nextstates[self.index] = data[2]
        self.rewards[self.index] = [data[3]]
        self.done[self.index] = [data[4]]

        self.index = self.index + 1
        if self.index == self.sizeBuffer:
            self.init = 1
            self.index = 0

    def get_minibatch(self, sizeBatch):
        states = []
        actions = []
        nextstates = []
        rewards = []
        done = []

        listIndex = []

        if self.init == 0:
            init = self.index
        else:
            init = self.sizeBuffer

        for k in sizeBatch:
            index = randint(init)
            while index in listIndex:
                index = randint(init)
            listIndex.append(index)
            states.append(self.statess[index])
            actions.append(self.actions[index])
            nextstates.append(self.nextstates[index])
            rewards.append(self.rewards[index])
            done.append(self.done[index])

        return states, actions, nextstates, rewards, done