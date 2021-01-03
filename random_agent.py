import argparse
import sys
from Buffer import Buffer
import matplotlib.pyplot as plt

import gym
from gym import wrappers, logger

from Neural_Network import Neural_Network


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


def afficher_graphique(reward, etapes):
    plt.title("Apprentissage")
    plt.plot(etapes, reward)
    plt.title("Reward")
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make('CartPole-v1')

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 1000
    reward = 0
    done = False
    sizeBuffer = 1000000
    sizeBatch = 100
    gamma = 0.9

    neural_network = Neural_Network(env.action_space.n, sizeBuffer)

    rewards = []
    etapes = []

    for i in range(episode_count):
        ob = env.reset()
        nbInteraction = 0
        totalReward = 0
        while True:
            """""choix de la stratégie :
            strategie = "aleatoire"
            strategie = "e-greedy", epsilon =
            strategie = "boltzmann", tau ="""""
            action = neural_network.get_action(ob,strategie="e-greedy",epsilon=0.1)
            ob_next, reward, done, _ = env.step(action)
            neural_network.add_memoire(ob, action, ob_next, reward, done)

            ob = ob_next

            nbInteraction = nbInteraction + 1
            totalReward = totalReward + reward

            neural_network.learn(sizeBatch, env.action_space.n, gamma)

            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

        print("Trace : episode : " + str(i) + " ; nb interactions : " + str(nbInteraction) + " ; recompenses : " +str(totalReward))
        rewards.append(totalReward)
        etapes.append(i)


    # Close the env and write monitor result info to disk
    env.close()
    afficher_graphique(rewards,etapes)


