
import numpy as np
import matplotlib.pyplot as plt

import parameters


def plotAndSave(saver, fig_name):
    for path, data in saver:
        plt.plot(data)
        plt.savefig(fig_name)



class Displayer:

    def __init__(self):
        self.rewards = []

    def add_reward(self, reward):
        self.rewards.append(reward)
        if len(self.rewards) % parameters.PLOT_FREQ == 0:
            self.dispR()
                    
    def dispR(self):
        saver = [("results/", self.rewards)] 
        plotAndSave(saver, "results/Reward.png")

    def displayVI(self,vEpisode,iEpisode,nbPlay):
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(vEpisode)
        axarr[1].plot(iEpisode)
        axarr[0].set_ylabel("v")
        axarr[1].set_ylabel("i")
        plt.savefig("results/VI"+str(nbPlay)+".png")

    def reset(self):
        self.rewards = []

DISPLAYER = Displayer()
