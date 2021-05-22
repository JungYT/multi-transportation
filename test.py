import numpy as np
import os
from datetime import datetime
from multiprocessing import Pool
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from matplotlib import pyplot as plt
import time

from fym.core import BaseEnv, BaseSystem
import fym.core as core
import fym.logging as logging


class Test(BaseEnv):
    def __init__(self, a, b):
        super().__init__(dt=0.1, max_t=5, solver="rk4")
        #self.pos = BaseSystem(np.array([1, 2]))
        #self.vel = BaseSystem(np.array([0, 3]))
        self.pos = BaseSystem(np.vstack((1, 2)))
        self.vel = BaseSystem(np.vstack((0, 3)))
        self.a = a
        self.b = b

    def set_dot(self, t):
        pos = self.pos.state
        vel = self.pos.state
        self.pos.dot = vel
        self.vel.dot = -self.a * vel - self.b * pos

    def reset(self):
        super().reset()

    def step(self):
        *_, done = self.update()
        info = {
            'pos': self.pos.state,
            'vel': self.vel.state,
        }
        return done, info

    def logger_callback(self, i, t, y, *args):
        return dict(time=t, pos=self.pos.state)

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.lin1 = nn.Linear(3, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 16)
        self.lin4 = nn.Linear(16, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.relu(self.lin1(x))
        x2 = self.relu(self.lin2(x1))
        x3 = self.relu(self.lin3(x2))
        x4 = self.tanh(self.lin4(x3))
        return x4

def main():
    # memory = deque()
    # x = np.array([1, 2, 3])
    # temp = np.array([4, 5])
    # item = (x, temp)
    # actor = ActorNet()
    # for i in range(10):
    #     memory.append(item)
    # x = np.vstack((1, 2, 3))
    # state, action = zip(*memory)
    # y = actor(torch.FloatTensor(x))
    # env = Test(1, 2)
    # env.logger = logging.Logger("temp.h5")
    # env.reset()
    # done, info = env.step()
    # env.close()
    # data = logging.load("temp.h5")
    # breakpoint()

    ite = 1
    start1 = time.time()
    for k in range(ite):
        a1 = [i + 100*np.sqrt(2) for i in range(3)]
        b1 = [i + 100*np.sqrt(2) for i in range(3)]
        c1 = [i + 100*np.sqrt(2) for i in range(3)]
    end1 = time.time()
    start2 = time.time()
    for k in range(ite):
        a = []
        b = []
        c = []
        for i in range(3):
            a.append(i + 100*np.sqrt(2))
            b.append(i + 100*np.sqrt(2))
            c.append(i + 100*np.sqrt(2))
    end2 = time.time()

    print("time1: :", end1 - start1)
    print("time2: :", end2 - start2)



def test():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot3D(1, 2, 3, marker="X")
    plt.show()



if __name__ == "__main__":
    main()
