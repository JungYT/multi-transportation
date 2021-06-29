import numpy as np
import numpy.linalg as nla
import os
from datetime import datetime
from multiprocessing import Pool
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from matplotlib import pyplot as plt
import time
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.art3d as art3d
import mpl_toolkits.mplot3d.axes3d as ax3d
from matplotlib.patches import Circle
import timeit

from fym.core import BaseEnv, BaseSystem
import fym.core as core
import fym.logging as logging
from fym.utils import rot
from utils import hat


class TestSystem(BaseEnv):
    def __init__(self):
        super().__init__()
        self.pos = BaseSystem(np.vstack((0., 0.)))
        self.vel = BaseSystem(np.vstack((0., 0.)))

    def set_dot(self, u):
        self.pos.dot = self.vel.state
        self.vel.dot = np.vstack((u, -u))


class TestEnv(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.1, max_t=1, solver='odeint', ode_step_len=10)

        self.sys = TestSystem()

    def reset(self):
        super().reset()

    def set_dot(self, t, u):
        self.sys.set_dot(u)
        self.method_in_setdot(self.sys.pos.state, self.sys.vel.state)
        self.method_not_in_setdot()

    def method_in_setdot(self, pos, vel):
        print('in setdot pos:', pos)
        print('in setdot vel:', vel)

    def method_not_in_setdot(self):
        print('not in setdot pos:', self.sys.pos.state)
        print('not in setdot pos:', self.sys.vel.state)

    def step(self):
        *_, done = self.update(u=1)
        return done

def main():
    sys = TestEnv()
    sys.reset()
    while True:
        done = sys.step()
        if done:
            break
    sys.close()


if __name__ == "__main__":
    main()








