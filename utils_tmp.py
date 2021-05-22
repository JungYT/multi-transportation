import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import fym.logging as logging
from fym.utils import rot


class OrnsteinUhlenbeckNoise:
    def __init__(self, rho, sigma, dt, size, x0=None):
        self.rho = rho
        self.mu = 0
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset()

    def reset(self):
        self.x = self.x0 if self.x0 is not None else np.zeros(self.size)

    def get_noise(self):
        self.x = self.x + self.rho*(self.mu-self.x)*self.dt \
            + np.sqrt(self.dt)*self.sigma*np.random.normal(size=self.size)
        return self.x

def softupdate(target, behavior, softupdate_const):
    for targetParam, behaviorParam in zip(
            target.parameters(),
            behavior.parameters()
    ):
        targetParam.data.copy_(
            targetParam.data*(1.0-softupdate_const) \
            + behaviorParam.data*softupdate_const
        )

def hardupdate(target, behavior):
    for targetParam, behaviorParam in zip(
            target.parameters(),
            behavior.parameters()
    ):
        targetParam.data.copy_(behaviorParam.data)

def hat(v):
    v1, v2, v3 = v.squeeze()
    return np.array([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0]
    ])

def wrap(angle):
    return (angle+np.pi) % (2*np.pi) - np.pi

def data_arrangement(data):

