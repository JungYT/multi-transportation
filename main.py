"""
Simulation for multi-quadrotors transporting with slung load.

Payload: point mass
Quadrotor: point mass
Dimension: 3-D

Reference
Taeyoung Lee,
"Geometric Control of Quadrotor UAVs Transporting a Cable-suspended Rigid Body,"
IEEE TRANSACTIONS ON CONTROL SYSTEMS TECHNOLOGY, vol. 26, no. 1, Jan, 2018.
"""
import numpy as np
import random
import os
from tqdm import tqdm
from datetime import datetime
from cProfile import Profile
from pstats import Stats, SortKey
from collections import deque
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from fym.core import BaseEnv, BaseSystem
import fym.core as core
import fym.logging as logging
from celluloid import Camera

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Simulation parameters
animation = False
epi_num = 2
epi_eval_interval = 1
time_max = 10
time_step = 0.01
goal = np.vstack((0.0, 0.0, -5.0))

# RL module parameters
action_scaled = [1, 1, np.pi]
dis_factor = 0.999
actor_learning_rate = 0.0001
critic_learning_rate = 0.001
softupdate_const = 0.001
memory_size = 20000
batch_size = 64
reward_weight = np.array((
    [4, 0, 0],
    [0, 1, 0],
    [0, 0, 0.01]
))
action_size = 4
state_size = 9
goal_size = 3

# Dynamic system parameters
load_mass = 1.0
quad_mass = 1.0
link_len = 1.0
quad_num = 2
input_saturation = 100
controller_chatter_bound = 0.1
controller_K = 5
quad_reach_criteria = 0.05
g = 9.81 * np.vstack((0.0, 0.0, 1.0))

class OrnsteinUhlenbeckNoise:
    def __init__(self, x0=None):
        self.rho = 0.15
        self.mu = 0
        self.sigma = 0.2
        self.dt = 0.1
        self.x0 = x0
        self.size = action_size
        self.reset()

    def reset(self):
        self.x = self.x0 if self.x0 is not None else np.zeros(self.size)

    def get_noise(self):
        x = (
            self.x
            + self.rho * (self.mu-self.x) * self.dt
            + np.sqrt(self.dt) * self.sigma * np.random.normal(size=self.size)
        )
        self.x = x
        return x

def hat(v):
    v1, v2, v3 = v.squeeze()
    return np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])

def wrap(angle):
    angle_wrap = (angle+np.pi) % (2*np.pi) - np.pi
    return angle_wrap

def softupdate(target, behavior, softupdate_const):
    for targetParam, behaviorParam in zip(target.parameters(), behavior.parameters()):
        targetParam.data.copy_(
            targetParam.data*(1.0-softupdate_const)\
            + behaviorParam.data*softupdate_const
        )

def hardupdate(target, behavior):
    for targetParam, behaviorParam in zip(target.parameters(), behavior.parameters()):
        targetParam.data.copy_(behaviorParam.data)

class Load(BaseEnv):
    def __init__(self):
        super().__init__()
        self.pos = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.vel = BaseSystem(np.vstack((0.0, 0.0, 0.0)))

    def set_dot(self, load_acc):
        vel = self.vel.state
        self.pos.dot = vel
        self.vel.dot = load_acc

class Link(BaseEnv):
    def __init__(self):
        super().__init__()
        self.link = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.link_ang_rate = BaseSystem(np.vstack((0.0, 0.0, 0.0)))

    def set_dot(self, u, load_acc):
        q = self.link.state
        w = self.link_ang_rate.state
        self.link.dot = hat(w).dot(q)
        self.link_ang_rate.dot = hat(q).dot(load_acc - g - u/quad_mass) / link_len

class PointMassLoadPointMassQuad3D(BaseEnv):
    def __init__(self):
        super().__init__(dt=time_step, max_t=10*time_max)
        self.load = Load()
        self.links = core.Sequential(
            **{f"link{i:02d}": Link() for i in range(quad_num)}
        )

    def reset(self, fixed_init=False):
        super().reset()
        if fixed_init:

