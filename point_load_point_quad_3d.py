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

"""
Global variables which are not related during parallel simulation

    animation: whether make & save gif file or not
    epi_num: total amount of episode used to learn
    eval_interval: how many episodes are trained before evaluating policy
    time_step: time step of simulation
    time_max: desired time which simulation is terminated in
    g: gravity vector

    load_mass: mass of payload
    load_pos_init: initial position of load when evaluating policy
    load_posx_rand: boundary of payload's initial position x-component
    load_posz_rand: boundary of payload's initial position z-component

    quad_num: amount of quadrotors
    quad_mass: mass of each quadrotors
    input_saturation: magnitude limit of quadrotor's input
    controller_chatter_bound: epsilon in sliding mode controller for suppressing chattering
    controller_K: control gain of SMC
    quad_reach_criteria: criteria whether quadrotor reaches to desired position or not
    action_scaled: proportional constant for scaling action,
                    a in y = a*x + b
    action_scaled_bias: bias constant for scaling action,
                    b in y = a*x + b

    link_len: length of link
    link_ang_rand: boundary of initial half of angle between two quadrotors

    reference: desired load position
    action_size: size of action vector
    state_size: size of state vector
    ref_size: size of reference vector
"""
animation = False
epi_num = 2
eval_interval = 1
time_step = 0.01
time_max = 10
g = 9.81 * np.vstack((0.0, 0.0, 1.0))

load_mass= 1.0
load_pos_init = np.vstack((0.0, 0.0, -1.0))
load_posxy_rand = [-3, 3]
load_posz_rand = [-1, -5]

quad_num = 2
quad_mass = 1.0
input_saturation = 100
controller_chatter_bound = 0.1
controller_K = 5
quad_reach_criteria = 0.05
action_scaled = [1, 1, np.pi]

link_len = 1.0

radius_quad = 0.5

reference = np.array([0.0, -5.0])
action_size = 4
state_size = 8
ref_size = 3

"""
Variables which are design parameters

    dis_factor: discount factor of DDPG (gamma)
    actor_learning_rate: learning rate of actor
    critic_learning_rate: learning rate of critic
    softupdate_const: time constant of soft update (tau)
    memory_size: maximum amount of item(data) saved for learning
    batch_size: size of batch update
    reward_weight: weight of reward,
                    load's x-position error,
                    load's z-position error,
                    half of angle between two quadrotos
                    in order
"""
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
            targetParam.data*(1.0-softupdate_const) + behaviorParam.data*softupdate_const
        )

def hardupdate(target, behavior):
    for targetParam, behaviorParam in zip(target.parameters(), behavior.parameters()):
        targetParam.data.copy_(behaviorParam.data)

class Load(BaseEnv):
    """
    A Class to represent a payload transported by quadrotors.
    The payload is modeled as point mass.
    """
    def __init__(self):
        """
        Make base systems for position and velocity of the payload.
        """
        super().__init__()
        self.pos = BaseSystem(load_pos_init)
        self.vel = BaseSystem(np.vstack((0.0, 0.0, 0.0)))

    def set_dot(self, load_acc):
        """
        Dynamic equations for the payload.
        It has to be modified if the payload will be modeled with volume.
        """
        vel = self.vel.state
        self.pos.dot = vel
        self.vel.dot = load_acc

class Link(BaseEnv):
    """
    A Class to represent link connecting payload and quadrotor.
    """
    def __init__(self):
        """
        Make base systems for link's unit vector and it's angular rate.
        "link" means unit vector with direction from quadrotor to load.
        "link_ang_rate" means the unit vector's angular rate which is normal to the unit vector.
        """
        super().__init__()
        self.link = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.link_ang_rate = BaseSystem(np.vstack((0.0, 0.0, 0.0)))

    def set_dot(self, u, load_acc):
        """
        Dynamic equations for link.
        """
        q = self.link.state
        w = self.link_ang_rate.state
        self.link.dot = hat(w).dot(q)
        self.link_ang_rate.dot = hat(q).dot(load_acc - g - u/quad_mass) / link_len

class PointMassLoadPointMassQuad3D(BaseEnv):
    """
    A Class to represent intergrated system.
    A point mass load, point mass quadrotors, and 2-D environment.
    Sliding Mode Controller with feedback linearization is used as position controller.
    """
    def __init__(self, reward_weight):
        """
        To terminate when quadrotors reach to desired position after desired terminate time,
        maximum time of simulation, "max_t", is set as 10 times of desired terminate time, "time_max".
        """
        super().__init__(dt=time_step, max_t=10*time_max)
        self.load = Load()
        self.links = core.Sequential(
            **{f"link{i:02d}": Link() for i in range(quad_num)}
        )
        self.reward_weight = reward_weight
        self.center = np.vstack((0.0, 0.0, 0.0))



def run_main(parameters, num_parallel):
    path = os.path.join('log', datetime.today().strftime('%Y%m%d-%H%M%S'), f"{num_parallel}")
    parameter_logger = logging.Logger(
        log_dir=path, file_name="parameters.h5"
    )
    parameter_logger.record(**parameters)
    parameter_logger.close()


if __name__ == "__main__":
    main(parameters)

