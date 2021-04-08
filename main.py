"""
Simulation for multi-quadrotors transporting with slung load.

Payload: point mass
Quadrotor: point mass
Dimension: 2-D

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
animation = True
epi_num = 2
eval_interval = 1
time_step = 0.01
time_max = 10
g = 9.81 * np.vstack((0.0, 0.0, 1.0))

load_mass= 1.0
load_pos_init = np.vstack((0.0, 0.0, -1.0))
load_posx_rand = [-3, 3]
load_posz_rand = [-1, -5]

quad_num = 2
quad_mass = 1.0
input_saturation = 100
controller_chatter_bound = 0.1
controller_K = 5
quad_reach_criteria = 0.05
action_scaled = [1, 1, np.pi / 8]
action_scaled_bias = [0, 0, np.pi * 5 / 24]

link_len = 1.0
link_ang_rand = [np.pi / 12, np.pi / 2]

reference = np.array([0.0, -5.0])
action_size = 3
state_size = 6
ref_size = 2

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

class PointMassLoadPointMassQuad2D(BaseEnv):
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

    def reset(self, fixed_init=False):
        """
        If "fixed_init" is true, initial states of the quadrotors and load are set as predifined values.
        If not, position of load and angle of quadrotors are set randomly.
        Velocitiy and angular rate of link are set as zero.
        """
        super().reset()
        if fixed_init:
            phi = np.pi / 4
            self.links.link00.link.state = np.vstack((np.sin(phi), 0.0, np.cos(phi)))
            self.links.link01.link.state = np.vstack((-np.sin(phi), 0.0, np.cos(phi)))
        else:
            self.load.pos.state = np.vstack(
                (np.random.uniform(low=load_posx_rand[0], high=load_posx_rand[1]),
                 0.0,
                 np.random.uniform(low=load_posz_rand[0], high=load_posz_rand[1]))
            )
            phi = np.random.uniform(low=link_ang_rand[0], high=link_ang_rand[1])
            self.links.link00.link.state = np.vstack((np.sin(phi), 0.0, np.cos(phi)))
            self.links.link01.link.state = np.vstack((-np.sin(phi), 0.0, np.cos(phi)))
        x = self.observe()
        return x

    def step(self, action, quad_des_pos, reference):
        quad_input = self.control_quad_pos(quad_des_pos)
        *_, done = self.update(u=quad_input)
        r = self.get_reward(action, reference)
        x = self.observe()
        quad_pos = self.convert_link2quad()
        distance = np.linalg.norm(quad_des_pos[0:3] - np.array(quad_pos[0]))
        done, time = self.terminate(distance)
        info = {
            'quad_pos': self.convert_link2quad(),
            'quad_input': quad_input,
            'distance': distance,
            'time': time,
            'load_pos': self.load.pos.state,
            'load_vel': self.load.vel.state,
            'theta': x[4],
            'theta_dot': x[5],
            'quad_des_pos': quad_des_pos,
            'action': action,
            'done': done,
            'reward': r,
            'reference': reference,
        }
        return x, r, done, info

    def terminate(self, distance):
        """
        If quadrotors reach at desired position and simulation time over the predifined maximum time,
        simulation is done.
        Also, if quadrotors reach at desired position and load's height is under the ground,
        which means z-component of load's position vector is positive, simulation is done.
        """
        time = self.clock.get()
        load_posz = self.load.pos.state[2]
        done = 1. if distance < quad_reach_criteria and (time > time_max or load_posz) > 0 else 0.
        return done, time

    def set_dot(self, t, u):
        """
        Dynamic equations of intergated system.
        """
        total_mass = load_mass + quad_num * quad_mass
        Mq = total_mass * np.eye(3)
        S = np.vstack((0.0, 0.0, 0.0))
        for i, system in enumerate(self.links.systems):
            q = system.link.state
            w = system.link_ang_rate.state
            q_hat = hat(q)
            Mq += quad_mass * q_hat.dot(q_hat)
            S += quad_mass * (
                q_hat.dot(q_hat.dot(g))
                - link_len * (np.transpose(w).dot(w)) * q
            ) + (np.eye(3) + q_hat.dot(q_hat)).dot(u[3 * i : 3 * (i + 1)])
        S += total_mass * g
        x_ddot = np.linalg.inv(Mq).dot(S)
        self.load.set_dot(x_ddot)
        self.links.link00.set_dot(u[0:3], x_ddot)
        self.links.link01.set_dot(u[3:6], x_ddot)

    def get_reward(self, action, reference):
        """
        If z-component of load's position vector is negative, which means load crashes,
        or is over the 2.1 times reference height, reward is -200.
        Else, it is sum of error between load's position and reference,
        and half of angle between two quadrotors.
        """
        load_pos = np.delete(self.load.pos.state, 1)
        if load_pos[1] > 0 or load_pos[1] < 2.1 * reference[1]:
            r = -200
        else:
            e = load_pos - reference
            x = np.append(e, action[2])
            r = -np.transpose(x).dot(self.reward_weight.dot(x))
        return np.array([r])

    def control_quad_pos(self, quad_des_pos):
        """
        To control quadrotor's position, Sliding Mode Control with feedback linearization is used.
        This controller only considers quadrotor by assumming load's effect as disturbance.
        """
        uncert_max = 0.5 * load_mass * 9.81 * np.vstack((1, 1, 1))
        u_set = []
        for i, system in enumerate(self.links.systems):
            q = system.link.state
            w = system.link_ang_rate.state
            x0 = self.load.pos.state
            x0dot = self.load.vel.state
            e = x0 - link_len * q - quad_des_pos[3 * i : 3 * (i + 1)]
            vel = x0dot - link_len * hat(w).dot(q)
            s = controller_K * e + vel
            s_clip = np.clip(s / controller_chatter_bound, -1, 1)

            mu = -controller_K * vel - (uncert_max + np.vstack((1, 1, 1))) * s_clip
            u = quad_mass * (-g - load_mass/quad_mass * (np.transpose(g).dot(q)) * q + mu)

            u_norm = np.linalg.norm(u)
            if u_norm > input_saturation:
                u = input_saturation * u / u_norm
            u_set.append(u)
        u_return = np.vstack(u_set)
        return u_return

    def convert_link2quad(self):
        """
        Convert link's unit vector to quadrotor's position.
        """
        quad_pos = [
            self.load.pos.state - link_len*system.link.state for system in self.links.systems
        ]
        return quad_pos

    def observe(self):
        """
        Convert intergrated system's state to DDPG's state.
        Here, intergrated system's state are
            1. payload's position and velocity vector in 3 dimension.
            2. each links' unit vector and it's angular rate in 3 dimension.
        DDPG's state are
            1. payload's position and velocity vecot rin 2 dimension.
            2. Half of angle between two quadrotors and it's rate.
        """
        load_pos2D = np.delete(self.load.pos.state, 1)
        load_vel2D = np.delete(self.load.vel.state, 1)
        theta = np.arctan2(
            self.links.link00.link.state[0],
            self.links.link00.link.state[2]
        )
        theta_dot = self.links.link00.link_ang_rate.state[1]
        obs = np.hstack((load_pos2D, load_vel2D, theta, theta_dot))
        return obs

    def convert_action2quad_des_pos(self, action):
        """
        Convert DDPG's action to quadrotor's desired position.
        Here, DDPG's action are
            1. Displacement of system along x and z direction. (2 elements)
            2. Desired half of angle between two quadrotors. (1 elements)
        """
        load_pos = self.load.pos.state
        phi = action[2]
        quad_posz_rel2load = action[1] - link_len * np.cos(phi)

        u = [load_pos
             + np.vstack((
                 action[0] + ((-1)**(i+1)) * link_len * np.sin(phi),
                 0,
                 quad_posz_rel2load))
             for i in range(quad_num)]
        u_return = np.vstack(u)
        return u_return

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.lin1 = nn.Linear(state_size, 64)
        self.lin2 = nn.Linear(64+ref_size, 32)
        self.lin3 = nn.Linear(32, 16)
        self.lin4 = nn.Linear(16, action_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state_w_ref):
        x, r = state_w_ref
        x1 = self.relu(self.lin1(x.view(-1,6)))
        x2 = self.relu(self.lin2(
            torch.cat([x1, r.view(-1, ref_size)], 1)
        ))
        x3 = self.relu(self.lin3(x2))
        x4 = self.tanh(self.lin4(x3))
        xScaled = x4 * torch.Tensor(action_scaled) + torch.Tensor(action_scaled_bias)
        return xScaled

class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.lin1 = nn.Linear(state_size, 64)
        self.lin2 = nn.Linear(64+action_size, 32)
        self.lin3 = nn.Linear(32, 16)
        self.lin4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, state_w_action):
        x, a = state_w_action
        x1 = self.relu(self.lin1(x))
        x2 = self.relu(self.lin2(torch.cat([x1, a], 1)))
        x3 = self.relu(self.lin3(x2))
        x4 = self.lin4(x3)
        return x4

class DDPG:
    def __init__(self, dis_factor, actor_learning_rate, critic_learning_rate,
                 softupdate_const, memory_size, batch_size):
        self.memory = deque(maxlen=memory_size)
        self.behavior_actor = ActorNet().float()
        self.behavior_critic = CriticNet().float()
        self.target_actor = ActorNet().float()
        self.target_critic = CriticNet().float()
        self.actor_optim = optim.Adam(
            self.behavior_actor.parameters(), lr=actor_learning_rate
        )
        self.critic_optim = optim.Adam(
            self.behavior_critic.parameters(), lr=critic_learning_rate
        )
        self.mse = nn.MSELoss()
        hardupdate(self.target_actor, self.behavior_actor)
        hardupdate(self.target_critic, self.behavior_critic)
        self.dis_factor = dis_factor
        self.softupdate_const = softupdate_const
        self.batch_size = batch_size

    def get_action(self, state, reference, net="behavior"):
        with torch.no_grad():
            if net == "behavior":
                action = self.behavior_actor(
                    [torch.FloatTensor(state), torch.FloatTensor(reference)]
                )
            else:
                action = self.target_actor(
                    [torch.FloatTensor(state), torch.FloatTensor(reference)]
                )
        return np.array(np.squeeze(action))

    def memorize(self, item):
        self.memory.append(item)

    def get_sample(self):
        sample = random.sample(self.memory, self.batch_size)
        state, action, reward, state_next, epi_done, reference = zip(*sample)
        x = torch.tensor(state, requires_grad=True).float()
        u = torch.tensor(action, requires_grad=True).float()
        r = torch.tensor(reward, requires_grad=True).float()
        xn = torch.tensor(state_next, requires_grad=True).float()
        done = torch.tensor(epi_done, requires_grad=True).float().view(-1,1)
        ref = torch.tensor(reference, requires_grad=True).float()
        return x, u, r, xn, done, ref

    def train(self):
        x, u, r, xn, done, ref = self.get_sample()

        with torch.no_grad():
            action = self.target_actor([xn, ref])
            Qn = self.target_critic([xn, action])
            target = r + (1-done) * self.dis_factor * Qn
        Q_w_noise_action = self.behavior_critic([x, u])
        self.critic_optim.zero_grad()
        critic_loss = self.mse(Q_w_noise_action, target)
        critic_loss.backward()
        self.critic_optim.step()

        Q = self.behavior_critic([x, self.behavior_actor([x, ref])])
        self.actor_optim.zero_grad()
        actor_loss = torch.sum(-Q)
        actor_loss.backward()
        self.actor_optim.step()

        softupdate(self.target_actor, self.behavior_actor, self.softupdate_const)
        softupdate(self.target_critic, self.behavior_critic, self.softupdate_const)

def main(dis_factor, actor_learning_rate, critic_learning_rate, softupdate_const,
         memory_size, batch_size, reward_weight):
    env = PointMassLoadPointMassQuad2D(reward_weight)
    agent = DDPG(dis_factor, actor_learning_rate, critic_learning_rate,
                 softupdate_const, memory_size, batch_size)
    noise = OrnsteinUhlenbeckNoise()
    parameter_logger = logging.Logger()
    parameters = {
        'dis_factor': dis_factor,
        'actor_learning_rate': actor_learning_rate,
        'critic_learning_rate': critic_learning_rate,
        'softupdate_const': softupdate_const,
        'memory_size': memory_size,
        'batch_size': batch_size,
        'reward_weight': reward_weight,
    }
    parameter_logger.record(**parameters)
    parameter_logger.close()

    for epi in tqdm(range(epi_num)):
        x = env.reset()
        noise.reset()
        train_logger = logging.Logger(
            log_dir=train_path, file_name=f"data_{epi:05d}.h5"
        )
        while True:
            u = agent.get_action(x, reference) + noise.get_noise()
            u[2] = np.clip(u[2], np.pi/12, np.pi/6)
            quad_des_pos = env.convert_action2quad_des_pos(u)
            while True:
                xn, r, done, info = env.step(u, quad_des_pos, reference)
                train_logger.record(**info)
                if info['distance'] < quad_reach_criteria:
                    break
            r_scaled = (r + 100) / 100
            item = (x, u, r_scaled, xn, done, reference)
            agent.memorize(item)
            x = xn
            if len(agent.memory) > 300:
                agent.train()
            if done:
                break
        train_logger.close()

        eval_trigger = divmod(epi+1, eval_interval)
        if eval_trigger[1] == 0:
            if animation:
                fig = plt.figure()
                camera = Camera(fig)
            x = env.reset(fixed_init=True)
            eval_logger = logging.Logger(
                log_dir=eval_path,
                file_name=f"data_trained_{(eval_trigger[0])*eval_interval:05d}.h5"
            )
            while True:
                u = agent.get_action(x, reference, net="target")
                quad_des_pos = env.convert_action2quad_des_pos(u)
                while True:
                    xn, r, done, info = env.step(u, quad_des_pos, reference)
                    eval_logger.record(**info)
                    if animation:
                        snap_GIF(info['load_pos'], info['quad_pos'], quad_des_pos, epi)
                        camera.snap()
                    if info['distance'] < quad_reach_criteria:
                        break
                x = xn
                if done:
                    break
            if animation:
                ani = camera.animate(interval=10, blit=True)
                eval_gif = os.path.join(eval_path, f"{epi+1}_ani.gif")
                ani.save(eval_gif)
            eval_logger.close()
            plt.close('all')
    env.close()

def make_figure(path, epi_num_trained, save_fig=True):
    eval_load = os.path.join(path, f"data_trained_{epi_num_trained:05d}.h5")
    data = logging.load(eval_load)

    time = data['time']
    quad_des_pos = data['quad_des_pos']
    quad_pos = data['quad_pos']
    load_pos = data['load_pos']
    reference = data['reference']
    quad_input = data['quad_input']
    action = data['action']
    theta = data['theta']
    distance = data['distance']
    reward = data['reward']

    G = 0
    for r in reward[::-1]:
        #G = (r+100)/100 + dis_factor * G
        G = r.item() + dis_factor * G

    fig1, ax1 = plt.subplots(nrows=2, ncols=1)
    line1, = ax1[0].plot(time, load_pos[:,0], 'r')
    line2, = ax1[0].plot(time, reference[:,0], 'b--')
    ax1[0].legend(handles=(line1, line2), labels=('load', 'reference'))
    ax1[1].plot(time, -load_pos[:,2], 'r', time, -reference[:,1], 'b--')
    ax1[0].set_title(f"Position of load after training {epi_num_trained:05d} Epi.")
    ax1[1].set_xlabel('time [s]')
    ax1[0].set_ylabel('x [m]')
    ax1[1].set_ylabel('height [m]')
    ax1[0].grid(True)
    ax1[1].grid(True)

    fig2, ax2 = plt.subplots(nrows=3, ncols=1)
    line3, = ax2[0].plot(time, quad_pos[:,0,0,:], 'r')
    line4, = ax2[0].plot(time, quad_des_pos[:,0], 'b--')
    ax2[0].legend(handles=(line3, line4), labels=('quad', 'desired'))
    ax2[1].plot(time, -quad_pos[:,0,2,:], 'r', time, -quad_des_pos[:,2], 'b--')
    ax2[2].plot(time, theta*180/np.pi, 'r', time, action[:,2]*180/np.pi, 'b--')
    ax2[0].set_title(
        f"Position and angle of quad0 after training {epi_num_trained:05d} Epi."
    )
    ax2[1].set_xlabel('time [s]')
    ax2[0].set_ylabel('x [m]')
    ax2[1].set_ylabel('height [m]')
    ax2[2].set_ylabel('$\\theta$ [deg]')
    ax2[0].grid(True)
    ax2[1].grid(True)
    ax2[2].grid(True)

    fig3, ax3 = plt.subplots(nrows=2, ncols=1)
    line5, = ax3[0].plot(time, quad_input[:,0], 'r')
    line6, = ax3[0].plot(time, quad_input[:,2], 'b')
    ax3[0].legend(handles=(line5, line6), labels=('quad0', 'quad1'), loc='upper left')
    ax3[1].plot(time, quad_input[:,3], 'r', time, quad_input[:,5], 'b')
    ax3[0].set_title(f"Quadrotors' input after training {epi_num_trained:05d} Epi.")
    ax3[1].set_xlabel('time [s]')
    ax3[0].set_ylabel('x direction [N]')
    ax3[1].set_ylabel('z direction [N]')
    ax3[0].grid(True)
    ax3[1].grid(True)

    if save_fig:
        fig1.savefig(
            os.path.join(path, f"{epi_num_trained:05d}_load_pos"),
            bbox_inches='tight'
        )
        fig2.savefig(
            os.path.join(path, f"{epi_num_trained:05d}_quad_pos"),
            bbox_inches='tight'
        )
        fig3.savefig(
            os.path.join(path, f"{epi_num_trained:05d}_quad_input"),
            bbox_inches='tight'
        )
    #plt.show()
    plt.close('all')
    return G

def snap_GIF(load_pos, quad_pos, quad_des_pos, epi):
    plt.grid(True)
    plt.axis([-10, 10, -3, 10])
    plt.plot(
        [load_pos[0], quad_pos[0][0]], [-load_pos[2], -quad_pos[0][2]], "k", linewidth=0.5
    )
    plt.plot(
        [load_pos[0], quad_pos[1][0]], [-load_pos[2], -quad_pos[1][2]], "k", linewidth=0.5
    )
    plt.scatter(reference[0], -reference[1], facecolors='y', edgecolors='y')
    plt.scatter(load_pos[0], -load_pos[2], facecolors='none', edgecolors="r")
    plt.scatter(quad_pos[0][0], -quad_pos[0][2], facecolors='none', edgecolors="b")
    plt.scatter(quad_pos[1][0], -quad_pos[1][2], facecolors='none', edgecolors="b")
    plt.scatter(quad_des_pos[0], -quad_des_pos[2], facecolors='none', edgecolors="k")
    plt.scatter(quad_des_pos[3], -quad_des_pos[5], facecolors='none', edgecolors="k")
    plt.title(f"{epi + 1} episodes are learned")


if __name__ == "__main__":

    """
    past = -1
    data_num = 12000
    loglist = sorted(os.listdir("./log"))
    load_path = os.path.join("log", loglist[past], 'eval')
    figure(load_path, data_num)
    """

    eval_path = os.path.join(
        'log', datetime.today().strftime('%Y%m%d-%H%M%S'), 'eval'
    )
    train_path = os.path.join(
        'log', datetime.today().strftime('%Y%m%d-%H%M%S'), 'train'
    )

    """
    profiler = Profile()
    profiler.runcall(main)
    stats = Stats(profiler)
    stats.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)
    """

    main(dis_factor, actor_learning_rate, critic_learning_rate, softupdate_const,
         memory_size, batch_size, reward_weight)
    plt.close('all')

    return_his = np.array(
        [[(i+1) * eval_interval, make_figure(eval_path, (i+1) * eval_interval)]
         for i in range(epi_num//eval_interval)]
    )

    fig4, ax4 = plt.subplots(nrows=1, ncols=1)
    ax4.plot(return_his[:,0], return_his[:,1], "*")
    ax4.grid(True)
    ax4.set_title(f"Return history with {epi_num:d} episode")
    ax4.set_xlabel('Number of trained episode')
    ax4.set_ylabel('Return')
    fig4.savefig(
        os.path.join(eval_path, f"Return_{epi_num:d}"),
        bbox_inches='tight'
    )
    plt.close('all')
    end_logger = logging.Logger(
        log_dir=eval_path,
        file_name="return_his.h5")
    end_logger.record(return_his=return_his)
    end_logger.close()


