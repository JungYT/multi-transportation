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
import time

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
animation = True
epi_num = 15000
epi_eval_interval = 500
time_max = 10
time_step = 0.01
goal = np.vstack((0.0, 0.0, -6.0))
center_pos_init = np.vstack((3.0, 3.0, -2.0))
theta_init = 0
phi_init = 0
center_pos_rand = [[-5, 5], [-5, 5], [-10, -1]]
theta_rand = [-np.pi, np.pi]
phi_rand = [-np.pi / 3, np.pi / 3]

# RL module parameters
action_scale = [1, 1, 1, np.pi]
reward_scale = 100
dis_factor = 0.999
actor_learning_rate = 0.0001
critic_learning_rate = 0.001
softupdate_const = 0.001
memory_size = 20000
batch_size = 64
reward_weight = np.array((
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
))
action_size = 4
state_size = 8
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
alpha = np.pi / 6

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
            center_pos_init = np.vstack((3.0, 3.0, -2.0))
            theta_init = 0
            phi_init = 0
            load_pos_init, link0_init, link1_init = self.convert_state_learn2dynamics(
                center_pos_init, theta_init, phi_init
            )
            self.load.pos.state = load_pos_init
            self.links.link00.link.state = link0_init
            self.links.link01.link.state = link1_init
        else:
            center_pos_init = np.vstack((
                np.random.uniform(
                    low=center_pos_rand[0][0],
                    high=center_pos_rand[0][1]
                ),
                np.random.uniform(
                    low=center_pos_rand[1][0],
                    high=center_pos_rand[1][1]
                ),
                np.random.uniform(
                    low=center_pos_rand[2][0],
                    high=center_pos_rand[2][1]
                )
            ))
            theta_init = np.random.uniform(
                low=theta_rand[0],
                high=theta_rand[1]
            )
            phi_init = np.random.uniform(
                low=phi_rand[0],
                high=phi_rand[1]
            )
            load_pos_init, link0_init, link1_init = self.convert_state_learn2dynamics(
                center_pos_init, theta_init, phi_init
            )
            self.load.pos.state = load_pos_init
            self.links.link00.link.state = link0_init
            self.links.link01.link.state = link1_init
        state_learn = self.observe()
        return state_learn

    def convert_state_learn2dynamics(self, center_pos, theta, phi):
        """
        phi : load's swing angle
        theta : circle of two quadrotors' rotation angle
        alpha : angle between z-axis and vector from quadrotors to load
        """
        load_pos = center_pos \
            + np.vstack((
                link_len*np.cos(alpha)*np.sin(phi)*np.sin(theta),
                -link_len*np.cos(alpha)*np.sin(phi)*np.cos(theta),
                link_len*np.cos(alpha)*np.cos(phi)
            ))
        #load_vel = self.load.vel.state
        quad0_pos = center_pos \
            + np.vstack((
                link_len*np.sin(alpha)*np.cos(theta),
                link_len*np.sin(alpha)*np.sin(theta),
                0
            ))
        quad1_pos = center_pos \
            - np.vstack((
                link_len*np.sin(alpha)*np.cos(theta),
                link_len*np.sin(alpha)*np.sin(theta),
                0
            ))
        quad02load = load_pos - quad0_pos
        link0 = quad02load / np.linalg.norm(quad02load)
        quad12load = load_pos - quad1_pos
        link1 = quad12load / np.linalg.norm(quad12load)
        return load_pos, link0, link1

    def observe(self):
        load_pos = self.load.pos.state
        load_vel = self.load.vel.state
        link0 = self.links.link00.link.state
        link1 = self.links.link01.link.state
        quad0_pos = load_pos - link_len*link0
        quad1_pos = load_pos - link_len*link1
        center_pos = (quad0_pos+quad1_pos) / 2
        center2quad0 = quad0_pos - center_pos
        center2load = load_pos - center_pos
        z_unit = np.vstack((0.0, 0.0, 1.0))
        normal_vector = np.cross(
            z_unit.reshape(-1,),
            center2quad0.reshape(-1,)
        )
        phi = np.arccos(np.clip(
            np.dot(
                center2load.reshape(-1,)/np.linalg.norm(center2load),
                normal_vector/np.linalg.norm(normal_vector)),
            -1.0,1.0)
        ) - np.pi/2
        theta = np.arctan2(center2quad0[1], center2quad0[0])
        obs = np.hstack((
            center_pos.reshape(-1,),
            theta,
            phi,
            load_vel.reshape(-1,)
        ))
        return obs

    def convert_action2quad_des_pos(self, action):
        center_del_pos = np.vstack(action[0:3])
        theta_des = action[3]
        load_pos = self.load.pos.state
        link0 = self.links.link00.link.state
        link1 = self.links.link01.link.state
        quad0_pos = load_pos - link_len*link0
        quad1_pos = load_pos - link_len*link1
        center_pos = (quad0_pos+quad1_pos) / 2
        center_des_pos = center_pos + center_del_pos
        quad_des_pos = [center_des_pos + ((-1)**(i)) * np.vstack((
            link_len*np.sin(alpha)*np.cos(theta_des),
            link_len*np.sin(alpha)*np.sin(theta_des),
            0)) for i in range(quad_num)]
        return quad_des_pos

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
            e = x0 - link_len * q - quad_des_pos[i]
            vel = x0dot - link_len * hat(w).dot(q)
            s = controller_K * e + vel
            s_clip = np.clip(s / controller_chatter_bound, -1, 1)

            mu = -controller_K * vel - (uncert_max + np.vstack((1, 1, 1))) * s_clip
            u = quad_mass * \
                (-g - load_mass/quad_mass*(np.transpose(g).dot(q)) * q + mu)

            u_norm = np.linalg.norm(u)
            if u_norm > input_saturation:
                u = input_saturation * u / u_norm
            u_set.append(u)
        return u_set

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
            ) + (np.eye(3) + q_hat.dot(q_hat)).dot(u[i])
        S += total_mass * g
        x_ddot = np.linalg.inv(Mq).dot(S)
        self.load.set_dot(x_ddot)
        self.links.link00.set_dot(u[0], x_ddot)
        self.links.link01.set_dot(u[1], x_ddot)

    def get_reward(self, goal):
        load_pos = self.load.pos.state
        state_learn = self.observe()
        center_pos = state_learn[0:3].reshape(-1,1)
        phi = state_learn[4]
        if load_pos[2] > 0:
            r = -100
        else:
            e_pos = center_pos - goal
            e = np.append(e_pos, phi)
            r = -e.dot(reward_weight.dot(e))
        return np.array([r])

    def convert_link2quad(self):
        """
        Convert link's unit vector to quadrotor's position.
        """
        quad_pos = [
            self.load.pos.state - link_len*system.link.state
            for system in self.links.systems
        ]
        return quad_pos

    def step(self, action, quad_des_pos, goal):
        quad_input = self.control_quad_pos(quad_des_pos)
        *_, done = self.update(u=quad_input)
        reward = self.get_reward(goal)
        state_learn = self.observe()
        quad_pos = self.convert_link2quad()
        distance = min(
            np.linalg.norm(quad_des_pos[0] - np.array(quad_pos[0])),
            np.linalg.norm(quad_des_pos[1] - np.array(quad_pos[1]))
        )
        done, time = self.terminate(distance)
        info = {
            'time': time,
            'quad_pos': self.convert_link2quad(),
            'quad_des_pos': quad_des_pos,
            'quad_input': quad_input,
            'distance': distance,
            'load_pos': self.load.pos.state,
            'load_vel': self.load.vel.state,
            'center_pos': state_learn[0:3],
            'theta': state_learn[3],
            'phi': state_learn[4],
            'action': action,
            'done': done,
            'reward': reward,
            'goal': goal,
        }
        return state_learn, reward, done, info

    def terminate(self, distance):
        time = self.clock.get()
        load_posz = self.load.pos.state[2]
        done = 1. if distance < quad_reach_criteria \
            and (time > time_max or load_posz > 0) else 0.
        return done, time

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.lin1 = nn.Linear(state_size+goal_size, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 16)
        self.lin4 = nn.Linear(16, action_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state_w_goal):
        x1 = self.relu(self.lin1(state_w_goal))
        x2 = self.relu(self.lin2(x1))
        x3 = self.relu(self.lin3(x2))
        x4 = self.tanh(self.lin4(x3))
        xScaled = x4 * torch.Tensor(action_scale)
        return xScaled

class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.lin1 = nn.Linear(state_size+action_size, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 16)
        self.lin4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, state_w_action):
        x1 = self.relu(self.lin1(state_w_action))
        x2 = self.relu(self.lin2(x1))
        x3 = self.relu(self.lin3(x2))
        x4 = self.lin4(x3)
        return x4

class DDPG:
    def __init__(self):
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

    def get_action(self, state_w_goal, net="behavior"):
        with torch.no_grad():
            if net == "behavior":
                action = self.behavior_actor(
                    torch.FloatTensor(state_w_goal)
                )
            else:
                action = self.target_actor(
                    torch.FloatTensor(state_w_goal)
                )
        return np.array(np.squeeze(action))

    def memorize(self, item):
        self.memory.append(item)

    def get_sample(self):
        sample = random.sample(self.memory, self.batch_size)
        state, action, reward, state_next, epi_done, goals = zip(*sample)
        x = torch.tensor(state, requires_grad=True).float()
        u = torch.tensor(action, requires_grad=True).float()
        r = torch.tensor(reward, requires_grad=True).float()
        xn = torch.tensor(state_next, requires_grad=True).float()
        done = torch.tensor(epi_done, requires_grad=True).float().view(-1,1)
        g = torch.tensor(goals, requires_grad=True).float()
        return x, u, r, xn, done, g

    def train(self):
        x, u, r, xn, done, g = self.get_sample()

        with torch.no_grad():
            action = self.target_actor(torch.cat([xn, g], 1))
            Qn = self.target_critic(torch.cat([xn, action], 1))
            target = r + (1-done) * self.dis_factor * Qn
        Q_w_noise_action = self.behavior_critic(torch.cat([x, u], 1))
        self.critic_optim.zero_grad()
        critic_loss = self.mse(Q_w_noise_action, target)
        critic_loss.backward()
        self.critic_optim.step()

        action_wo_noise = self.behavior_actor(torch.cat([x, g], 1))
        Q = self.behavior_critic(torch.cat([x, action_wo_noise], 1))
        self.actor_optim.zero_grad()
        actor_loss = torch.sum(-Q)
        actor_loss.backward()
        self.actor_optim.step()

        softupdate(self.target_actor, self.behavior_actor, self.softupdate_const)
        softupdate(self.target_critic, self.behavior_critic, self.softupdate_const)

def get2goal_w_RL():
    env = PointMassLoadPointMassQuad3D()
    agent = DDPG()
    noise = OrnsteinUhlenbeckNoise()
    parameter_logger = logging.Logger(
        log_dir=path_base, file_name='parameters.h5'
    )
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
            log_dir=path_train, file_name=f"data_{epi:05d}.h5"
        )
        while True:
            u = agent.get_action(np.concatenate((x, goal.reshape(-1,)))) \
                + noise.get_noise()
            quad_des_pos = env.convert_action2quad_des_pos(u)
            while True:
                xn, r, done, info = env.step(u, quad_des_pos, goal)
                train_logger.record(**info)
                if info['distance'] < quad_reach_criteria:
                    break
            r_scaled = r / reward_scale + 1
            item = (x, u, r_scaled, xn, done, goal.reshape(-1,))
            agent.memorize(item)
            x = xn
            if len(agent.memory) > batch_size * 5:
                agent.train()
            if done:
                break
        train_logger.close()

        if (epi+1) % epi_eval_interval == 0:
            if animation:
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                camera = Camera(fig)
            eval_logger = logging.Logger(
                log_dir=path_eval,
                file_name=f"data_trained_{(epi+1):05d}.h5"
            )
            x = env.reset(fixed_init=True)
            while True:
                u = agent.get_action(
                    np.concatenate((x, goal.reshape(-1,))),
                    net='target'
                )
                quad_des_pos = env.convert_action2quad_des_pos(u)
                while True:
                    xn, r, done, info = env.step(u, quad_des_pos, goal)
                    eval_logger.record(**info)
                    if animation:
                        snap_ani(
                            ax,
                            info['load_pos'],
                            info['quad_pos'],
                            info['quad_des_pos'],
                            info['center_pos']
                        )
                        camera.snap()
                    if info['distance'] < quad_reach_criteria:
                        break
                x = xn
                if done:
                    break
            if animation:
                ani = camera.animate(interval=1000*time_step, blit=True)
                path_gif = os.path.join(path_eval, f"{epi+1}_ani.mp4")
                ani.save(path_gif)
            eval_logger.close()
            plt.close('all')
    env.close()
    plt.close('all')

def get2goal_wo_RL():
    env = PointMassLoadPointMassQuad3D()
    _ = env.reset(fixed_init=True)
    u = np.vstack((0, 0, 0, 0))
    logger = logging.Logger(log_dir=path_compare, file_name='data_trained_00000.h5')
    if animation:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        camera = Camera(fig)
    quad_des_pos = [goal + ((-1)**(i)) * np.vstack((
            link_len*np.sin(alpha)*np.cos(theta_init),
            link_len*np.sin(alpha)*np.sin(theta_init),
            0)) for i in range(quad_num)]
    while True:
        while True:
            xn, r, done, info = env.step(u, quad_des_pos, goal)
            logger.record(**info)
            if animation:
                snap_ani(ax,
                        info['load_pos'],
                        info['quad_pos'],
                        info['quad_des_pos'],
                        info['center_pos'])
                camera.snap()
            if info['distance'] < quad_reach_criteria:
                break
        if done:
            break
    if animation:
        ani = camera.animate(interval=10, blit=True)
        path_ani = os.path.join(path_compare, "ani.mp4")
        ani.save(path_ani)
    logger.close()
    env.close()
    plt.close('all')

def make_figure(path, epi_num_trained, save_fig=True):
    data = logging.load(os.path.join(
        path,
        f"data_trained_{epi_num_trained:05d}.h5"
    ))

    time = data['time']
    quad_des_pos = data['quad_des_pos']
    quad_pos = data['quad_pos']
    load_pos = data['load_pos']
    goal = data['goal']
    quad_input = data['quad_input']
    action = data['action']
    center_pos = data['center_pos']
    theta = data['theta']
    phi = data['phi']
    distance = data['distance']
    reward = data['reward']

    fig1, ax1 = plt.subplots(nrows=3, ncols=1)
    line1, = ax1[0].plot(time, center_pos[:,0], 'r')
    line2, = ax1[0].plot(time, goal[:,0], 'b--')
    ax1[0].legend(handles=(line1, line2), labels=('center', 'goal'))
    ax1[1].plot(time, -center_pos[:,1], 'r', time, -goal[:,1], 'b--')
    ax1[2].plot(time, -center_pos[:,2], 'r', time, -goal[:,2], 'b--')
    if epi_num_trained == 0:
        ax1[0].set_title("Position of center w/o RL")
        ax1_title = "position_of_center_wo_RL"
    else:
        ax1[0].set_title(
            f"Position of center after training {epi_num_trained:05d} Epi."
        )
        ax1_title = f"{epi_num_trained:05d}_center_pos"
    ax1[2].set_xlabel('time [s]')
    ax1[0].set_ylabel('x [m]')
    ax1[1].set_ylabel('y [m]')
    ax1[2].set_ylabel('height [m]')
    ax1[0].grid(True)
    ax1[1].grid(True)
    ax1[2].grid(True)

    fig2, ax2 = plt.subplots(nrows=2, ncols=1)
    line3, = ax2[0].plot(time, theta*180/np.pi, 'r')
    line4, = ax2[0].plot(time, action[:,3]*180/np.pi, 'b--')
    ax2[0].legend(handles=(line3, line4), labels=('true', 'desired'))
    ax2[1].plot(time, phi*180/np.pi, 'r')
    if epi_num_trained == 0:
        ax2[0].set_title('Angles w/o RL')
        ax2_title = "angles_wo_RL"
    else:
        ax2[0].set_title(
            f"Angles after training {epi_num_trained:05d} Epi."
        )
        ax2_title = f"{epi_num_trained:05d}_angles"
    ax2[0].set_ylabel('rotation angle [deg]')
    ax2[1].set_ylabel('swing angle [deg]')
    ax2[1].set_xlabel('time [s]')
    ax2[0].grid(True)
    ax2[1].grid(True)

    fig3 = plt.figure()
    ax3 = plt.axes(projection='3d')
    ax3.plot3D(center_pos[:,0], center_pos[:,1], -center_pos[:,2], alpha=0.6)
    ax3.set_title('Trajectory of center position')
    ax3.set_xlabel('x [m]')
    ax3.set_ylabel('y [m]')
    ax3.set_zlabel('height [m]')
    if epi_num_trained == 0:
        ax2[0].set_title('Angles w/o RL')
        ax3_title = "center_trajectory_wo_RL"
    else:
        ax2[0].set_title(
            f"Angles after training {epi_num_trained:05d} Epi."
        )
        ax3_title = f"{epi_num_trained:05d}_center_trajectory"

    fig4, ax4 = plt.subplots(nrows=2, ncols=1)
    line5, = ax4[0].plot(time, quad_pos[:,0,2,:], 'r')
    line6, = ax4[0].plot(time, quad_pos[:,1,2,:], 'b')
    ax4[0].legend(handles=(line5, line6), labels=('Quad0', 'Quad1'))
    ax4[1].plot(time, quad_des_pos[:,0,2,:], 'r', time, quad_des_pos[:,1,2,:], 'b')
    if epi_num_trained == 0:
        ax4[0].set_title("Position of quadrotors' z-dir. w/o RL")
        ax4_title = "position_of_quadrotors_z_dir"
    else:
        ax4[0].set_title(
            f"Position of quadrotors' z-dir. after training {epi_num_trained:05d} Epi."
        )
        ax4_title = f"{epi_num_trained:05d}_quad_pos"
    ax4[1].set_xlabel('time [s]')
    ax4[0].set_ylabel("Quad.'s z-dir. pos. [m]")
    ax4[1].set_ylabel("Quad.'s z-dir. desired pos. [m]")
    ax4[0].grid(True)
    ax4[1].grid(True)

    if save_fig:
        fig1.savefig(
            os.path.join(path, ax1_title),
            bbox_inches='tight'
        )
        fig2.savefig(
            os.path.join(path, ax2_title),
            bbox_inches='tight'
        )
        fig3.savefig(
            os.path.join(path, ax3_title),
            bbox_inches='tight'
        )
        fig4.savefig(
            os.path.join(path, ax4_title),
            bbox_inches='tight'
        )
    plt.close('all')

    if epi_num_trained:
        G = 0
        for r in reward[::-1]:
            G = r.item() + dis_factor * G
        return -G

def snap_ani(ax, load_pos, quad_pos, quad_des_pos, center_pos):
    ax.plot3D(
        [load_pos[0][0], quad_pos[0][0][0]],
        [load_pos[1][0], quad_pos[0][1][0]],
        [-load_pos[2][0], -quad_pos[0][2][0]],
        alpha=0.6, c="k"
    )
    ax.plot3D(
        [load_pos[0][0], quad_pos[1][0][0]],
        [load_pos[1][0], quad_pos[1][1][0]],
        [-load_pos[2][0], -quad_pos[1][2][0]],
        alpha=0.6, c="k"
    )
    """
    ax.scatter(
        goal[0],
        goal[1],
        -goal[2],
        facecolors='y', edgecolors='y'
    )
    ax.scatter(
        center_pos[0],
        center_pos[1],
        -center_pos[2],
        facecolors='y', edgecolors='y'
    )
    ax.scatter(
        load_pos[0],
        load_pos[1],
        -load_pos[2],
        facecolors='none', edgecolors="r"
    )
    ax.scatter(
        quad_pos[0][0],
        quad_pos[0][1],
        -quad_pos[0][2],
        facecolors='none', edgecolors="b"
    )
    ax.scatter(
        quad_pos[1][0],
        quad_pos[1][1],
        -quad_pos[1][2],
        facecolors='none', edgecolors="b"
    )
    ax.scatter(
        quad_des_pos[0][0],
        quad_des_pos[0][1],
        -quad_des_pos[0][2],
        facecolors='none', edgecolors="k"
    )
    ax.scatter(
        quad_des_pos[1][0],
        quad_des_pos[1][1],
        -quad_des_pos[1][2],
        facecolors='none', edgecolors="k"
    )
    """

if __name__ == "__main__":
    path_base = os.path.join(
        'log', datetime.today().strftime('%Y%m%d-%H%M%S')
    )
    path_eval = os.path.join(path_base, 'eval')
    path_train = os.path.join(path_base, 'train')
    path_compare = os.path.join(
        'log', datetime.today().strftime('%Y%m%d-%H%M%S'), 'compare'
    )

    print('Simulating without RL')
    get2goal_wo_RL()
    make_figure(path_compare, 0)

    print('Simulating with RL')
    get2goal_w_RL()
    cost_his = np.array(
        [[(i+1) * epi_eval_interval,
          make_figure(path_eval, (i+1) * epi_eval_interval)]
         for i in range(epi_num//epi_eval_interval)]
    )

    fig4, ax4 = plt.subplots(nrows=1, ncols=1)
    ax4.plot(cost_his[:,0], cost_his[:,1], "*")
    ax4.grid(True)
    ax4.set_title(f"Cost according to number of trained episode")
    ax4.set_xlabel('Number of trained episode')
    ax4.set_ylabel('Cost')
    fig4.savefig(
        os.path.join(path_eval, f"Cost_{epi_num:d}"),
        bbox_inches='tight'
    )
    plt.close('all')
    end_logger = logging.Logger(
        log_dir=path_eval,
        file_name="cost_his.h5")
    end_logger.record(cost_his=cost_his)
    end_logger.close()

