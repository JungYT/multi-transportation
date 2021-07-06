import numpy as np
import random
from types import SimpleNamespace as SN
from pathlib import Path

from tqdm import tqdm
from datetime import datetime
from collections import deque
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import fym.logging as logging
from fym.utils import rot
from dynamics import MultiQuadSlungLoad
from utils import draw_plot, compare_episode, hardupdate, softupdate, \
    OrnsteinUhlenbeckNoise

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def load_config():
    cfg = SN()
    cfg.epi_train = 5000
    cfg.epi_eval = 200
    cfg.dt = 0.1
    cfg.max_t = 3.
    cfg.solver = 'odeint'
    cfg.ode_step_len = 10
    cfg.dir = Path('log', datetime.today().strftime('%Y%m%d-%H%M%S'))
    cfg.g = np.vstack((0., 0., -9.81))

    cfg.animation = SN()
    cfg.animation.quad_size = 0.315
    cfg.animation.rotor_size = 0.15
    cfg.animation.view_angle = [None, None]

    cfg.quad = SN()
    cfg.quad.num = 3
    cfg.quad.mass = 0.755
    cfg.quad.J = np.diag([0.0820, 0.0845, 0.1377])
    cfg.quad.iscollision = 0.5
    cfg.quad.psi_des = cfg.quad.num*[0]
    cfg.quad.omega_init = np.vstack((0., 0., 0.))

    cfg.load = SN()
    cfg.load.pos_bound = [[-5, 5], [-5, 5], [1, 10]]
    cfg.load.att_bound = [
        [-np.pi/4, np.pi/4],
        [-np.pi/4, np.pi/4],
        [-np.pi, np.pi]
    ]
    cfg.load.pos_init = np.vstack((0., 0., 3.))
    cfg.load.dcm_init = rot.angle2dcm(0., 0., 0.).T
    cfg.load.vel_init = np.vstack((0., 0., 0.))
    cfg.load.mass = 1.5
    cfg.load.J = np.diag([0.2, 0.2, 0.2])
    cfg.load.cg = np.vstack((0., 0., -0.7))
    cfg.load.size = 1.
    cfg.load.pos_des = np.vstack((0., 0., 5.))
    cfg.load.dcm_des = np.eye(3)

    cfg.link = SN()
    cfg.link.len = cfg.quad.num * [0.5]
    cfg.link.anchor = [
        np.vstack((
            cfg.load.size * np.cos(i*2*np.pi/cfg.quad.num),
            cfg.load.size * np.sin(i*2*np.pi/cfg.quad.num),
            0
        )) - cfg.load.cg for i in range(cfg.quad.num)
    ]
    cfg.link.uvec_bound = [[-np.pi, np.pi], [0, np.pi/4]]

    cfg.controller = SN()
    cfg.controller.Ke = 20.
    cfg.controller.Ks = 80.
    cfg.controller.chattering_bound = 0.5
    cfg.controller.unc_max = 0.1
    cfg.controller.Kpos = 1.
    cfg.controller.Kvel = 0.5
    cfg.controller.Kdcm = 0.1
    cfg.controller.Komega = 0.05

    cfg.ddpg = SN()
    cfg.ddpg.P = np.diag([1., 1., 1.])
    cfg.ddpg.reward_max = 120
    cfg.ddpg.state_dim = 8
    cfg.ddpg.action_dim = 3
    cfg.ddpg.action_scaling = torch.tensor(
        [3., np.pi, np.pi/12],
        requires_grad=True
    )
    cfg.ddpg.action_bias = torch.tensor(
        [15., 0., np.pi/12],
        requires_grad=True
    )
    cfg.ddpg.memory_size = 20000
    cfg.ddpg.actor_lr = 0.0001
    cfg.ddpg.critic_lr = 0.001
    cfg.ddpg.discount = 0.999
    cfg.ddpg.softupdate = 0.001
    cfg.ddpg.batch_size = 64

    cfg.noise = SN()
    cfg.noise.rho = np.array([0.15, 0.01, 0.01])
    cfg.noise.mu = np.array([0., 0., 0.])
    cfg.noise.sigma = np.array([0.2, 0.01, 0.2])

    return cfg


class ActorNet(nn.Module):
    def __init__(self, cfg):
        super(ActorNet, self).__init__()
        self.lin1 = nn.Linear(cfg.ddpg.state_dim, 32)
        self.lin2 = nn.Linear(32, 64)
        self.lin3 = nn.Linear(64, 32)
        self.lin4 = nn.Linear(32, 16)
        self.lin5 = nn.Linear(16, cfg.ddpg.action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(16)
        self.cfg = cfg

    def forward(self, state):
        x1 = self.relu(self.bn1(self.lin1(state)))
        x2 = self.relu(self.bn2(self.lin2(x1)))
        x3 = self.relu(self.bn3(self.lin3(x2)))
        x4 = self.relu(self.bn4(self.lin4(x3)))
        x5 = self.tanh(self.lin5(x4))
        x_scaled = x5 * self.cfg.ddpg.action_scaling \
            + self.cfg.ddpg.action_bias
        return x_scaled

class CriticNet(nn.Module):
    def __init__(self, cfg):
        super(CriticNet, self).__init__()
        self.lin1 = nn.Linear(cfg.ddpg.state_dim+cfg.ddpg.action_dim, 32)
        self.lin2 = nn.Linear(32, 64)
        self.lin3 = nn.Linear(64, 128)
        self.lin4 = nn.Linear(128, 64)
        self.lin5 = nn.Linear(64, 32)
        self.lin6 = nn.Linear(32, 16)
        self.lin7 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(32)
        self.bn6 = nn.BatchNorm1d(16)

    def forward(self, state_w_action):
        x1 = self.relu(self.bn1(self.lin1(state_w_action)))
        x2 = self.relu(self.bn2(self.lin2(x1)))
        x3 = self.relu(self.bn3(self.lin3(x2)))
        x4 = self.relu(self.bn4(self.lin4(x3)))
        x5 = self.relu(self.bn5(self.lin5(x4)))
        x6 = self.relu(self.bn6(self.lin6(x5)))
        x7 = self.lin7(x6)
        return x7


class DDPG:
    def __init__(self, cfg):
        self.memory = deque(maxlen=cfg.ddpg.memory_size)
        self.behavior_actor = ActorNet(cfg).float()
        self.behavior_critic = CriticNet(cfg).float()
        self.target_actor = ActorNet(cfg).float()
        self.target_critic = CriticNet(cfg).float()
        self.actor_optim = optim.Adam(
            self.behavior_actor.parameters(), lr=cfg.ddpg.actor_lr
        )
        self.critic_optim = optim.Adam(
            self.behavior_critic.parameters(), lr=cfg.ddpg.critic_lr
        )
        self.mse = nn.MSELoss()
        hardupdate(self.target_actor, self.behavior_actor)
        hardupdate(self.target_critic, self.behavior_critic)
        self.cfg = cfg

    def get_action(self, state, net="behavior"):
        with torch.no_grad():
            action = self.behavior_actor(torch.FloatTensor(state)) \
                if net == "behavior" \
                else self.target_actor(torch.FloatTensor(state))
        return np.array(np.squeeze(action))

    def memorize(self, item):
        self.memory.append(item)

    def get_sample(self):
        sample = random.sample(self.memory, self.cfg.ddpg.batch_size)
        state, action, reward, state_next, epi_done = zip(*sample)
        # x = torch.tensor(state, requires_grad=True).float()
        # u = torch.tensor(action, requires_grad=True).float()
        # r = torch.tensor(reward, requires_grad=True).float()
        # xn = torch.tensor(state_next, requires_grad=True).float()
        # done = torch.tensor(epi_done, requires_grad=True).float().view(-1,1)
        x = torch.FloatTensor(state)
        u = torch.FloatTensor(action)
        r = torch.FloatTensor(reward)
        xn = torch.FloatTensor(state_next)
        done = torch.FloatTensor(epi_done).view(-1,1)
        return x, u, r, xn, done

    def train(self):
        x, u, r, xn, done = self.get_sample()
        with torch.no_grad():
            action = self.target_actor(xn)
            Qn = self.target_critic(torch.cat([xn, action], 1))
            target = r + (1-done) * self.cfg.ddpg.discount * Qn
        Q_w_noise_action = self.behavior_critic(torch.cat([x,u], 1))
        self.critic_optim.zero_grad()
        critic_loss = self.mse(Q_w_noise_action, target)
        critic_loss.backward()
        self.critic_optim.step()

        # Q_w_noise_action_tmp = self.behavior_critic(torch.cat([x,u], 1))
        # critic_loss_tmp = self.mse(Q_w_noise_action_tmp, target)
        # if critic_loss_tmp - critic_loss == 0.:
        #     print("Critic loss does not change:", critic_loss_tmp-critic_loss)

        action_wo_noise = self.behavior_actor(x)
        Q = self.behavior_critic(torch.cat([x, action_wo_noise],1))
        self.actor_optim.zero_grad()
        actor_loss = torch.sum(-Q)
        actor_loss.backward()
        self.actor_optim.step()

        # action_wo_noise_tmp = self.behavior_actor(x)
        # Q_tmp = self.behavior_critic(torch.cat([x, action_wo_noise_tmp],1))
        # actor_loss_tmp = torch.sum(-Q_tmp)
        # if actor_loss_tmp - actor_loss == 0.:
        #     print("Actor loss does not change:", actor_loss_tmp-actor_loss)

        softupdate(
            self.target_actor,
            self.behavior_actor,
            self.cfg.ddpg.softupdate)
        softupdate(
            self.target_critic,
            self.behavior_critic,
            self.cfg.ddpg.softupdate)

    def save_parameters(self, path_save):
        torch.save({
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'behavior_actor': self.behavior_actor.state_dict(),
            'behavior_critic': self.behavior_critic.state_dict()
        }, path_save)

    def set_train_mode(self):
        self.behavior_actor.train()
        self.behavior_critic.train()
        self.target_actor.train()
        self.target_critic.train()

    def set_eval_mode(self):
        self.behavior_actor.eval()
        self.behavior_critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()


def train(agent, des, cfg, noise, env):
    agent.set_train_mode()
    x = env.reset()
    # x = env.reset(fixed_init=True)
    noise.reset()
    while True:
        # noise_set = np.array([
        #     noise.get_noise(),
        #     noise.get_noise(),
        #     noise.get_noise()
        # ])
        noise_set = np.array([
            np.random.normal(size=3),
            np.random.normal(scale=0.1, size=3),
            np.random.normal(scale=0.03, size=3)
        ])
        action = np.clip(
            agent.get_action(x) + noise_set,
            -cfg.ddpg.action_scaling.detach().numpy() \
            + cfg.ddpg.action_bias.detach().numpy(),
            cfg.ddpg.action_scaling.detach().numpy() \
            + cfg.ddpg.action_bias.detach().numpy()
        )
        xn, r, done, _ = env.step(action, des)
        for i in range(cfg.quad.num):
            item = (x[i], action[i], r[i], xn[i], done)
            agent.memorize(item)
        # agent.memorize((x, action, r, xn, done))
        x = xn
        if len(agent.memory) > 5 * cfg.ddpg.batch_size:
            agent.train()
        if done:
            break
    plt.close('all')


def evaluate(env, agent, des, cfg, dir_env_data, dir_agent_data):
    agent.set_eval_mode()
    env.logger = logging.Logger(dir_env_data)
    env.logger.set_info(cfg=cfg)
    logger_agent = logging.Logger(dir_agent_data)
    x = env.reset(fixed_init=True)
    while True:
        action = agent.get_action(x)
        xn, _, done, info = env.step(action, des)
        logger_agent.record(**info)
        x = xn
        if done:
            break
    logger_agent.close()
    env.logger.close()
    plt.close('all')


def main():
    cfg = load_config()
    env = MultiQuadSlungLoad(cfg)
    agent = DDPG(cfg)
    noise = OrnsteinUhlenbeckNoise(
        cfg.noise.rho,
        cfg.noise.mu,
        cfg.ddpg.action_scaling.detach().numpy(),
        cfg.dt,
        cfg.ddpg.action_dim
    )
    des = [cfg.load.pos_des, cfg.load.dcm_des, cfg.quad.psi_des]

    for epi_num in tqdm(range(cfg.epi_train)):
        train(agent, des, cfg, noise, env)

        if (epi_num+1) % cfg.epi_eval == 0:
            dir_save = Path(cfg.dir, f"epi_after_{epi_num+1:05d}")
            dir_env_data = Path(dir_save, "env_data.h5")
            dir_agent_data = Path(dir_save, "agent_data.h5")
            dir_agent_params = Path(dir_save, "agent_params.pt")

            evaluate(env, agent, des, cfg, dir_env_data, dir_agent_data)
            draw_plot(dir_env_data, dir_agent_data, dir_save)
            agent.save_parameters(dir_agent_params)
    env.close()


if __name__ == "__main__":
    main()

    past = -1
    compare_episode(past, ani=True)

    plt.close('all')



