"""
rot.angle2dcm converts angle to transformation matrix
which transforms from ref. to body coordinate.
In simulation, rotation matrix follows robotic convention,
which means transformation matrix from body to ref.
"""
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

from fym.core import BaseEnv, BaseSystem
import fym.core as core
import fym.logging as logging
from fym.utils import rot
from celluloid import Camera
from utils import OrnsteinUhlenbeckNoise, softupdate, \
    hardupdate, hat, make_figure, snap_ani

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

cfg = SN()

def load_config():
    cfg.animation = True
    cfg.epi_train = 1
    cfg.epi_eval = 1
    cfg.dt = 0.1
    cfg.ode_step = 10
    cfg.max_t = 1.
    cfg.dir = Path('log', datetime.today().strftime('%Y%m%d-%H%M%S'))
    cfg.collision = 0.5

    cfg.controller = SN()
    cfg.controller.K_e = 20.
    cfg.controller.K_s = 80.
    cfg.controller.chattering = 0.5
    cfg.controller.unc_max = 0.1

    cfg.load = SN()
    cfg.load.mass = 10.
    cfg.load.pos = np.vstack((0., 0., -5.))
    cfg.load.dcm = rot.angle2dcm(-np.pi/6, np.pi/6, np.pi/6).T
    cfg.load.cg = np.vstack((0., 0., 1.))

    cfg.quad = SN()
    cfg.quad.num = 3
    cfg.quad.dcm = cfg.quad.num * [
        rot.angle2dcm(-np.pi/6, np.pi/6, np.pi/6).T
    ]

    cfg.link = SN()
    cfg.link.len = cfg.quad.num * [3.]
    cfg.link.dir = cfg.quad.num * [
        np.vstack((0., 0., 1.))
    ]
    cfg.link.rho = [
        cfg.quad.num * np.vstack((
            1. * np.cos(i*2*np.pi/cfg.quad.num),
            1. * np.sin(i*2*np.pi/cfg.quad.num),
            0
        )) - cfg.load.cg for i in range(cfg.quad.num)
    ]


class Load(BaseEnv):
    def __init__(self, mass, pos_init, dcm_init):
        super().__init__()
        self.pos = BaseSystem(pos_init)
        self.vel = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.dcm = BaseSystem(dcm_init)
        self.omega = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.mass = mass
        self.J = np.diag([0.2, 0.2, 0.2])

    def set_dot(self, load_acc, ang_acc):
        self.pos.dot = self.vel.state
        self.vel.dot = load_acc
        self.omega.dot = ang_acc
        R = self.dcm.state
        self.dcm.dot = R.dot(hat(self.omega.state))

class Link(BaseEnv):
    def __init__(self, link_init, length, rho):
        super().__init__()
        self.link = BaseSystem(link_init)
        self.omega = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.len = length
        self.rho = rho

    def set_dot(self, ang_acc):
        w = self.omega.state
        q = self.link.state
        self.link.dot = (hat(w)).dot(q)
        self.omega.dot = ang_acc

class Quadrotor(BaseEnv):
    def __init__(self, dcm_init):
        super().__init__()
        self.dcm = BaseSystem(dcm_init)
        self.omega = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.J = np.diag([0.0820, 0.0845, 0.1377])
        self.mass = 4.34

    def set_dot(self, M):
        R = self.dcm.state
        Omega = self.omega.state
        self.dcm.dot = R.dot(hat(Omega))
        self.omega.dot = np.linalg.inv(self.J).dot(
            M - (hat(Omega)).dot(self.J.dot(Omega))
        )

class IntergratedDynamics(BaseEnv):
    def __init__(self):
        super().__init__(dt=cfg.dt, max_t=cfg.max_t, solver="odeint", ode_step_len=cfg.ode_step)
        self.quad_num = cfg.quad.num
        self.load = Load(
            cfg.load.mass,
            cfg.load.pos,
            cfg.load.dcm
        )
        self.links = core.Sequential(
            **{f"link{i:02d}": Link(
            cfg.link.dir[i],
            cfg.link.len[i],
            cfg.link.rho[i]) for i in range(self.quad_num)}
        )
        self.quads = core.Sequential(
            **{f"quad{i:02d}": Quadrotor(
            cfg.quad.dcm[i]
            ) for i in range(self.quad_num)}
        )
        self.collision_criteria = cfg.collision
        self.g = 9.81
        self.e3 = np.vstack((0.0, 0.0, 1.0))
        self.K_e = cfg.controller.K_e
        self.K_s = cfg.controller.K_s
        self.chatter_bound = cfg.controller.chattering
        self.unc_max = cfg.controller.unc_max
        self.M = deque(maxlen=self.quad_num)
        self.max_t = cfg.max_t
        self.S1_set = deque(maxlen=self.quad_num)
        self.S2_set = deque(maxlen=self.quad_num)
        self.S3_set = deque(maxlen=self.quad_num)
        self.S4_set = deque(maxlen=self.quad_num)
        self.S5_set = deque(maxlen=self.quad_num)
        self.S6_set = deque(maxlen=self.quad_num)
        self.S7_set = deque(maxlen=self.quad_num)

    def reset(self, random_init=True):
        super().reset()
        if random_init:
            self.load.pos.state = np.vstack((
                np.random.uniform(
                    low=-5.,
                    high=5.
                ),
                np.random.uniform(
                    low=-5.,
                    high=5.
                ),
                np.random.uniform(
                    low=-5.,
                    high=-10.
                )
            ))
            self.load.vel.state = np.vstack((
                np.random.uniform(
                    low=-0.5,
                    high=0.5
                ),
                np.random.uniform(
                    low=-0.5,
                    high=0.5
                ),
                np.random.uniform(
                    low=-0.5,
                    high=0.5
                )
            ))
            self.load.dcm.state = rot.angle2dcm(
                np.random.uniform(
                    low=-np.pi/4,
                    high=np.pi/4
                ),
                np.random.uniform(
                    low=-np.pi/4,
                    high=np.pi/4
                ),
                np.random.uniform(
                    low=-np.pi/4,
                    high=np.pi/4
                )
            ).T
            self.load.omega.state = np.vstack((
                np.random.uniform(
                    low=-0.5,
                    high=0.5
                ),
                np.random.uniform(
                    low=-0.5,
                    high=0.5
                ),
                np.random.uniform(
                    low=-0.5,
                    high=0.5
                )
            ))
            tmp = [np.random.rand(3, 1) for i in range(self.quad_num)]
            for i, link in enumerate(self.links.systems):
                link.link.state = tmp[i]/np.linalg.norm(tmp[i])

            for link in self.links.systems:
                link.omega.state = np.vstack((
                    np.random.uniform(
                        low=-0.5,
                        high=0.5
                    ),
                    np.random.uniform(
                        low=-0.5,
                        high=0.5
                    ),
                    np.random.uniform(
                        low=-0.5,
                        high=0.5
                    )
                ))
            for quad in self.quads.systems:
                quad.dcm.state = rot.angle2dcm(
                    np.random.uniform(
                        low=-np.pi/4,
                        high=np.pi/4
                    ),
                    np.random.uniform(
                        low=-np.pi/4,
                        high=np.pi/4
                    ),
                    0
                ).T
        obs = self.observe()
        return obs

    def set_dot(self, t, R_des, f_des):
        m_T = self.load.mass
        R0 = self.load.dcm.state
        Omega = self.load.omega.state
        Omega_hat = hat(Omega)
        Omega_hat2 = Omega_hat.dot(Omega_hat)

        for i, (link, quad) in enumerate(
                zip(self.links.systems, self.quads.systems)
        ):
            l = link.len
            rho = link.rho
            q = link.link.state
            w = link.omega.state
            m = quad.mass
            R = quad.dcm.state
            u = -f_des[i] * R.dot(self.e3)

            m_T += m
            q_hat2 = (hat(q)).dot(hat(q))
            q_qT = np.eye(3) + q_hat2
            rho_hat = hat(rho)
            rhohat_R0T = rho_hat.dot(R0.T)
            w_norm = np.linalg.norm(w)
            l_wnorm2_q = l * w_norm * w_norm * q
            R0_Omega2_rho = R0.dot(Omega_hat2.dot(rho))

            S1_temp = q_qT.dot(u - m*R0_Omega2_rho) - m*l_wnorm2_q
            S2_temp = m * q_qT.dot(rhohat_R0T.T)
            S3_temp = m * rhohat_R0T.dot(q_qT.dot(rhohat_R0T.T))
            S4_temp = m * q_hat2
            S5_temp = m * rhohat_R0T.dot(q_qT)
            S6_temp = rhohat_R0T.dot(
                q_qT.dot(u + m*self.g*self.e3) \
                - m*q_hat2.dot(R0_Omega2_rho) \
                - m*l_wnorm2_q
            )
            S7_temp = m*rho_hat.dot(rho_hat)
            self.S1_set.append(S1_temp)
            self.S2_set.append(S2_temp)
            self.S3_set.append(S3_temp)
            self.S4_set.append(S4_temp)
            self.S5_set.append(S5_temp)
            self.S6_set.append(S6_temp)
            self.S7_set.append(S7_temp)
        S1 = sum(self.S1_set)
        S2 = sum(self.S2_set)
        S3 = sum(self.S3_set)
        S4 = sum(self.S4_set)
        S5 = sum(self.S5_set)
        S6 = sum(self.S6_set)
        S7 = sum(self.S7_set)

        J_bar = self.load.J - S7
        J_hat = self.load.J + S3
        J_hat_inv = np.linalg.inv(J_hat)
        Mq = m_T*np.eye(3) + S4
        A = -J_hat_inv.dot(S5)
        B = J_hat_inv.dot(S6 - Omega_hat.dot(J_bar.dot(Omega)))
        C = Mq + S2.dot(A)

        load_acc = np.linalg.inv(C).dot(Mq.dot(self.g*self.e3) + S1 - S2.dot(B))
        load_ang_acc = A.dot(load_acc) + B
        self.load.set_dot(load_acc, load_ang_acc)

        M_set = self.control_attitude(R_des)
        for i, (link, quad) in enumerate(
                zip(self.links.systems, self.quads.systems)
        ):
            l = link.len
            rho = link.rho
            q = link.link.state
            q_hat = hat(q)
            m = quad.mass
            R = quad.dcm.state
            u = -f_des[i] * R.dot(self.e3)
            R0_Omega2_rho = R0.dot(Omega_hat2.dot(rho))
            D = R0.dot(hat(rho).dot(load_ang_acc)) + self.g*self.e3 + u/m

            link_ang_acc = q_hat.dot(load_acc + R0_Omega2_rho - D) / l
            link.set_dot(link_ang_acc)
            quad.set_dot(M_set[i])

    def step(self, action):
        des_force_set = 3*[90]
        des_attitude_set = 3*[np.vstack((0.0, 0.0, 0.0))]

        # des_attitude_set, des_force_set = self.reshape_action(action)
        *_, done = self.update(attitude_des = des_attitude_set, f_des=des_force_set)
        quad_pos, quad_vel, quad_ang, quad_omega, \
            quad_dcm, anchor_pos, collisions = self.compute_quad_state()
        load_ang = np.vstack(rot.dcm2angle(self.load.dcm.state.T))[::-1]
        done, time = self.terminate(collisions, done)
        distance = [np.linalg.norm(quad_pos[i]-anchor_pos[i])
                    for i in range(self.quad_num)]
        obs = self.observe()
        reward = self.get_reward(collisions, load_ang)
        e_set = [quad_ang[i] - des_attitude_set[i] for i in range(self.quad_num)]

        # if self.clock.get() > 1:
        #     breakpoint()
        info = {
            "time": time,
            "load_pos": self.load.pos.state,
            "load_vel": self.load.vel.state,
            "load_ang": load_ang,
            "load_omega": self.load.omega.state,
            "load_dcm": self.load.dcm.state,
            "quad_pos": quad_pos,
            "quad_vel": quad_vel,
            "quad_ang": quad_ang,
            "quad_omega": quad_omega,
            "quad_dcm": quad_dcm,
            "moment": self.M,
            "distance": distance,
            "anchor_pos": anchor_pos,
            "collisions": collisions,
            "reward": reward,
            "des_attitude": des_attitude_set,
            "des_force": des_force_set,
            "error": e_set,
        }
        return obs, reward, done, info

    def logger_callback(self, t, y, i, t_hist, ode_hist, attitude_des, f_des):
        M_set = []
        states = self.observe_dict(y)
        for i in range(3):
            M = self.control(
                attitude_des[i], states['quads'][f'quad{i:02d}']['dcm'], states['quads'][f'quad{i:02d}']['omega']
            )
            M_set.append(M)
        # breakpoint()

        return dict(time=t, moment=M_set, **states)

    def control(self, des_attitude, quad_dcm, quad_omega):
        quad_ang = np.vstack(
            rot.dcm2angle(quad_dcm.T)[::-1]
        )
        phi = quad_ang[0][0]
        theta = quad_ang[1][0]
        omega = quad_omega
        omega_hat = hat(omega)
        wx = quad_omega[0].item()
        wy = quad_omega[1].item()
        wz = quad_omega[2].item()

        L = np.array([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ])
        L2 = np.array([
            [wy*np.cos(phi)*np.tan(theta) - wz*np.sin(phi)*np.tan(theta),
            wy*np.sin(phi)/(np.cos(theta))**2 + wz*np.cos(phi)/(np.cos(theta))**2,
            0],
            [-wy*np.sin(phi)-wz*np.cos(phi), 0, 0],
            [wy*np.cos(phi)/np.cos(theta) - wz*np.sin(phi)*np.cos(theta),
            wy*np.sin(phi)*np.tan(theta)/np.cos(theta) - \
            wz*np.cos(phi)*np.tan(theta)/np.cos(theta),
            0]
        ])
        b = np.vstack((wx, 0., 0.))
        J = np.diag([0.0820, 0.0845, 0.1377])

        e2 = L.dot(omega)
        e1 = quad_ang - des_attitude
        s = 20.*e1 + e2
        s_clip = np.clip(s/0.5, -1, 1)
        M = (J.dot(np.linalg.inv(L))).dot(
            -20.*e2 - b - L2.dot(e2) - s_clip*(0.1+80.)
        ) + omega_hat.dot(J.dot(omega))

        return M

    def reshape_action(self, action):
        des_attitude_set = [np.vstack(np.append(action[2*i:2*(i+1)], 0.))
                            for i in range(self.quad_num)]
        des_force_set = action[self.quad_num*2:]
        return des_attitude_set, des_force_set

    def observe(self):
        load_pos = self.load.pos.state.reshape(-1,)
        load_vel = self.load.vel.state.reshape(-1,)
        load_attitude = np.array(
            rot.dcm2angle(self.load.dcm.state.T)
        )[::-1]
        load_omega = self.load.omega.state.reshape(-1,)
        load_state = [load_pos[2], load_vel[2], load_attitude, load_omega]
        link_state = [
            rot.velocity2polar(link.link.state)[1:]
            for link in self.links.systems
        ] + [link.omega.state.reshape(-1,) for link in self.links.systems]
        quad_state = [
            np.array(rot.dcm2angle(quad.dcm.state.T))[::-1][0:2]
            for quad in self.quads.systems
        ]
        obs = np.hstack(load_state + link_state + quad_state)
        return obs

    def compute_quad_state(self):
        load_pos = self.load.pos.state
        load_dcm = self.load.dcm.state
        load_vel = self.load.vel.state
        load_omega = self.load.omega.state
        load_omega_hat = hat(load_omega)

        quad_pos = [
            load_pos + load_dcm.dot(link.rho) - link.len*link.link.state
            for link in self.links.systems
        ]
        quad_vel = [
            load_vel + load_dcm.dot(load_omega_hat.dot(link.rho)) \
            - link.len*hat(link.omega.state).dot(link.link.state)
            for link in self.links.systems
        ]
        quad_ang = [
            np.array(rot.dcm2angle(quad.dcm.state.T))[::-1]
            for quad in self.quads.systems
        ]
        quad_omega = [quad.omega.state for quad in self.quads.systems]
        quad_dcm = [
            quad.dcm.state for quad in self.quads.systems
        ]
        anchor_pos = [load_pos + load_dcm.dot(link.rho) for link in self.links.systems]
        collisions = [np.linalg.norm(quad_pos[i]-quad_pos[i+1])
                      for i in range(self.quad_num-1)]
        collisions.append(np.linalg.norm(quad_pos[-1] - quad_pos[0]))

        return quad_pos, quad_vel, quad_ang, quad_omega,\
            quad_dcm, anchor_pos, collisions

    def control_attitude(self, des_attitude_set):
        M_set = []
        for i, quad in enumerate(self.quads.systems):
            quad_ang = np.vstack(
                rot.dcm2angle(quad.dcm.state.T)[::-1]
            )
            phi = quad_ang[0][0]
            theta = quad_ang[1][0]
            omega = quad.omega.state
            omega_hat = hat(omega)
            wx = quad.omega.state[0].item()
            wy = quad.omega.state[1].item()
            wz = quad.omega.state[2].item()

            L = np.array([
                [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                [0, np.cos(phi), -np.sin(phi)],
                [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
            ])
            L2 = np.array([
                [wy*np.cos(phi)*np.tan(theta) - wz*np.sin(phi)*np.tan(theta),
                 wy*np.sin(phi)/(np.cos(theta))**2 + wz*np.cos(phi)/(np.cos(theta))**2,
                 0],
                [-wy*np.sin(phi)-wz*np.cos(phi), 0, 0],
                [wy*np.cos(phi)/np.cos(theta) - wz*np.sin(phi)*np.cos(theta),
                 wy*np.sin(phi)*np.tan(theta)/np.cos(theta) - \
                 wz*np.cos(phi)*np.tan(theta)/np.cos(theta),
                 0]
            ])
            b = np.vstack((wx, 0., 0.))

            e2 = L.dot(quad.omega.state)
            e1 = quad_ang - des_attitude_set[i]
            s = self.K_e*e1 + e2
            s_clip = np.clip(s/self.chatter_bound, -1, 1)
            M = (quad.J.dot(np.linalg.inv(L))).dot(
                -self.K_e*e2 - b - L2.dot(e2) - s_clip*(self.unc_max+self.K_s)
            ) + omega_hat.dot(quad.J.dot(omega))

            M_set.append(M)
            self.M.append(M)
        return M_set

    def terminate(self, collisions, done):
        time = self.clock.get()
        load_posz = self.load.pos.state[2]
        done = 1. if (load_posz > 0 or done or
                      any(x < self.collision_criteria for x in collisions)) else 0.
        return done, time

    def get_reward(self, collisions, load_ang):
        load_posz = self.load.pos.state[2]
        if (load_posz > 0 or
                any(x < self.collision_criteria for x in collisions)):
            r = np.array([-50])
        else:
            r = -(np.linalg.norm(load_ang)**2 +
                  (load_posz + 5)**2)
        r_scaled = (r + 25) / 25
        return r_scaled

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.lin1 = nn.Linear(29, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 32)
        self.lin4 = nn.Linear(32, 16)
        self.lin5 = nn.Linear(16, 9)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        x1 = self.relu(self.lin1(state))
        x2 = self.relu(self.lin2(x1))
        x3 = self.relu(self.lin3(x2))
        x4 = self.relu(self.lin4(x3))
        x5 = self.tanh(self.lin5(x4))
        xScaled = x5 * torch.Tensor(6*[np.pi/3] + 3*[20.]) + \
            torch.Tensor(6*[0.] + 3*[80.0])
        return xScaled

class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.lin1 = nn.Linear(29+9, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 32)
        self.lin4 = nn.Linear(32, 16)
        self.lin5 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, state_w_action):
        x1 = self.relu(self.lin1(state_w_action))
        x2 = self.relu(self.lin2(x1))
        x3 = self.relu(self.lin3(x2))
        x4 = self.relu(self.lin4(x3))
        x5 = self.lin5(x4)
        return x5

class DDPG:
    def __init__(self):
        self.memory = deque(maxlen=20000)
        self.behavior_actor = ActorNet().float()
        self.behavior_critic = CriticNet().float()
        self.target_actor = ActorNet().float()
        self.target_critic = CriticNet().float()
        self.actor_optim = optim.Adam(
            self.behavior_actor.parameters(), lr=0.0001
        )
        self.critic_optim = optim.Adam(
            self.behavior_critic.parameters(), lr=0.001
        )
        self.mse = nn.MSELoss()
        hardupdate(self.target_actor, self.behavior_actor)
        hardupdate(self.target_critic, self.behavior_critic)
        self.dis_factor = 0.999
        self.softupdate_const = 0.001
        self.batch_size = 64

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
        state, action, reward, state_next, epi_done = zip(*sample)
        x = torch.tensor(state, requires_grad=True).float()
        u = torch.tensor(action, requires_grad=True).float()
        r = torch.tensor(reward, requires_grad=True).float()
        xn = torch.tensor(state_next, requires_grad=True).float()
        done = torch.tensor(epi_done, requires_grad=True).float().view(-1,1)
        return x, u, r, xn, done

    def train(self):
        x, u, r, xn, done = self.get_sample()

        with torch.no_grad():
            action = self.target_actor(xn)
            Qn = self.target_critic(torch.cat([xn, action], 1))
            target = r + (1-done) * self.dis_factor * Qn
        Q_w_noise_action = self.behavior_critic(torch.cat([x, u], 1))
        self.critic_optim.zero_grad()
        critic_loss = self.mse(Q_w_noise_action, target)
        critic_loss.backward()
        self.critic_optim.step()

        action_wo_noise = self.behavior_actor(x)
        Q = self.behavior_critic(torch.cat([x, action_wo_noise], 1))
        self.actor_optim.zero_grad()
        actor_loss = torch.sum(-Q)
        actor_loss.backward()
        self.actor_optim.step()

        softupdate(self.target_actor, self.behavior_actor, self.softupdate_const)
        softupdate(self.target_critic, self.behavior_critic, self.softupdate_const)


def main():
    env = IntergratedDynamics()
    agent = DDPG()
    noise = OrnsteinUhlenbeckNoise()
    cost_his = []
    for epi in tqdm(range(cfg.epi_train)):
        x = env.reset()
        noise.reset()
        if (epi+1) % cfg.epi_eval == 0 or epi == 0:
            train_logger = logging.Logger(
                log_dir=Path(cfg.dir, "train"),
                file_name=f"data_{epi+1:05d}.h5"
            )
            while True:
                u = agent.get_action(x) + noise.get_noise()
                xn, r, done, info = env.step(u)
                item = (x, u, r, xn, done)
                agent.memorize(item)
                train_logger.record(**info)
                x = xn
                if len(agent.memory) > 64*5:
                    agent.train()
                if done:
                    break
            train_logger.close()

            env.logger = logging.Logger('test.h5')
            x = env.reset(random_init=False)
            eval_logger = logging.Logger(
                log_dir=Path(cfg.dir, f"eval/epi_{epi+1:05d}"),
                file_name=f"data_{(epi+1):05d}.h5"
            )
            if cfg.animation:
                fig = plt.figure()# {{{
                ax = fig.gca(projection='3d')
                camera =Camera(fig)# }}}
                while True:
                    u = agent.get_action(x)
                    xn, r, done, info = env.step(u)
                    snap_ani(ax, info, cfg)
                    camera.snap()
                    eval_logger.record(**info)
                    x = xn
                    if done:
                        break
                ani = camera.animate(
                    interval=1000*cfg.dt, blit=True
                )
                path_ani = Path(cfg.dir, f"ani_{(epi+1):05d}.mp4")
                ani.save(path_ani)
            else:
                while True:
                    u = agent.get_action(x)
                    xn, r, done, info = env.step(u)
                    eval_logger.record(**info)
                    x = xn
                    if done:
                        break
            eval_logger.close()
            plt.close('all')
            env.logger.close()
            cost = make_figure(
                Path(cfg.dir, f'eval/epi_{epi+1:05d}'),
                (epi+1),
                cfg
            )
            cost_his.append([epi+1, cost])
            torch.save({
                'target_actor': agent.target_actor.state_dict(),
                'target_critic': agent.target_critic.state_dict()
            }, Path(cfg.dir, f"eval/parameters_{epi+1:05d}.pt"))
        else:
            while True:
                u = agent.get_action(x) + noise.get_noise()
                xn, r, done, info = env.step(u)
                item = (x, u, r, xn, done)
                agent.memorize(item)
                x = xn
                if len(agent.memory) > 64*5:
                    agent.train()
                if done:
                    break
    env.close()
    cost_his = np.array(cost_his)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(cost_his[:,0], cost_his[:,1], "*")
    ax.grid(True)
    ax.set_title(f"Cost according to number of trained episode")
    ax.set_xlabel("Number of trained episode")
    ax.set_ylabel("Cost")
    fig.savefig(
        Path(cfg.dir, f"Cost_{cfg.epi_train:d}"),
        bbox_inches='tight'
    )
    plt.close('all')
    env.close()
    logger = logging.Logger(
        log_dir=cfg.dir, file_name='config_ang_cost.h5'
    )
    logger.set_info(cfg=cfg)
    logger.record(cost_his=cost_his)
    logger.close()


if __name__ == "__main__":
    load_config()
    main()

