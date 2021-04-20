import numpy as np
import random
import os

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

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


def hat(v):
    v1, v2, v3 = v.squeeze()
    return np.array([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0]
    ])

def wrap(angle):
    return (angle+np.pi) % (2*np.pi) - np.pi

class Load(BaseEnv):
    def __init__(self, mass, pos_init, rotation_mat_init):
        super().__init__()
        self.pos = BaseSystem(pos_init)
        self.vel = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.ang_rate = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.rotation_mat = BaseSystem(rotation_mat_init)
        self.mass = mass
        self.J = np.diag([0.2, 0.2, 0.2])

    def set_dot(self, load_acc, ang_acc):
        self.pos.dot = self.vel.state
        self.vel.dot = load_acc
        self.ang_rate.dot = ang_acc
        R = self.rotation_mat.state
        self.rotation_mat.dot = R.dot(hat(self.ang_rate.state))

class Link(BaseEnv):
    def __init__(self, link_init, length, rho):
        super().__init__()
        self.link = BaseSystem(link_init)
        self.ang_rate = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.len = length
        self.rho = rho

    def set_dot(self, ang_acc):
        self.link.dot = (hat(self.ang_rate.state)).dot(self.link.state)
        self.ang_rate.dot = ang_acc

class Quadrotor(BaseEnv):
    def __init__(self, rotation_mat_init):
        super().__init__()
        self.rotation_mat = BaseSystem(rotation_mat_init)
        self.ang_rate = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.J = np.diag([0.0820, 0.0845, 0.1377])
        self.mass = 4.34

    def set_dot(self, M):
        R = self.rotation_mat.state
        self.rotation_mat.dot = R.dot(hat(self.ang_rate.state))
        Omega = self.ang_rate.state
        self.ang_rate.dot = np.linalg.inv(self.J).dot(
            M - (hat(Omega)).dot(self.J.dot(Omega))
        )

class IntergratedDynamics(BaseEnv):
    def __init__(self, env_params):
        super().__init__(dt=env_params['time_step'], max_t=env_params['max_t'])
        self.quad_num = env_params['quad_num']
        self.load = Load(
            env_params['load_mass'],
            env_params['load_pos_init'],
            env_params['load_rot_mat_init']
        )
        self.links = core.Sequential(
            **{f"link{i:02d}": Link(
            env_params['link_init'][i],
            env_params['link_len'][i],
            env_params['link_rho'][i]) for i in range(self.quad_num)}
        )
        self.quads = core.Sequential(
            **{f"quad{i:02d}": Quadrotor(
            env_params['quad_rot_mat_init'][i]
            ) for i in range(self.quad_num)}
        )
        self.g = 9.81
        self.e3 = np.vstack((0.0, 0.0, 1.0))
        self.S1_set = deque(maxlen=self.quad_num)
        self.S2_set = deque(maxlen=self.quad_num)
        self.S3_set = deque(maxlen=self.quad_num)
        self.S4_set = deque(maxlen=self.quad_num)
        self.S5_set = deque(maxlen=self.quad_num)
        self.S6_set = deque(maxlen=self.quad_num)

    def reset(self):
        super().reset()

    def set_dot(self, t, f, M):
        m_T = self.load.mass
        R0 = self.load.rotation_mat.state
        Omega = self.load.ang_rate.state
        Omega_hat2 = (hat(Omega)).dot(hat(Omega))

        for i, (link, quad) in enumerate(
                zip(self.links.systems, self.quads.systems)
        ):
            l = link.len
            rho = link.rho
            q = link.link.state
            w = link.ang_rate.state
            m = quad.mass
            R = quad.rotation_mat.state
            u = -f[i] * R.dot(self.e3)

            m_T += m
            rho_hat = hat(rho)
            q_qT = q.dot(q.T)
            m_l_w_q = m * l * np.linalg.norm(w)*np.linalg.norm(w) * q
            R0_Omega2_rho = R0.dot(Omega_hat2.dot(rho))
            R0_rhohat = R0.dot(rho_hat)
            rhohat_R0T = rho_hat.dot(R0.T)
            q_hat2 = (hat(q)).dot(hat(q))

            S1_temp = q_qT.dot(u) - m_l_w_q - m*q_qT.dot(R0_Omega2_rho)
            S2_temp = m * q_qT.dot(R0_rhohat)
            S3_temp = m * rhohat_R0T.dot(q_qT.dot(R0_rhohat))
            S4_temp = m * q_hat2
            S5_temp = m * rhohat_R0T.dot(q_qT)
            S6_temp = rhohat_R0T.dot(
                q_qT.dot(u + m*self.g*self.e3) \
                - m*q_hat2.dot(R0_Omega2_rho) \
                - m_l_w_q
            )
            self.S1_set.append(S1_temp)
            self.S2_set.append(S2_temp)
            self.S3_set.append(S3_temp)
            self.S4_set.append(S4_temp)
            self.S5_set.append(S5_temp)
            self.S6_set.append(S6_temp)
        S1 = sum(self.S1_set)
        S2 = sum(self.S2_set)
        S3 = sum(self.S3_set)
        S4 = sum(self.S4_set)
        S5 = sum(self.S5_set)
        S6 = sum(self.S6_set)

        Mq = m_T*np.eye(3) + S4
        J_hat = self.load.J - S3
        J_hat_inv = np.linalg.inv(J_hat)
        A = -J_hat_inv.dot(S5)
        B = J_hat_inv.dot(S6)
        C = Mq - S2.dot(A)

        load_acc = np.linalg.inv(C).dot(Mq.dot(self.g*self.e3) + S1 + S2.dot(B))
        load_ang_acc = A.dot(load_acc) + B
        self.load.set_dot(load_acc, load_ang_acc)

        for i, (link, quad) in enumerate(
                zip(self.links.systems, self.quads.systems)
        ):
            l = link.len
            rho = link.rho
            q = link.link.state
            m = quad.mass
            R = quad.rotation_mat.state
            u = -f[i] * R.dot(self.e3)
            D = R0.dot(Omega_hat2.dot(rho)) - self.g*self.e3 - u/m
            dOmega = A.dot(load_acc) + B

            link_ang_acc = (hat(q)).dot(
                load_acc + D - R0.dot(hat(rho).dot(dOmega))) / l
            link.set_dot(link_ang_acc)
            quad.set_dot(M[i])

    def step(self):
        f_set = 3*[5]
        M_set = 3*[np.vstack((0.0, 0.0, 0.0))]
        *_, done = self.update(f=f_set, M=M_set)
        return done


env_params = {
    'time_step': 0.01,
    'max_t': 5,
    'load_mass': 10,
    'load_pos_init': np.vstack((0.0, 0.0, 0.0)),
    'load_rot_mat_init': np.eye(3),
    'quad_num': 3,
    'quad_rot_mat_init': 3*[np.eye(3)],
    'link_len': 3*[1],
    'link_init': 3*[np.vstack((0.0, 0.0, 1.0))],
    'link_rho': [np.vstack((1.0, 0.0, 0.0)),
                 np.vstack((-0.5, np.sqrt(3)/2, 0.0)),
                 np.vstack((-0.5, -np.sqrt(3)/2, 0.0))
                 ],
}

def main():
    env = IntergratedDynamics(env_params)
    env.reset()
    while True:
        done = env.step()
        if done:
            break
    env.close()

if __name__ == "__main__":
    main()





