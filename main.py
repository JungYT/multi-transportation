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


class OrnsteinUhlenbeckNoise:
    def __init__(self, x0=None):
        self.rho = 0.15
        self.mu = 0
        self.sigma = 0.2
        self.dt = 0.1
        self.x0 = x0
        self.size = 9
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

def softupdate(target, behavior, softupdate_const):
    for targetParam, behaviorParam in zip(target.parameters(), behavior.parameters()):
        targetParam.data.copy_(
            targetParam.data*(1.0-softupdate_const)\
            + behaviorParam.data*softupdate_const
        )

def hardupdate(target, behavior):
    for targetParam, behaviorParam in zip(target.parameters(), behavior.parameters()):
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

class Load(BaseEnv):
    def __init__(self, mass, pos_init, rot_mat_init):
        super().__init__()
        self.pos = BaseSystem(pos_init)
        self.vel = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.rot_mat = BaseSystem(rot_mat_init)
        self.ang_rate = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.mass = mass
        self.J = np.diag([0.2, 0.2, 0.2])

    def set_dot(self, load_acc, ang_acc):
        self.pos.dot = self.vel.state
        self.vel.dot = load_acc
        self.ang_rate.dot = ang_acc
        R = self.rot_mat.state
        self.rot_mat.dot = R.dot(hat(self.ang_rate.state))

class Link(BaseEnv):
    def __init__(self, link_init, length, rho):
        super().__init__()
        self.link = BaseSystem(link_init)
        self.ang_rate = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.len = length
        self.rho = rho

    def set_dot(self, ang_acc):
        w = self.ang_rate.state
        q = self.link.state
        self.link.dot = (hat(w)).dot(q)
        self.ang_rate.dot = ang_acc

class Quadrotor(BaseEnv):
    def __init__(self, rot_mat_init):
        super().__init__()
        self.rot_mat = BaseSystem(rot_mat_init)
        self.ang_rate = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.J = np.diag([0.0820, 0.0845, 0.1377])
        self.mass = 4.34

    def set_dot(self, M):
        R = self.rot_mat.state
        Omega = self.ang_rate.state
        self.rot_mat.dot = R.dot(hat(Omega))
        self.ang_rate.dot = np.linalg.inv(self.J).dot(
            M - (hat(Omega)).dot(self.J.dot(Omega))
        )

class IntergratedDynamics(BaseEnv):
    def __init__(self, env_params):
        super().__init__(dt=env_params['time_step'], max_t=env_params['max_t'], solver="odeint")
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
        self.collision_criteria = env_params['collision_criteria']
        self.g = 9.81
        self.e3 = np.vstack((0.0, 0.0, 1.0))
        self.K_e = env_params['K_e']
        self.K_s = env_params['K_s']
        self.chatter_bound = env_params['chatter_bound']
        self.unc_max = env_params['unc_max']
        self.M = deque(maxlen=self.quad_num)
        self.max_t = env_params['max_t']
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
            self.load.pos.state[2] = np.random.uniform(
                low=-5.,
                high=-10.
            ),
            self.load.ang_rate.dot = np.vstack((
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
            self.load.rot_mat.state = rot.angle2dcm(
                np.random.uniform(
                    low=-np.pi/6,
                    high=np.pi/6
                ),
                np.random.uniform(
                    low=-np.pi/6,
                    high=np.pi/6
                ),
                np.random.uniform(
                    low=-np.pi/6,
                    high=np.pi/6
                )
            ).T
            tmp = [np.random.rand(3, 1) for i in range(self.quad_num)]
            for i, link in enumerate(self.links.systems):
                link.link.state = tmp[i]/np.linalg.norm(tmp[i])

            for quad in self.quads.systems:
                quad.rot_mat.state = rot.angle2dcm(
                    np.random.uniform(
                        low=-np.pi/6,
                        high=np.pi/6
                    ),
                    np.random.uniform(
                        low=-np.pi/6,
                        high=np.pi/6
                    ),
                    0
                ).T
        obs = self.observe()
        return obs

    def set_dot(self, t, R_des, f_des):
        m_T = self.load.mass
        R0 = self.load.rot_mat.state
        Omega = self.load.ang_rate.state
        Omega_hat = hat(Omega)
        Omega_hat2 = Omega_hat.dot(Omega_hat)

        for i, (link, quad) in enumerate(
                zip(self.links.systems, self.quads.systems)
        ):
            l = link.len
            rho = link.rho
            q = link.link.state
            w = link.ang_rate.state
            m = quad.mass
            R = quad.rot_mat.state
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
            R = quad.rot_mat.state
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
        *_, done = self.update(R_des = des_attitude_set, f_des=des_force_set)
        quad_pos, quad_vel, quad_ang, quad_ang_rate, \
            quad_rot_mat, anchor_pos, collisions = self.compute_quad_state()
        load_ang = np.vstack(rot.dcm2angle(self.load.rot_mat.state.T))[::-1]
        done, time = self.terminate(collisions)
        distance = [np.linalg.norm(quad_pos[i]-anchor_pos[i])
                    for i in range(self.quad_num)]
        obs = self.observe()
        reward = self.get_reward(collisions, load_ang)

        info = {
            "time": time,
            "load_pos": self.load.pos.state,
            "load_vel": self.load.vel.state,
            "load_ang": load_ang,
            "load_ang_rate": self.load.ang_rate.state,
            "load_rot_mat": self.load.rot_mat.state,
            "quad_pos": quad_pos,
            "quad_vel": quad_vel,
            "quad_ang": quad_ang,
            "quad_ang_rate": quad_ang_rate,
            "quad_rot_mat": quad_rot_mat,
            "moment": self.M,
            "distance": distance,
            "anchor_pos": anchor_pos,
            "collisions": collisions,
            "reward": reward,
            "des_attitude": des_attitude_set,
            "des_force": des_force_set
        }
        return obs, reward, done, info

    def reshape_action(self, action):
        des_attitude_set = [np.vstack(np.append(action[2*i:2*(i+1)], 0.))
                            for i in range(self.quad_num)]
        des_force_set = action[self.quad_num*2:]
        return des_attitude_set, des_force_set

    def observe(self):
        load_pos = self.load.pos.state.reshape(-1,)
        load_vel = self.load.vel.state.reshape(-1,)
        load_attitude = np.array(
            rot.dcm2angle(self.load.rot_mat.state.T)
        )[::-1]
        load_ang_rate = self.load.ang_rate.state.reshape(-1,)
        load_state = [load_pos[2], load_vel[2], load_attitude, load_ang_rate]
        link_state = [
            rot.velocity2polar(link.link.state)[1:]
            for link in self.links.systems
        ]
        quad_state = [
            np.array(rot.dcm2angle(quad.rot_mat.state.T))[::-1][0:2]
            for quad in self.quads.systems
        ]
        obs = np.hstack(load_state + link_state + quad_state)
        return obs

    def compute_quad_state(self):
        load_pos = self.load.pos.state
        load_rot_mat = self.load.rot_mat.state
        load_vel = self.load.vel.state
        load_ang_rate = self.load.ang_rate.state
        load_ang_rate_hat = hat(load_ang_rate)

        quad_pos = [
            load_pos + load_rot_mat.dot(link.rho) - link.len*link.link.state
            for link in self.links.systems
        ]
        quad_vel = [
            load_vel + load_rot_mat.dot(load_ang_rate_hat.dot(link.rho)) \
            - link.len*hat(link.ang_rate.state).dot(link.link.state)
            for link in self.links.systems
        ]
        quad_ang = [
            np.array(rot.dcm2angle(quad.rot_mat.state.T))[::-1]
            for quad in self.quads.systems
        ]
        quad_ang_rate = [quad.ang_rate.state for quad in self.quads.systems]
        quad_rot_mat = [
            quad.rot_mat.state for quad in self.quads.systems
        ]
        anchor_pos = [load_pos + load_rot_mat.dot(link.rho) for link in self.links.systems]
        collisions = [np.linalg.norm(quad_pos[i]-quad_pos[i+1])
                      for i in range(self.quad_num-1)]
        collisions.append(np.linalg.norm(quad_pos[-1] - quad_pos[0]))

        return quad_pos, quad_vel, quad_ang, quad_ang_rate,\
            quad_rot_mat, anchor_pos, collisions

    def control_attitude(self, des_attitude_set):
        M_set = []
        for i, quad in enumerate(self.quads.systems):
            quad_ang = np.vstack(
                rot.dcm2angle(quad.rot_mat.state.T)[::-1]
            )
            phi = quad_ang[0][0]
            theta = quad_ang[1][0]
            ang_rate = quad.ang_rate.state
            ang_rate_hat = hat(ang_rate)
            wx = quad.ang_rate.state[0].item()
            wy = quad.ang_rate.state[1].item()
            wz = quad.ang_rate.state[2].item()

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
            if np.linalg.matrix_rank(L) < 3:
                print(np.linalg.matrix_rank(L))

            e2 = L.dot(quad.ang_rate.state)
            e1 = quad_ang - des_attitude_set[i]
            s = self.K_e*e1 + e2
            s_clip = np.clip(s/self.chatter_bound, -1, 1)
            M = (quad.J.dot(np.linalg.inv(L))).dot(
                -self.K_e*e2 - b - L2.dot(e2) - s_clip*(self.unc_max+self.K_s)
            ) + ang_rate_hat.dot(quad.J.dot(ang_rate))

            M_set.append(M)
            self.M.append(M)
        return M_set

    def terminate(self, collisions):
        time = self.clock.get()
        load_posz = self.load.pos.state[2]
        done = 1. if (load_posz > 0 or time > self.max_t or
                      any(x < self.collision_criteria for x in collisions)) else 0.
        return done, time

    def get_reward(self, collisions, load_ang):
        load_posz = self.load.pos.state[2]
        load_velz = self.load.vel.state[2]
        if (load_posz > 0 or
                any(x < self.collision_criteria for x in collisions)):
            r = np.array([-50])
        else:
            r = -(np.linalg.norm(load_ang) +
                  np.linalg.norm(self.load.ang_rate.state) +
                  abs(load_velz)
                  )
        r_scaled = (r + 25) / 25
        return r_scaled

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.lin1 = nn.Linear(20, 128)
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
        self.lin1 = nn.Linear(20+9, 128)
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


def main(path_base, env_params):
    env = IntergratedDynamics(env_params)
    agent = DDPG()
    noise = OrnsteinUhlenbeckNoise()
    cost_his = []
    tmp = 1
    for epi in tqdm(range(env_params["epi_num"])):
        x = env.reset()
        noise.reset()
        if (epi+1) % env_params["epi_show"] == 0 or epi == 0:
            train_logger = logging.Logger(
                log_dir=os.path.join(path_base, 'train'),
                file_name=f"data_{epi+1:05d}.h5"
            )
            while True:
                u = agent.get_action(x) + noise.get_noise()
                for i in range(tmp):
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

            x = env.reset(random_init=False)
            eval_logger = logging.Logger(
                log_dir=os.path.join(path_base, 'eval', f"epi_{epi+1:05d}"),
                file_name=f"data_{(epi+1):05d}.h5"
            )
            if env_params['animation']:
                fig = plt.figure()# {{{
                ax = fig.gca(projection='3d')
                camera =Camera(fig)# }}}
                while True:
                    u = agent.get_action(x)
                    for i in range(tmp):
                        xn, r, done, info = env.step(u)
                        snap_ani(ax, info, env_params)
                        camera.snap()
                        eval_logger.record(**info)
                    x = xn
                    if done:
                        break
                ani = camera.animate(
                    interval=1000*env_params['time_step'], blit=True
                )
                path_ani = os.path.join(path_base, f"ani_{(epi+1):05d}.mp4")
                ani.save(path_ani)
            else:
                while True:
                    u = agent.get_action(x)
                    for i in range(tmp):
                        xn, r, done, info = env.step(u)
                        eval_logger.record(**info)
                    x = xn
                    if done:
                        break
            eval_logger.close()
            plt.close('all')
            cost = make_figure(
                os.path.join(path_base, 'eval', f'epi_{epi+1:05d}'),
                (epi+1),
                env_params
            )
            cost_his.append([epi+1, cost])
            torch.save({
                'target_actor': agent.target_actor.state_dict(),
                'target_critic': agent.target_critic.state_dict()
            }, os.path.join(path_base, 'eval', f"parameters_{epi+1:05d}.pt"))
        else:
            while True:
                u = agent.get_action(x) + noise.get_noise()
                for i in range(tmp):
                    xn, r, done, info = env.step(u)
                item = (x, u, r, xn, done)
                agent.memorize(item)
                x = xn
                if len(agent.memory) > 64*5:
                    agent.train()
                if done:
                    break
    cost_his = np.array(cost_his)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(cost_his[:,0], cost_his[:,1], "*")
    ax.grid(True)
    ax.set_title(f"Cost according to number of trained episode")
    ax.set_xlabel("Number of trained episode")
    ax.set_ylabel("Cost")
    fig.savefig(
        os.path.join(path_base, f"Cost_{env_params['epi_num']:d}"),
        bbox_inches='tight'
    )
    plt.close('all')
    env.close()
    logger = logging.Logger(
        log_dir=path_base, file_name='params_and_cost.h5'
    )
    logger.set_info(**env_params)
    logger.record(cost_his=cost_his)
    logger.close()

def make_figure(path, epi_num, env_params):
    data = logging.load(os.path.join(path, f"data_{epi_num:05d}.h5"))# {{{

    quad_num = env_params['quad_num']
    time = data['time']
    load_pos = data['load_pos']
    load_vel = data['load_vel']
    load_ang = data['load_ang']*180/np.pi
    load_ang_rate = data['load_ang_rate']*180/np.pi
    quad_pos = data['quad_pos']
    quad_vel = data['quad_vel']
    quad_ang = data['quad_ang']*180/np.pi
    quad_ang_rate = data['quad_ang_rate']*180/np.pi
    moment = data['moment']
    distance = data['distance']
    collisions = data['collisions']
    reward = data['reward']
    des_attitude = data['des_attitude'].squeeze()*180/np.pi
    des_force = data['des_force']

    pos_ylabel = ["X [m]", "Y [m]", "Height [m]"]
    make_figure_3col(
        time,
        load_pos*np.vstack((1, 1, -1)),
        "Position of payload",
        "time [s]",
        pos_ylabel,
        f"load_pos_{epi_num:05d}",
        path
    )
    vel_ylabel = ["$V_x$ [m/s]", "$V_y$ [m/a]", "$V_z$ [m/s]"]
    make_figure_3col(
        time,
        load_vel,
        "Velocity of payload",
        "time [s]",
        vel_ylabel,
        f"load_vel_{epi_num:05d}",
        path
    )
    ang_ylabel = ["$\phi$ [deg]", "$\\theta$ [deg]", "$\psi$ [deg]"]
    make_figure_3col(
        time,
        load_ang,
        "Euler angle of payload",
        "time [s]",
        ang_ylabel,
        f"load_ang_{epi_num:05d}",
        path,
        unwrap=True
    )
    ang_rate_ylabel = [
        "$\dot{\phi}$ [deg/s]",
        "$\dot{\\theta}$ [deg/s]",
        "$\dot{\psi}$ [deg/s]"
    ]
    make_figure_3col(
        time,
        load_ang_rate,
        "Euler angle rate of payload",
        "time [s]",
        ang_rate_ylabel,
        f"load_ang_rate_{epi_num:05d}",
        path
    )

    [make_figure_3col(
        time,
        quad_pos[:,i,:,:]*np.vstack((1, 1, -1)),
        f"Position of quadrotor {i}",
        "time [s]",
        pos_ylabel,
        f"quad_pos_{i}_{epi_num:05d}",
        path
    ) for i in range(quad_num)]

    [make_figure_3col(
        time,
        quad_vel[:,i,:,:],
        f"Velocity of quadrotor {i}",
        "time [s]",
        vel_ylabel,
        f"quad_vel_{i}_{epi_num:05d}",
        path
    ) for i in range(quad_num)]

    [make_figure_compare(
        time,
        quad_ang[:,i,:],
        des_attitude[:,i,:],
        ['Quad.', 'Des.'],
        f"Euler angle of quadrotor {i}",
        "time [s]",
        ang_ylabel,
        f"quad_ang_{i}_{epi_num:05d}",
        path,
        unwrap=True
    ) for i in range(quad_num)]

    [make_figure_3col(
        time,
        quad_ang_rate[:,i,:],
        f"Euler angle rate of quadrotor {i}",
        "time [s]",
        ang_rate_ylabel,
        f"quad_ang_rate_{i}_{epi_num:05d}",
        path
    ) for i in range(quad_num)]

    moment_ylabel = ["$M_x$ [Nm]", "$M_y$ [Nm]", "$M_z$ [Nm]"]
    [make_figure_3col(
        time,
        moment[:,i,:],
        f"Moment of quadrotor {i}",
        "time [s]",
        moment_ylabel,
        f"quad_moment_{i}_{epi_num:05d}",
        path
    ) for i in range(quad_num)]

    distance_ylabel = ["quad0 [m]", "quad1 [m]", "quad2 [m]"]
    link_len = env_params['link_len']
    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(time, distance[:,0])
    ax[1].plot(time, distance[:,1])
    ax[2].plot(time, distance[:,2])
    ax[0].set_title("distance from quad to anchor")
    ax[0].set_ylabel(distance_ylabel[0])
    ax[1].set_ylabel(distance_ylabel[1])
    ax[2].set_ylabel(distance_ylabel[2])
    ax[2].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(3)]
    [ax[i].set_ylim(link_len[i]-0.5, link_len[i]+0.5) for i in range(3)]
    fig.align_ylabels(ax)
    fig.savefig(
        os.path.join(path, f"distance_link_to_anchor_{epi_num:05d}"),
        bbox_inches='tight'
    )

    fig, ax = plt.subplots(nrows=1, ncols=1)
    line1, = ax.plot(time, collisions[:,0], 'r')
    line2, = ax.plot(time, collisions[:,1], 'b')
    line3, = ax.plot(time, collisions[:,2], 'k')
    ax.legend(handles=(line1, line2, line3),
              labels=('quad0-quad1', 'quad1-quad2', 'quad2-quad0'))
    ax.set_title("Distance between quadrotors")
    ax.set_ylabel('distance [m]')
    ax.set_xlabel("time [s]")
    ax.grid(True)
    fig.savefig(
        os.path.join(path, f"distance_btw_quads_{epi_num:05d}"),
        bbox_inches='tight'
    )

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(time, reward, 'r')
    ax.set_title("Reward")
    ax.set_ylabel('reward')
    ax.set_xlabel("time [s]")
    ax.grid(True)
    fig.savefig(
        os.path.join(path, f"reward_{epi_num:05d}"),
        bbox_inches='tight'
    )

    force_ylabel = ["quad0 [N]", "quad1 [N]", "quad2 [N]"]
    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(time, des_force[:,0])
    ax[1].plot(time, des_force[:,1])
    ax[2].plot(time, des_force[:,2])
    ax[0].set_title("Desired net force")
    ax[0].set_ylabel(force_ylabel[0])
    ax[1].set_ylabel(force_ylabel[1])
    ax[2].set_ylabel(force_ylabel[2])
    ax[2].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(3)]
    fig.align_ylabels(ax)
    fig.savefig(
        os.path.join(path, f"des_net_force_{epi_num:05d}"),
        bbox_inches='tight'
    )

    plt.close('all')

    G = 0
    for r in reward[::-1]:
        G = r.item() + 0.999*G
    return G

def make_figure_3col(x, y, title, xlabel, ylabel,
                     file_name, path, unwrap=False):
    fig, ax = plt.subplots(nrows=3, ncols=1)# {{{
    if unwrap:
        ax[0].plot(x, np.unwrap(y[:,0], axis=0))
        ax[1].plot(x, np.unwrap(y[:,1], axis=0))
        ax[2].plot(x, np.unwrap(y[:,2], axis=0))
    else:
        ax[0].plot(x, y[:,0])
        ax[1].plot(x, y[:,1])
        ax[2].plot(x, y[:,2])
    ax[0].set_title(title)
    ax[0].set_ylabel(ylabel[0])
    ax[1].set_ylabel(ylabel[1])
    ax[2].set_ylabel(ylabel[2])
    ax[2].set_xlabel(xlabel)
    [ax[i].grid(True) for i in range(3)]
    fig.align_ylabels(ax)
    fig.savefig(
        os.path.join(path, file_name),
        bbox_inches='tight'
    )
    plt.close('all')# }}}

def make_figure_compare(x, y1, y2, legend, title, xlabel, ylabel,
                     file_name, path, unwrap=False):
    fig, ax = plt.subplots(nrows=3, ncols=1)# {{{
    if unwrap:
        line1, = ax[0].plot(x, np.unwrap(y1[:,0], axis=0), 'r')
        line2, = ax[0].plot(x, np.unwrap(y2[:,0], axis=0), 'b--')
        ax[0].legend(
            handles=(line1, line2),
            labels=(legend[0], legend[1])
        )
        ax[1].plot(x, np.unwrap(y1[:,1], axis=0), 'r',
                   x, np.unwrap(y2[:,1], axis=0), 'b--')
        ax[2].plot(x, np.unwrap(y1[:,2], axis=0), 'r',
                   x, np.unwrap(y2[:,2], axis=0), 'b--')
    else:
        line1, = ax[0].plot(x, y1[:,0], 'r')
        line2, = ax[0].plot(x, y2[:,0], 'b--')
        ax[0].legend(
            handles=(line1, line2),
            labels=(legend[0], legend[1])
        )
        ax[1].plot(x, y1[:,1], 'r', x, y2[:,1], 'b--')
        ax[2].plot(x, y1[:,2], 'r', x, y2[:,2], 'b--')
    ax[0].set_title(title)
    ax[0].set_ylabel(ylabel[0])
    ax[1].set_ylabel(ylabel[1])
    ax[2].set_ylabel(ylabel[2])
    ax[2].set_xlabel(xlabel)
    [ax[i].grid(True) for i in range(3)]
    fig.align_ylabels(ax)
    fig.savefig(
        os.path.join(path, file_name),
        bbox_inches='tight'
    )
    plt.close('all')# }}}

def snap_ani(ax, info, params):
    load_pos = info['load_pos']# {{{
    quad_pos = info['quad_pos']
    quad_num = params['quad_num']
    anchor_pos = info["anchor_pos"]
    # link
    [
        ax.plot3D(
            [anchor_pos[i][0][0], quad_pos[i][0][0]],
            [anchor_pos[i][1][0], quad_pos[i][1][0]],
            [-anchor_pos[i][2][0], -quad_pos[i][2][0]],
            alpha=0.6, c="k"
        ) for i in range(quad_num)
    ]
    # rho
    [
        ax.plot3D(
            [anchor_pos[i][0][0], load_pos[0][0]],
            [anchor_pos[i][1][0], load_pos[1][0]],
            [-anchor_pos[i][2][0], -load_pos[2][0]],
            alpha=0.6, c="k"
        ) for i in range(quad_num)
    ]
    # load shape
    [
        ax.plot3D(
            [anchor_pos[i%quad_num][0][0], anchor_pos[(i+1)%quad_num][0][0]],
            [anchor_pos[i%quad_num][1][0], anchor_pos[(i+1)%quad_num][1][0]],
            [-anchor_pos[i%quad_num][2][0], -anchor_pos[(i+1)%quad_num][2][0]],
            alpha=0.6, c="k"
        ) for i in range(quad_num)
    ]
    # quadrotors
    [
        ax.plot3D(
            quad_pos[i][0][0],
            quad_pos[i][1][0],
            -quad_pos[i][2][0],
            alpha=0.6, c="r", marker="X"
        ) for i in range(quad_num)
    ]

    # axis limit
    ax.set_xlim3d(-30, 30)
    ax.set_ylim3d(-30, 30)
    ax.set_zlim3d(-5, 55)# }}}

if __name__ == "__main__":
    path_base = os.path.join(
        'log', datetime.today().strftime('%Y%m%d-%H%M%S')
    )
    quad_num = 3
    """
    rot.angle2dcm converts angle to transformation matrix
    which transforms from ref. to body coordinate.
    In simulation, rotation matrix follows robotic convention,
    which means transformation matrix from body to ref.
    """
    quad_rot_mat_init = rot.angle2dcm(-np.pi/3, np.pi/4, np.pi/6).T # z-y-x order
    # quad_rot_mat_init = rot.angle2dcm(0., np.pi/4, np.pi/6).T # z-y-x order
    # quad_rot_mat_init = rot.angle2dcm(0, 0, np.pi/6).T # z-y-x order
    anchor_radius = 1.
    cg_bias = np.vstack((0.0, 0.0, 1.))
    env_params = {
        'epi_num': 1,
        'epi_show': 1,
        'time_step': 0.1,
        'max_t': 10.,
        'load_mass': 10.,
        'load_pos_init': np.vstack((0.0, 0.0, -5.0)),
        'load_rot_mat_init': np.eye(3),
        'quad_num': quad_num,
        'quad_rot_mat_init': quad_num*[quad_rot_mat_init],
        'link_len': quad_num*[3.],
        'link_init': quad_num*[np.vstack((0.0, 0.0, 1.0))],
        'link_rho': [
            3*np.vstack((
                anchor_radius * np.cos(i*2*np.pi/quad_num),
                anchor_radius * np.sin(i*2*np.pi/quad_num),
                0
            )) - cg_bias for i in range(quad_num)
        ],
        'collision_criteria': 0.5,
        'K_e': 20.,
        'K_s': 80.,
        'chatter_bound': 0.5,
        'unc_max': 0.1,
        'anchor_radius': anchor_radius,
        'cg_bias': cg_bias,
        'animation': False,
    }

    main(path_base, env_params)

