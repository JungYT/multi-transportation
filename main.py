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
        self.link.dot = (hat(self.ang_rate.state)).dot(self.link.state)
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
        self.rot_mat.dot = R.dot(hat(self.ang_rate.state))
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
        self.K = env_params['gain']
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

    def reset(self):
        super().reset()

    def set_dot(self, t, f_des):
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
            q_hat2 = hat(q).dot(hat(q))
            q_qT = np.eye(3) + q_hat2
            rho_hat = hat(rho)
            rhohat_R0T = rho_hat.dot(R0.T)
            w_norm = np.linalg.norm(w)
            l_wnorm_q = l * w_norm * w_norm * q
            R0_Omega2_rho = R0.dot(Omega_hat2.dot(rho))

            S1_temp = q_qT.dot(u - m*R0_Omega2_rho) - m*l_wnorm_q
            S2_temp = m * q_qT.dot(rhohat_R0T.T)
            S3_temp = m * rhohat_R0T.dot(q_qT.dot(rhohat_R0T.T))
            S4_temp = m * q_hat2
            S5_temp = m * rhohat_R0T.dot(q_qT)
            S6_temp = rhohat_R0T.dot(
                q_qT.dot(u + m*self.g*self.e3) \
                - m*q_hat2.dot(R0_Omega2_rho) \
                - m*l_wnorm_q
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
        if np.linalg.det(J_hat) < 0.01:
            print('J_hat seems to be singular')
        J_hat_inv = np.linalg.inv(J_hat)
        Mq = m_T*np.eye(3) + S4

        A = -J_hat_inv.dot(S5)
        B = J_hat_inv.dot(S6 - Omega_hat.dot(J_bar.dot(Omega)))
        C = Mq + S2.dot(A)
        if np.linalg.det(C) < 0.01:
            print('C seems to be singular')

        load_acc = np.linalg.inv(C).dot(Mq.dot(self.g*self.e3) + S1 - S2.dot(B))
        load_ang_acc = A.dot(load_acc) + B
        self.load.set_dot(load_acc, load_ang_acc)

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
            quad.set_dot(self.M[i])

    def step(self):
        f_des_set = 3*[90]
        des_attitude_set = 3*[np.vstack((0.0, 0.0, 0.0))]
        self.controller(des_attitude_set)
        """
        for i in range(3):
            self.M.append(np.vstack((1.0, 0.0, 0.0)))
        """
        *_, done = self.update(f_des=f_des_set)
        quad_pos, quad_vel, quad_ang, quad_ang_rate, quad_rot_mat \
            = self.compute_quad_state()
        load_ang = np.vstack(rot.dcm2angle(self.load.rot_mat.state.T))[::-1]
        done, time = self.terminate()
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
        }
        return done, info

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
        return quad_pos, quad_vel, quad_ang, quad_ang_rate, quad_rot_mat

    def controller(self, des_attitude_set):
        for i, quad in enumerate(self.quads.systems):
            quad_ang = np.vstack(
                rot.dcm2angle(quad.rot_mat.state.T)[::-1]
            )
            L = np.array([
                [1, np.sin(quad_ang[0][0])*np.tan(quad_ang[1][0]),
                 np.cos(quad_ang[0][0])*np.tan(quad_ang[1][0])],
                [0, np.cos(quad_ang[0][0]), -np.sin(quad_ang[0][0])],
                [0, np.sin(quad_ang[0][0])/np.cos(quad_ang[1][0]),
                 np.cos(quad_ang[0][0])/np.cos(quad_ang[1][0])]
            ])
            quad_euler_rate = L.dot(quad.ang_rate.state)
            s = self.K * (quad_ang - des_attitude_set[i]) + quad_euler_rate
            s_clip = np.clip(s/self.chatter_bound, -1, 1)

            mu = -self.K * quad_euler_rate - (self.unc_max + 1) * s_clip
            self.M.append(quad.J.dot(mu))

    def terminate(self):
        time = self.clock.get()
        load_posz = self.load.pos.state[2]
        done = 1. if load_posz > 0 or time > self.max_t else 0.
        return done, time


def main(path_base, env_params):
    env = IntergratedDynamics(env_params)
    env.reset()
    logger = logging.Logger(
        log_dir=path_base, file_name='test.h5'
    )
    logger.record(**env_params)
    if env_params['animation']:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        camera =Camera(fig)
    while True:
        done, info = env.step()
        logger.record(**info)
        if env_params['animation']:
            snap_ani(ax, info, env_params)
            camera.snap()
        if done:
            break
    if env_params['animation']:
        ani = camera.animate(interval=10, blit=True)
        path_ani = os.path.join(path_base, "ani.mp4")
        ani.save(path_ani)
    plt.close('all')
    logger.close()
    env.close()

def make_figure(path, params):
    data = logging.load(os.path.join(
        path, 'test.h5'
    ))
    quad_num = params['quad_num']
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

    pos_ylabel = ["X [m]", "Y [m]", "Height [m]"]
    make_figure_3col(
        time,
        load_pos*np.vstack((1, 1, -1)),
        "Position of payload",
        "time [s]",
        pos_ylabel,
        "load_pos",
        path
    )
    vel_ylabel = ["$V_x$ [m/s]", "$V_y$ [m/a]", "$V_z$ [m/s]"]
    make_figure_3col(
        time,
        load_vel,
        "Velocity of payload",
        "time [s]",
        vel_ylabel,
        "load_vel",
        path
    )
    ang_ylabel = ["$\phi$ [deg]", "$\\theta$ [deg]", "$\psi$ [deg]"]
    make_figure_3col(
        time,
        load_ang,
        "Euler angle of payload",
        "time [s]",
        ang_ylabel,
        "load_ang",
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
        "load_ang_rate",
        path
    )

    [make_figure_3col(
        time,
        quad_pos[:,i,:,:]*np.vstack((1, 1, -1)),
        f"Position of quadrotor {i}",
        "time [s]",
        pos_ylabel,
        f"quad_pos_{i}",
        path
    ) for i in range(quad_num)]

    [make_figure_3col(
        time,
        quad_vel[:,i,:,:],
        f"Velocity of quadrotor {i}",
        "time [s]",
        vel_ylabel,
        f"quad_vel_{i}",
        path
    ) for i in range(quad_num)]

    [make_figure_3col(
        time,
        quad_ang[:,i,:],
        f"Euler angle of quadrotor {i}",
        "time [s]",
        ang_ylabel,
        f"quad_ang_{i}",
        path,
        unwrap=True
    ) for i in range(quad_num)]

    [make_figure_3col(
        time,
        quad_ang_rate[:,i,:],
        f"Euler angle rate of quadrotor {i}",
        "time [s]",
        ang_rate_ylabel,
        f"quad_ang_rate_{i}",
        path
    ) for i in range(quad_num)]

    moment_ylabel = ["$M_x$ [Nm]", "$M_y$ [Nm]", "$M_z$ [Nm]"]
    [make_figure_3col(
        time,
        moment[:,i,:],
        f"Moment of quadrotor {i}",
        "time [s]",
        moment_ylabel,
        f"quad_moment_{i}",
        path
    ) for i in range(quad_num)]
    """
    for i in range(quad_num):
        fig, ax = plt.subplot(nrows=3, ncols=1)
        ax[0].plot(time, quad_pos[i][:,0])
        ax[1].plot(time, quad_pos[i][:,1])
        ax[2].plot(time, -quad_pos[i][:,2])
        ax[0].set_title(f"Position of quadrotor {i}")
        ax[0].set_ylabel("x [m]")
        ax[1].set_ylabel("y [m]")
        ax[2].set_ylabel("z [m]")
        ax[2].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(3)]
        fig.savefig(
            os.path.join(path, f"quad_pos_{i}"),
            bbox_inches='tight'
        )
        fig1, ax1 = plt.subplot(nrows=3, ncols=1)
        ax1[0].plot(time, quad_vel[i][:,0])
        ax1[1].plot(time, quad_vel[i][:,1])
        ax1[2].plot(time, quad_vel[i][:,2])
        ax1[0].set_title(f"Velocity of quadrotor {i}")
        ax1[0].set_ylabel("vx [m/s]")
        ax1[1].set_ylabel("vy [m/s]")
        ax1[2].set_ylabel("vz [m/s]")
        ax1[2].set_xlabel("time [s]")
        [ax1[i].grid(True) for i in range(3)]
        fig1.savefig(
            os.path.join(path, f"quad_vel_{i}"),
            bbox_inches='tight'
        )
        fig2, ax2 = plt.subplots(nrows=3, ncols=1)
        ax2[0].plot(time, quad_ang[i][:,0]*np.pi/180)
        ax2[1].plot(time, quad_ang[i][:,1]*np.pi/180)
        ax2[2].plot(time, quad_ang[i][:,2]*np.pi/180)
        ax2[0].set_title(f"Euler angle of quadrotor {i}")
        ax2[0].set_ylabel("Yaw [deg]")
        ax2[1].set_ylabel("Pitch [deg]")
        ax2[2].set_ylabel("Roll [deg]")
        ax2[2].set_xlabel("time [s]")
        [ax2[i].grid(True) for i in range(3)]
        fig2.savefig(
            os.path.join(path, f"quad_ang_{i}"),
            bbox_inches='tight'
        )
        fig3, ax3 = plt.subplots(nrows=3, ncols=1)
        ax3[0].plot(time, quad_ang_rate[i][:,0]*np.pi/180)
        ax3[1].plot(time, quad_ang_rate[i][:,1]*np.pi/180)
        ax3[2].plot(time, quad_ang_rate[i][:,2]*np.pi/180)
        ax3[0].set_title(f"Euler angle rate of quadrotor {i}")
        ax3[0].set_ylabel("Yaw rate [deg/s]")
        ax3[1].set_ylabel("Pitch rate [deg/s]")
        ax3[2].set_ylabel("Roll rate [deg/s]")
        ax3[2].set_xlabel("time [s]")
        [ax3[i].grid(True) for i in range(3)]
        fig3.savefig(
            os.path.join(path, f"quad_ang_rate_{i}"),
            bbox_inches='tight'
        )
    """

    plt.close('all')

def make_figure_3col(x, y, title, xlabel, ylabel, file_name, path, unwrap=False):
    fig, ax = plt.subplots(nrows=3, ncols=1)
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


def snap_ani(ax, info, params):
    load_pos = info['load_pos']
    quad_pos = info['quad_pos']
    load_rot_mat = info['load_rot_mat']
    rho = params['link_rho']
    quad_num = params['quad_num']

    anchor_pos = [load_pos + load_rot_mat.dot(rho[i]) for i in range(quad_num)]
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
    ax.set_xlim3d(-20, 20)
    ax.set_ylim3d(-20, 20)
    ax.set_zlim3d(-10, 30)

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
    quad_rot_mat_init = rot.angle2dcm(-np.pi/3, np.pi/4, np.pi/6).T # z-y-x ord
    anchor_radius = 1
    cg_bias = np.vstack((0.0, 0.0, 1))
    env_params = {
        'time_step': 0.01,
        'max_t': 5,
        'load_mass': 10,
        'load_pos_init': np.vstack((0.0, 0.0, -5.0)),
        'load_rot_mat_init': np.eye(3),
        'quad_num': quad_num,
        'quad_rot_mat_init': quad_num*[quad_rot_mat_init],
        'link_len': quad_num*[3],
        'link_init': quad_num*[np.vstack((0.0, 0.0, 1.0))],
        'link_rho': [
            3*np.vstack((
                anchor_radius * np.cos(i*2*np.pi/quad_num),
                anchor_radius * np.sin(i*2*np.pi/quad_num),
                0
            )) - cg_bias for i in range(quad_num)
        ],
        'gain': 5,
        'chatter_bound': 0.1,
        'unc_max': 1,
        'anchor_radius': anchor_radius,
        'cg_bias': cg_bias,
        'animation': False,
    }
    main(path_base, env_params)
    make_figure(path_base, env_params)

