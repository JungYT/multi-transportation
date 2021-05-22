import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import fym.logging as logging
from fym.utils import rot

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

def make_figure(path, epi_num, cfg):
    data = logging.load(
        Path(path, f"data_{epi_num:05d}.h5")
    )
    data_test = logging.load('test.h5')

    quad_num = cfg.quad.num
    link_len = cfg.link.len
    time = data['time']
    load_pos = data['load_pos']
    load_vel = data['load_vel']
    load_ang = np.unwrap(data['load_ang'])*180/np.pi
    load_omega = np.unwrap(data['load_omega'])*180/np.pi
    quad_pos = data['quad_pos']
    quad_vel = data['quad_vel']
    quad_ang = np.unwrap(data['quad_ang'])*180/np.pi
    quad_omega = np.unwrap(data['quad_omega'])*180/np.pi
    moment = data['moment']
    distance = data['distance']
    collisions = data['collisions']
    reward = data['reward']
    des_attitude = np.unwrap(data['des_attitude'].squeeze())*180/np.pi
    des_force = data['des_force']
    error = data['error']
    moment_test = data_test['moment']
    time_test = data_test['time']
    quad_dcm_test = data_test['quads']['quad00']['dcm']
    quad_omega_test = data_test['quads']['quad00']['omega']

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
        path
    )
    omega_ylabel = [
        "$\dot{\phi}$ [deg/s]",
        "$\dot{\\theta}$ [deg/s]",
        "$\dot{\psi}$ [deg/s]"
    ]
    make_figure_3col(
        time,
        load_omega,
        "Euler angle rate of payload",
        "time [s]",
        omega_ylabel,
        f"load_omega_{epi_num:05d}",
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
        path
    ) for i in range(quad_num)]

    [make_figure_3col(
        time,
        quad_omega[:,i,:],
        f"Euler angle rate of quadrotor {i}",
        "time [s]",
        omega_ylabel,
        f"quad_omega_{i}_{epi_num:05d}",
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

    moment_ylabel = ["$M_x$ [Nm]", "$M_y$ [Nm]", "$M_z$ [Nm]"]
    [make_figure_3col(
        time_test,
        moment_test[:,i,:],
        f"Moment of quadrotor {i}",
        "time [s]",
        moment_ylabel,
        f"test_{i}_{epi_num:05d}",
        path
    ) for i in range(quad_num)]

    distance_ylabel = ["quad0 [m]", "quad1 [m]", "quad2 [m]"]
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
        Path(path, f"distance_link_to_anchor_{epi_num:05d}"),
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
        Path(path, f"distance_btw_quads_{epi_num:05d}"),
        bbox_inches='tight'
    )

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(time, reward, 'r')
    ax.set_title("Reward")
    ax.set_ylabel('reward')
    ax.set_xlabel("time [s]")
    ax.grid(True)
    fig.savefig(
        Path(path, f"reward_{epi_num:05d}"),
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
        Path(path, f"des_net_force_{epi_num:05d}"),
        bbox_inches='tight'
    )

    plt.close('all')
    # breakpoint()

    G = 0
    for r in reward[::-1]:
        G = r.item() + 0.999*G
    return G

def make_figure_3col(x, y, title, xlabel, ylabel,
                     file_name, path):
    fig, ax = plt.subplots(nrows=3, ncols=1)# {{{
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
        Path(path, file_name),
        bbox_inches='tight'
    )
    plt.close('all')# }}}

def make_figure_compare(x, y1, y2, legend, title, xlabel, ylabel,
                     file_name, path):
    fig, ax = plt.subplots(nrows=3, ncols=1)# {{{
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
        Path(path, file_name),
        bbox_inches='tight'
    )
    plt.close('all')# }}}

def snap_ani(ax, info, cfg):
    load_pos = info['load_pos']# {{{
    quad_pos = info['quad_pos']
    quad_num = cfg.quad.num
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
