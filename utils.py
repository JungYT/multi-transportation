import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.art3d as art3d
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib.patches import Circle

import fym.logging as logging
from fym.utils import rot


class OrnsteinUhlenbeckNoise:
    def __init__(self, rho, sigma, dt, size, x0=None):
        self.rho = rho
        self.mu = 0
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset()

    def reset(self):
        self.x = self.x0 if self.x0 is not None else np.zeros(self.size)

    def get_noise(self):
        self.x = self.x + self.rho*(self.mu-self.x)*self.dt \
            + np.sqrt(self.dt)*self.sigma*np.random.normal(size=self.size)
        return self.x

def softupdate(target, behavior, softupdate_const):
    for targetParam, behaviorParam in zip(
            target.parameters(),
            behavior.parameters()
    ):
        targetParam.data.copy_(
            targetParam.data*(1.0-softupdate_const) \
            + behaviorParam.data*softupdate_const
        )

def hardupdate(target, behavior):
    for targetParam, behaviorParam in zip(
            target.parameters(),
            behavior.parameters()
    ):
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

def draw_plot(path_data, path_fig):
    data, info = logging.load(path_data, with_info=True)
    cfg = info['cfg']

    time = data['time']
    load_pos = data['load']['pos']
    load_vel = data['load']['vel']
    # load_dcm = data['load']['dcm']
    load_att = np.unwrap(data['load_att'], axis=0) * 180/np.pi
    load_omega = np.unwrap(data['load']['omega'], axis=0) * 180/np.pi
    quad_moment = data['quad_moment']
    quad_pos = data['quad_pos']
    quad_vel = data['quad_vel']
    quad_att = np.unwrap(data['quad_att'], axis=0) * 180/np.pi
    quad_att_des = np.unwrap(data['quad_att_des'], axis=0) * 180/np.pi
    distance_btw_quads = data['distance_btw_quads']
    distance_btw_quad2anchor = data['distance_btw_quad2anchor']
    anchor_pos = data['anchor_pos']
    # check_dynamics = data['check_dynamics']

    for i in range(cfg.quad.num):
        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].plot(time, quad_pos[:,i,0])
        ax[1].plot(time, quad_pos[:,i,1])
        ax[2].plot(time, quad_pos[:,i,2])
        ax[0].set_title(f"Position of Quadrotor {i}")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].axes.xaxis.set_ticklabels([])
        ax[0].set_ylabel("X [m]")
        ax[1].set_ylabel("Y [m]")
        ax[2].set_ylabel("Z [m]")
        ax[2].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(3)]
        fig.align_ylabels(ax)
        fig.savefig(
            Path(path_fig, f"quad_{i}_pos.png"),
            bbox_inches='tight'
        )
        plt.close('all')

        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].plot(time, quad_vel[:,i,0])
        ax[1].plot(time, quad_vel[:,i,1])
        ax[2].plot(time, quad_vel[:,i,2])
        ax[0].set_title(f"Velocity of Quadrotor {i}")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].axes.xaxis.set_ticklabels([])
        ax[0].set_ylabel("$V_x$ [m/s]")
        ax[1].set_ylabel("$V_y$ [m/s]")
        ax[2].set_ylabel("$V_z$ [m/s]")
        ax[2].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(3)]
        fig.align_ylabels(ax)
        fig.savefig(
            Path(path_fig, f"quad_{i}_vel.png"),
            bbox_inches='tight'
        )
        plt.close('all')

        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].plot(time, quad_moment[:,i,0])
        ax[1].plot(time, quad_moment[:,i,1])
        ax[2].plot(time, quad_moment[:,i,2])
        ax[0].set_title(f"Moment of Quadrotor {i}")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].axes.xaxis.set_ticklabels([])
        ax[0].set_ylabel("$M_x$ [Nm]")
        ax[1].set_ylabel("$M_y$ [Nm]")
        ax[2].set_ylabel("$M_z$ [Nm]")
        ax[2].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(3)]
        fig.align_ylabels(ax)
        fig.savefig(
            Path(path_fig, f"quad_{i}_moment.png"),
            bbox_inches='tight'
        )
        plt.close('all')

        fig, ax = plt.subplots(nrows=3, ncols=1)
        line1, = ax[0].plot(time, quad_att[:,i,0], 'r')
        line2, = ax[0].plot(time, quad_att_des[:,i,0], 'b--')
        ax[0].legend(handles=(line1, line2), labels=('true', 'des.'))
        ax[1].plot(time, quad_att[:,i,1], 'r',
                   time, quad_att_des[:,i,1], 'b--')
        ax[2].plot(time, quad_att[:,i,2], 'r',
                   time, quad_att_des[:,i,2], 'b--')
        ax[0].set_title(f"Euler angle of Quadrotor {i}")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].axes.xaxis.set_ticklabels([])
        ax[0].set_ylabel("$\phi$ [deg]")
        ax[1].set_ylabel("$\\theta$ [deg]")
        ax[2].set_ylabel("$\psi$ [deg]")
        ax[2].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(3)]
        fig.align_ylabels(ax)
        fig.savefig(
            Path(path_fig, f"quad_{i}_att.png"),
            bbox_inches='tight'
        )
        plt.close('all')

        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].plot(time, data['quads'][f'quad{i:02d}']['omega'][:,0,:])
        ax[1].plot(time, data['quads'][f'quad{i:02d}']['omega'][:,1,:])
        ax[2].plot(time, data['quads'][f'quad{i:02d}']['omega'][:,2,:])
        ax[0].set_title(f"Angular velocity of Quadrotor {i}")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].axes.xaxis.set_ticklabels([])
        ax[0].set_ylabel("$\omega_x$ [deg/s]")
        ax[1].set_ylabel("$\omega_y$ [deg/s]")
        ax[2].set_ylabel("$\omega_z$ [deg/s]")
        ax[2].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(3)]
        fig.align_ylabels(ax)
        fig.savefig(
            Path(path_fig, f"quad_{i}_omega.png"),
            bbox_inches='tight'
        )
        plt.close('all')

    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(time, load_pos[:,0])
    ax[1].plot(time, load_pos[:,1])
    ax[2].plot(time, load_pos[:,2])
    ax[0].set_title("Position of Load")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].axes.xaxis.set_ticklabels([])
    ax[0].set_ylabel("X [m]")
    ax[1].set_ylabel("Y [m]")
    ax[2].set_ylabel("Z [m]")
    ax[2].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(3)]
    fig.align_ylabels(ax)
    fig.savefig(
        Path(path_fig, "load_pos.png"),
        bbox_inches='tight'
    )
    plt.close('all')

    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(time, load_vel[:,0])
    ax[1].plot(time, load_vel[:,1])
    ax[2].plot(time, load_vel[:,2])
    ax[0].set_title("Velocity of Load")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].axes.xaxis.set_ticklabels([])
    ax[0].set_ylabel("$V_x$ [m/s]")
    ax[1].set_ylabel("$V_y$ [m/s]")
    ax[2].set_ylabel("$V_z$ [m/s]")
    ax[2].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(3)]
    fig.align_ylabels(ax)
    fig.savefig(
        Path(path_fig, "load_vel.png"),
        bbox_inches='tight'
    )
    plt.close('all')

    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(time, load_att[:,0])
    ax[1].plot(time, load_att[:,1])
    ax[2].plot(time, load_att[:,2])
    ax[0].set_title("Euler angle of Load")
    ax[0].set_ylabel("$\phi$ [deg]")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].axes.xaxis.set_ticklabels([])
    ax[1].set_ylabel("$\\theta$ [deg]")
    ax[2].set_ylabel("$\psi$ [deg]")
    ax[2].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(3)]
    fig.align_ylabels(ax)
    fig.savefig(
        Path(path_fig, "load_att.png"),
        bbox_inches='tight'
    )
    plt.close('all')

    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(time, load_omega[:,0])
    ax[1].plot(time, load_omega[:,1])
    ax[2].plot(time, load_omega[:,2])
    ax[0].set_title("Angular rate of Load")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].axes.xaxis.set_ticklabels([])
    ax[0].set_ylabel("$\omega_x$ [deg/s]")
    ax[1].set_ylabel("$\omega_y$ [deg/s]")
    ax[2].set_ylabel("$\omega_z$ [deg/s]")
    ax[2].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(3)]
    fig.align_ylabels(ax)
    fig.savefig(
        Path(path_fig, "load_omega.png"),
        bbox_inches='tight'
    )
    plt.close('all')

    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(time, distance_btw_quad2anchor[:,0])
    ax[1].plot(time, distance_btw_quad2anchor[:,1])
    ax[2].plot(time, distance_btw_quad2anchor[:,2])
    ax[0].set_title("Distance between quadrotor and anchor")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].axes.xaxis.set_ticklabels([])
    ax[0].set_ylabel("Quad 0 [m]")
    ax[1].set_ylabel("Quad 1 [m]")
    ax[2].set_ylabel("Quad 2 [m]")
    ax[2].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(3)]
    [ax[i].set_ylim(cfg.link.len[i]-0.1, cfg.link.len[i]+0.1)
     for i in range(cfg.quad.num)]
    fig.align_ylabels(ax)
    fig.savefig(
        Path(path_fig, "distance_btw_quad2anchor.png"),
        bbox_inches='tight'
    )
    plt.close('all')

    fig, ax = plt.subplots(nrows=1, ncols=1)
    line1, = ax.plot(time, distance_btw_quads[:,0], 'r')
    line2, = ax.plot(time, distance_btw_quads[:,1], 'g')
    line3, = ax.plot(time, distance_btw_quads[:,2], 'b')
    line4, = ax.plot(time, cfg.quad.iscollision*np.ones(len(time)), 'k--')
    ax.legend(handles=(line1, line2, line3, line4),
              labels=(
                  'quad0<->quad1',
                  'quad1<->quad2',
                  'quad2<->quad0',
                  'collision criteria'
              ))
    ax.set_title("Distance between quadrotors")
    ax.set_ylabel('distance [m]')
    ax.set_xlabel("time [s]")
    ax.grid(True)
    fig.savefig(
        Path(path_fig, "distance_btw_quads.png"),
        bbox_inches='tight'
    )
    plt.close('all')


    quad_pos_test = quad_pos[:,:,0,:]
    quad_dcm_test = data['quads']['quad00']['dcm']

    fig = plt.figure()
    ax = axes3d.Axes3D(fig)
    # ax.set_xlim3d([load_pos[:,0,:].min()-10, load_pos[:,0,:].max()+10])
    # ax.set_xlabel('X [m]')
    # ax.set_ylim3d([load_pos[:,1,:].min()-10, load_pos[:,1,:].max()+10])
    # ax.set_xlabel('Y [m]')
    # ax.set_zlim3d([-load_pos[:,2,:].max()-10, load_pos[:,2,:].min()+10])
    # ax.set_xlabel('Height [m]')
    ani = Animator(fig, ax, quad_pos_test, quad_dcm_test)
    ani.animate()
    ani.save(Path(path_fig, "test-animation.mp4"))
    breakpoint()

class Quad_ani:
    def __init__(self, ax, quad_pos, dcm):
        d = 0.315
        r = 0.15

        body_segs = np.array([
            [[d, 0, 0], [0, 0, 0]],
            [[-d, 0, 0], [0, 0, 0]],
            [[0, d, 0], [0, 0, 0]],
            [[0, -d, 0], [0, 0, 0]]
        ])
        colors = (
            (1, 0, 0, 1),
            (0, 0, 1, 1),
            (0, 0, 1, 1),
            (0, 0, 1, 1),
        )
        self.body = art3d.Line3DCollection(
            body_segs,
            colors=colors,
            linewidths=3
        )

        kwargs = dict(radius=r, ec="k", fc="k", alpha=0.3)
        self.rotors = [
            Circle((d, 0), **kwargs),
            Circle((0, d), **kwargs),
            Circle((-d, 0), **kwargs),
            Circle((0, -d), **kwargs),
        ]

        ax.add_collection3d(self.body)
        for rotor in self.rotors:
            ax.add_patch(rotor)
            art3d.pathpatch_2d_to_3d(rotor, z=0)

        self.body._base = self.body._segments3d
        for rotor in self.rotors:
            rotor._segment3d = np.array(rotor._segment3d)
            rotor._center = np.array(rotor._center + (0,))
            rotor._base = rotor._segment3d

        self.set(quad_pos[0].squeeze(), dcm[0].squeeze())

    def set(self, pos, dcm=np.eye(3)):
        self.body._segments3d = np.array([
            dcm @ point for point in self.body._base.reshape(-1, 3)
        ]).reshape(self.body._base.shape)

        for rotor in self.rotors:
            rotor._segment3d = np.array([
                dcm @ point for point in rotor._base
            ])

        self.body._segments3d += pos
        for rotor in self.rotors:
            rotor._segment3d += pos


class Animator:
    def __init__(self, fig, ax, quad_pos, quad_dcm):
        self.offsets = ['collections', 'patches', 'lines', 'texts',
                        'artists', 'images']
        self.fig = fig
        self.ax = ax
        self.quad_pos = quad_pos
        self.dcm = quad_dcm

    def init(self):
        self.frame_artists = []

        self.ax.quad = Quad_ani(self.ax, self.quad_pos, self.dcm)
        self.ax.set_xlim3d([-5, 5])
        self.ax.set_ylim3d([-5, 5])
        self.ax.set_zlim3d([-1, 5])
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        for name in self.offsets:
            self.frame_artists += getattr(self.ax, name)

        return self.frame_artists

    def get_sample(self, frame):
        self.init()
        self.update(frame)
        self.fig.show()

    def update(self, frame):
        self.ax.quad.set(self.quad_pos[frame].squeeze(), self.dcm[frame].squeeze())
        return self.frame_artists

    def animate(self, *args, **kwargs):
        frames = range(0, len(self.quad_pos), 10)
        self.anim = FuncAnimation(
            self.fig, self.update, init_func=self.init,
            frames=frames, interval=1000, blit=True,
            *args, **kwargs
        )

    def save(self, path, *args, **kwargs):
        self.anim.save(path, writer="ffmpeg", fps=30, *args, **kwargs)


