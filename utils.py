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
    def __init__(self, rho, mu, sigma, dt, size, x0=None):
        self.rho = rho
        self.mu = mu
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

def unhat(R):
    return np.vstack((R[1][0], R[0][2], R[2][1]))

def wrap(angle):
    return (angle+np.pi) % (2*np.pi) - np.pi

def block_diag(R, num):
    rows, cols = R.shape
    result = np.zeros((num, rows, num, cols), dtype=R.dtype)
    diag = np.einsum('ijik->ijk', result)
    diag[:] = R
    return result.reshape(rows * num, cols * num)

def split_int(num):
    num_sqrt = np.sqrt(num)
    if np.isclose(num_sqrt, int(num_sqrt)):
        result = 2*[int(num_sqrt)]
    elif (int(num_sqrt)+1) * int(num_sqrt) >= num:
        result = [int(num_sqrt), int(num_sqrt)+1]
    else:
        result = [int(num_sqrt)+1, int(num_sqrt)+1]
    return result

def draw_plot(dir_env_data, dir_agent_data, dir_save):
    env_data, info = logging.load(dir_env_data, with_info=True)
    agent_data = logging.load(dir_agent_data)
    cfg = info['cfg']

    time = env_data['time']
    load_pos = env_data['load']['pos']
    load_vel = env_data['load']['vel']
    # load_dcm = data['load']['dcm']
    load_att = np.unwrap(env_data['load_att'], axis=0) * 180/np.pi
    load_omega = np.unwrap(env_data['load']['omega'], axis=0) * 180/np.pi
    quad_moment = env_data['quad_moment']
    quad_pos = env_data['quad_pos']
    quad_vel = env_data['quad_vel']
    quad_att = np.unwrap(env_data['quad_att'], axis=0) * 180/np.pi
    quad_att_des = np.unwrap(env_data['quad_att_des'], axis=0) * 180/np.pi
    distance_btw_quads = env_data['distance_btw_quads']
    distance_btw_quad2anchor = env_data['distance_btw_quad2anchor']
    link = env_data['links']

    time_agent = agent_data['time']
    action = agent_data['action']
    reward = agent_data['reward']
    tension = agent_data['tension']
    tension_des = agent_data['tension_des']
    # anchor_pos = env_data['anchor_pos']
    # check_dynamics = env_data['check_dynamics']
    # breakpoint()

    for i in range(cfg.quad.num):
        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].plot(time, link[f"link{i:02d}"]['omega'][:,0,0])
        ax[1].plot(time, link[f"link{i:02d}"]['omega'][:,1,0])
        ax[2].plot(time, link[f"link{i:02d}"]['omega'][:,2,0])
        ax[0].set_title(f"Angular velocity of link {i}")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].axes.xaxis.set_ticklabels([])
        ax[0].set_ylabel("$\omega_x$ [rad/s]")
        ax[1].set_ylabel("$\omega_y$ [rad/s]")
        ax[2].set_ylabel("$\omega_z$ [rad/s]")
        ax[2].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(3)]
        fig.align_ylabels(ax)
        fig.savefig(
            Path(dir_save, f"link_{i}_omega.png"),
            bbox_inches='tight'
        )
        plt.close('all')

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
            Path(dir_save, f"quad_{i}_pos.png"),
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
            Path(dir_save, f"quad_{i}_vel.png"),
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
            Path(dir_save, f"quad_{i}_moment.png"),
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
            Path(dir_save, f"quad_{i}_att.png"),
            bbox_inches='tight'
        )
        plt.close('all')

        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].plot(time, env_data['quads'][f'quad{i:02d}']['omega'][:,0,:])
        ax[1].plot(time, env_data['quads'][f'quad{i:02d}']['omega'][:,1,:])
        ax[2].plot(time, env_data['quads'][f'quad{i:02d}']['omega'][:,2,:])
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
            Path(dir_save, f"quad_{i}_omega.png"),
            bbox_inches='tight'
        )
        plt.close('all')

        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].plot(time_agent, action[:,i,0])
        ax[1].plot(time_agent, action[:,i,1]*180/np.pi)
        ax[2].plot(time_agent, action[:,i,2]*180/np.pi)
        ax[0].set_title(f"Action of Quadrotor {i}")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].axes.xaxis.set_ticklabels([])
        ax[0].set_ylabel("Total thrust [N]")
        ax[1].set_ylabel("$chi$ [deg]")
        ax[2].set_ylabel("$gamma$ [deg]")
        ax[2].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(3)]
        fig.align_ylabels(ax)
        fig.savefig(
            Path(dir_save, f"quad_{i}_action.png"),
            bbox_inches='tight'
        )
        plt.close('all')

        fig, ax = plt.subplots(nrows=3, ncols=1)
        line1, = ax[0].plot(time_agent, tension[:,i,0], 'r')
        line2, = ax[0].plot(time_agent, tension_des[:,i,0], 'b--')
        ax[0].legend(handles=(line1, line2), labels=('estimates', 'des.'))
        ax[1].plot(time_agent, tension[:,i,1], 'r',
                   time_agent, tension_des[:,i,1], 'b--')
        ax[2].plot(time_agent, tension[:,i,2], 'r',
                   time_agent, tension_des[:,i,2], 'b--')
        ax[0].set_title(f"Desired tension and estimates of Quadrotor {i}")
        ax[0].axes.xaxis.set_ticklabels([])
        ax[1].axes.xaxis.set_ticklabels([])
        ax[0].set_ylabel("X [N]")
        ax[1].set_ylabel("Y [N]")
        ax[2].set_ylabel("Z [N]")
        ax[2].set_xlabel("time [s]")
        [ax[i].grid(True) for i in range(3)]
        fig.align_ylabels(ax)
        fig.savefig(
            Path(dir_save, f"quad_{i}_tension.png"),
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
        Path(dir_save, "load_pos.png"),
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
        Path(dir_save, "load_vel.png"),
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
        Path(dir_save, "load_att.png"),
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
        Path(dir_save, "load_omega.png"),
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
        Path(dir_save, "distance_btw_quad2anchor.png"),
        bbox_inches='tight'
    )
    plt.close('all')

    fig, ax = plt.subplots(nrows=1, ncols=1)
    line1, = ax.plot(time, distance_btw_quads[:,1], 'r')
    line2, = ax.plot(time, distance_btw_quads[:,2], 'g')
    line3, = ax.plot(time, distance_btw_quads[:,0], 'b')
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
        Path(dir_save, "distance_btw_quads.png"),
        bbox_inches='tight'
    )
    plt.close('all')

    fig, _ = plt.subplots(1, 1, subplot_kw=dict(projection="3d"))
    ani = Animator(fig, [env_data], cfg)
    ani.animate()
    ani.save(Path(dir_save, "animation.mp4"))


class Quad_ani:
    def __init__(self, ax, cfg):
        d = cfg.animation.quad_size
        r = cfg.animation.rotor_size

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
            linewidths=2
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


    def set(self, pos, dcm=np.eye(3)):
        self.body._segments3d = np.array([
            dcm @ point for point in self.body._base.reshape(-1, 3)
        ]).reshape(self.body._base.shape)

        for rotor in self.rotors:
            rotor._segment3d = np.array([
                dcm @ point for point in rotor._base
            ])

        self.body._segments3d = self.body._segments3d + pos
        for rotor in self.rotors:
            rotor._segment3d += pos

class Link_ani:
    def __init__(self, ax):
        self.link = art3d.Line3D(
            [ ],
            [ ],
            [ ],
            color="k",
            linewidth=1
        )
        ax.add_line(self.link)

    def set(self, quad_pos, anchor_pos):
        self.link.set_data_3d(
            [anchor_pos[0], quad_pos[0]],
            [anchor_pos[1], quad_pos[1]],
            [anchor_pos[2], quad_pos[2]]
        )

class Payload_ani:
    def __init__(self, ax, edge_num):
        load_segs = np.array(edge_num*[[[0, 0, 0],[1, 1, 1]]])
        colors = edge_num*["k"]
        self.edge_num = edge_num

        self.load = art3d.Line3DCollection(
            load_segs,
            colors=colors,
            linewidths=1.5
        )
        ax.add_collection3d(self.load)

        verts = [3*[[1, 1, 1]]]
        self.cover = art3d.Poly3DCollection(verts)
        self.cover.set_facecolor(colors="k")
        self.cover.set_edgecolor(colors="k")
        self.cover.set_alpha(alpha=0.3)
        ax.add_collection3d(self.cover)

    def set(self, load_verts, load_cg):
        load_edge = [[load_verts[i], load_cg] for i in range(self.edge_num)]
        self.load.set_segments(load_edge)
        verts = [load_verts]
        self.cover.set_verts(verts)


class Animator:
    def __init__(self, fig, data_list, cfg, simple=False):
        self.offsets = ['collections', 'patches', 'lines', 'texts',
                        'artists', 'images']
        self.fig = fig
        self.axes = fig.axes
        # self.axes = axes
        self.data_list = data_list
        self.cfg = cfg
        self.len = len(data_list)
        self.simple = simple

    def init(self):
        self.frame_artists = []
        max_x = np.array(
            [data['load']['pos'][:,0,:].max() for data in self.data_list]
        ).max()
        min_x = np.array(
            [data['load']['pos'][:,0,:].min() for data in self.data_list]
        ).min()
        max_y = np.array(
            [data['load']['pos'][:,1,:].max() for data in self.data_list]
        ).max()
        min_y = np.array(
            [data['load']['pos'][:,1,:].min() for data in self.data_list]
        ).min()
        max_z = np.array(
            [data['load']['pos'][:,2,:].max() for data in self.data_list]
        ).max()
        min_z = np.array(
            [data['load']['pos'][:,2,:].min() for data in self.data_list]
        ).min()

        for i, ax in enumerate(self.axes):
            ax.quad = [Quad_ani(ax, self.cfg)
                       for _ in range(self.cfg.quad.num)]
            ax.link = [Link_ani(ax) for _ in range(self.cfg.quad.num)]
            ax.load = Payload_ani(ax, self.cfg.quad.num)
            ax.set_xlim3d([
                min_x - self.cfg.load.size - self.cfg.animation.quad_size,
                max_x + self.cfg.load.size + self.cfg.animation.quad_size
            ])
            ax.set_ylim3d([
                min_y - self.cfg.load.size - self.cfg.animation.quad_size,
                max_y + self.cfg.load.size + self.cfg.animation.quad_size
            ])
            ax.set_zlim3d([
                max(min_z - self.cfg.link.len[0], 0.),
                max_z + self.cfg.link.len[0] - self.cfg.load.cg[2]\
                + self.cfg.animation.quad_size
            ])
            ax.view_init(
                self.cfg.animation.view_angle[0],
                self.cfg.animation.view_angle[1]
            )
            if i >= self.len:
                ax.set_title("empty", fontsize='small', fontweight='bold')
            else:
                ax.set_title(
                    f"{(i+1)*self.cfg.epi_eval:05d}_epi",
                    fontsize='small',
                    fontweight='bold'
                )
            if self.simple:
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                ax.axes.zaxis.set_ticklabels([])
            else:
                ax.set_xlabel('x [m]', fontsize='small')
                ax.set_ylabel('y [m]', fontsize='small')
                ax.set_zlabel('z [m]', fontsize='small')
                ax.tick_params(axis='both', which='major', labelsize=8)

            for name in self.offsets:
                self.frame_artists += getattr(ax, name)
        self.fig.tight_layout()
        if not self.simple:
            self.fig.subplots_adjust(
                left=0,
                bottom=0.1,
                right=1,
                top=0.95,
                hspace=0.5,
                wspace=0
            )


        return self.frame_artists

    def get_sample(self, frame):
        self.init()
        self.update(frame)
        self.fig.show()

    def update(self, frame):
        for data, ax in zip(self.data_list, self.axes):
            load_verts = data['anchor_pos'][frame].squeeze().tolist()
            load_cg = data['load']['pos'][frame].squeeze().tolist()
            ax.load.set(load_verts, load_cg)

            for i in range(self.cfg.quad.num):
                ax.quad[i].set(
                    data["quad_pos"][frame,i,:,:].squeeze(),
                    data["quads"][f"quad{i:02d}"]["dcm"][frame,:,:].squeeze()
                )
                ax.link[i].set(
                    data["quad_pos"][frame,i,:,:].squeeze(),
                    data["anchor_pos"][frame,i,:,:].squeeze()
                )
        return self.frame_artists

    def animate(self, *args, **kwargs):
        data_len = [len(self.data_list[i]['time'])
                    for i in range(self.len)]
        frames = range(0, min(data_len), 10)
        self.anim = FuncAnimation(
            self.fig, self.update, init_func=self.init,
            frames=frames, interval=200, blit=True,
            *args, **kwargs
        )

    def save(self, path, *args, **kwargs):
        self.anim.save(path, writer="ffmpeg", fps=30, *args, **kwargs)


def compare_episode(past, ani=True):
    dir_save = list(Path('log').glob("*"))[past]
    epi_list = [x for x in dir_save.glob("*")]
    env_data_list = [
        logging.load(Path(epi_dir, "env_data.h5")) for epi_dir in epi_list
    ]
    agent_data_list = [
        logging.load(Path(epi_dir, "agent_data.h5")) for epi_dir in epi_list
    ]
    _, info = logging.load(Path(epi_list[0], "env_data.h5"), with_info=True)
    cfg = info['cfg']
    if ani == True:
        data_num = len(env_data_list)
        fig_shape = split_int(data_num)
        simple = False
        if fig_shape[0] >= 3:
            simple=True

        fig, _ = plt.subplots(
            fig_shape[0],
            fig_shape[1],
            subplot_kw=dict(projection="3d"),
        )

        ani = Animator(fig, env_data_list, cfg, simple=simple)
        ani.animate()
        ani.save(Path(dir_save, "compare-animation.mp4"))
        plt.close('all')

    return_list = []
    for i, data in enumerate(agent_data_list):
        G = [0]*cfg.quad.num
        for r in data['reward'][::-1]:
            for j in range(cfg.quad.num):
                G[j] = r[j].item() + cfg.ddpg.discount*G[j]
            # G = r.item() + cfg.ddpg.discount*G
        return_list_tmp = [(i+1)*cfg.epi_eval]
        for j in range(cfg.quad.num):
            return_list_tmp.append(G[j])
        return_list.append(return_list_tmp)
    return_list = np.array(return_list)

    for i in range(cfg.quad.num):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(return_list[:,0], return_list[:,i+1], "*")
        ax.set_title(f"Return for {i}th quadrotor")
        ax.set_ylabel("Return")
        ax.set_xlabel("Episode")
        ax.grid(True)
        fig.savefig(
            Path(dir_save, f"return_quad_{i}.png"),
            bbox_inches='tight'
        )
        plt.close('all')





