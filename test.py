import numpy as np
import numpy.linalg as nla
import os
from datetime import datetime
from multiprocessing import Pool
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from matplotlib import pyplot as plt
import time
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.art3d as art3d
import mpl_toolkits.mplot3d.axes3d as ax3d
from matplotlib.patches import Circle

from fym.core import BaseEnv, BaseSystem
import fym.core as core
import fym.logging as logging
from fym.utils import rot


class Test(BaseEnv):
    def __init__(self, a, b):
        super().__init__(dt=0.1, max_t=5, solver="rk4")
        #self.pos = BaseSystem(np.array([1, 2]))
        #self.vel = BaseSystem(np.array([0, 3]))
        self.pos = BaseSystem(np.vstack((1, 2)))
        self.vel = BaseSystem(np.vstack((0, 3)))
        self.a = a
        self.b = b

    def set_dot(self, t):
        pos = self.pos.state
        vel = self.pos.state
        self.pos.dot = vel
        self.vel.dot = -self.a * vel - self.b * pos

    def reset(self):
        super().reset()

    def step(self):
        *_, done = self.update()
        info = {
            'pos': self.pos.state,
            'vel': self.vel.state,
        }
        return done, info

    def logger_callback(self, i, t, y, *args):
        return dict(time=t, pos=self.pos.state)

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.lin1 = nn.Linear(3, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 16)
        self.lin4 = nn.Linear(16, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.relu(self.lin1(x))
        x2 = self.relu(self.lin2(x1))
        x3 = self.relu(self.lin3(x2))
        x4 = self.tanh(self.lin4(x3))
        return x4

def main():
    # memory = deque()
    # x = np.array([1, 2, 3])
    # temp = np.array([4, 5])
    # item = (x, temp)
    # actor = ActorNet()
    # for i in range(10):
    #     memory.append(item)
    # x = np.vstack((1, 2, 3))
    # state, action = zip(*memory)
    # y = actor(torch.FloatTensor(x))
    # env = Test(1, 2)
    # env.logger = logging.Logger("temp.h5")
    # env.reset()
    # done, info = env.step()
    # env.close()
    # data = logging.load("temp.h5")
    # breakpoint()

    ite = 1
    start1 = time.time()
    for k in range(ite):
        a1 = [i + 100*np.sqrt(2) for i in range(3)]
        b1 = [i + 100*np.sqrt(2) for i in range(3)]
        c1 = [i + 100*np.sqrt(2) for i in range(3)]
    end1 = time.time()
    start2 = time.time()
    for k in range(ite):
        a = []
        b = []
        c = []
        for i in range(3):
            a.append(i + 100*np.sqrt(2))
            b.append(i + 100*np.sqrt(2))
            c.append(i + 100*np.sqrt(2))
    end2 = time.time()

    print("time1: :", end1 - start1)
    print("time2: :", end2 - start2)



def test():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot3D(1, 2, 3, marker="X")
    plt.show()

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

        self.set(quad_pos[0], dcm[0])

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
    def __init__(self, fig, data_set):
        self.offsets = ['collections', 'patches', 'lines', 'texts',
                        'artists', 'images']
        self.fig = fig
        self.axes = fig.axes
        self.data_set = data_set
        self.quad_pos = data_set['quad']
        self.dcm = data_set['dcm']

    def init(self):
        self.frame_artists = []

        for ax in self.axes:
            ax.quad = Quad_ani(ax, self.quad_pos, self.dcm)
            ax.set_xlim3d([-1, 1])
            ax.set_ylim3d([-1, 1])
            ax.set_zlim3d([-1, 1])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            for name in self.offsets:
                self.frame_artists += getattr(ax, name)

        return self.frame_artists

    def get_sample(self):
        self.init()
        self.update(0)
        self.fig.show()

    def update(self, frame):
        for data, ax in zip(self.data_set, self.axes):
            pos = data
            R
            ax.quad.set(pos, R)
        return self.frame_artists

    def animate(self, *args, **kwargs):
        frames = range(0, 1000, 10)
        self.anim = FuncAnimation(
            self.fig, self.update, init_func=self.init,
            frames=frames, interval=10, blit=True,
            *args, **kwargs
        )



def Animator():

        # segments = [[d, 0, 0], [-d, 0, 0], [0, d, 0], [0, -d, 0]]
        # rotor_pos = [dcm@seg + pos for seg in segments]
        # body_segs = np.array([[rotor_pos[i], pos] for i in range(4)])

        # for i, rotor in enumerate(rotors):
        #     ax.add_patch(rotor)
        #     pathpatch_2d_to_3d(rotor, z=rotor_pos[i][2], dcm=dcm)
            # pathpatch_2d_to_3d(rotor, z=0, dcm=dcm)

        fig = plt.figure()
        ax = ax3d.Axes3D(fig)
        ax.set_xlim3d([-1, 1])
        ax.set_xlabel('x')
        ax.set_ylim3d([-1, 1])
        ax.set_ylabel('y')
        ax.set_zlim3d([-1, 1])
        ax.set_zlabel('z')
        quad_ani = Quad_ani(ax)
        # quad_ani.set(np.array([0, 0, 0]), np.eye(3))
        quad_ani.set(np.array([0, 0, 0]), rot.angle2dcm(0, np.pi/3, 0).T)

        breakpoint()

    # quad_ani(np.array((0, 0, 0)), rot.angle2dcm(0, np.pi/3, 0).T)






if __name__ == "__main__":
    Animator()
