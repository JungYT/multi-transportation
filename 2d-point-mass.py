import numpy as np
from fym.core import BaseEnv, BaseSystem
import fym.core as core
import fym.logging as logging
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from fym.agents import PID

torch.manual_seed(0)
np.random.seed(0)

g = 9.81 * np.vstack((0., 0., 1.))
zero = np.vstack((0., 0., 0.))

def hat(v):
    v1, v2, v3 = v.squeeze()
    return np.array([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0]
    ])


class Payload(BaseEnv):
    def __init__(self):
        super().__init__()
        self.pos = BaseSystem(zero)
        self.vel = BaseSystem(zero)
        self.m = 1.

    def set_dot(self, xddot):
        vel = self.vel.state

        self.pos.dot = vel
        self.vel.dot = xddot


class Link(BaseEnv):
    def __init__(self):
        super().__init__()
        self.q = BaseSystem(zero)
        self.w = BaseSystem(zero)
        self.m = 1.
        self.l = 0.5

    def set_dot(self, u, xddot):
        q = self.q.state
        w = self.w.state
        self.q.dot = hat(w).dot(q)
        self.w.dot = hat(q).dot(xddot - g - u / self.m) / self.l


class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.01, max_t=50)
        self.payload = Payload()
        self.links = core.Sequential(**{f"link{i:02d}":Link() for i in range(2)
                                       })
        self.gain = np.array([1, 0.1, 0.1])
        self.pid = PID.PID(self.gain)

    def reset(self):
        super().reset()
        self.payload.pos.state = np.vstack((0., 0.,
                                            np.random.uniform(low=-1, high=0)))
        i = 0
        for system in self.links.systems:
            if i == 0:
                tmp = np.vstack((np.random.uniform(low=0, high=2), 0.,
                                 np.random.uniform(low=0, high=1)))
            else:
                tmp = np.vstack((np.random.uniform(low=-2, high=0), 0.,
                                 np.random.uniform(low=0, high=1)))

            system.q.state = tmp / np.linalg.norm(tmp)
            i += 1
        print("debug")

        return self.state

    def step(self, action):
        u = self.controller(action)
        *_, done = self.update(u=u)
        r = self.reward(u)
        info = {
        }
        self.logger.record(**info)
        return self.state, r, done

    def set_dot(self, t, u):
        mt = self.payload.m
        for system in self.links.systems:
            mt += system.m

        Mq = mt * np.eye(3)
        for system in self.links.systems:
            q = system.q.state
            w = system.w.state
            Mq += system.m * hat(q).dot(hat(q))

        S = np.vstack((0., 0., 0.))
        i = 0
        for system in self.links.systems:
            q = system.q.state
            w = system.w.state
            S += system.m * (hat(q).dot(hat(q)).dot(g) \
                             - system.l * (np.transpose(w).dot(w)) * q) \
                + (np.eye(3) + hat(q).dot(hat(q))).dot(u[3*i:3*(i+1)])
            i += 1

        S += mt * g
        xddot = np.linalg.inv(Mq).dot(S)
        self.payload.set_dot(xddot)

        i = 0
        for system in self.links.systems:
            system.set_dot(u[3*i:3*(i+1)], xddot)
            i += 1

    def reward(self, u):
        return 1

    def controller(self, action):

        action1 = action[0:3]
        action2 = action[3:6]
        x0 = self.payload.pos.state

        q1 = self.links.link00.q.state
        l1 = self.links.link00.l
        x1 = x0 - l1 * q1
        e1 = action1 - x1

        q2 = self.links.link01.q.state
        l2 = self.links.link01.l
        x2 = x0 - l2 * q2
        e2 = action2 - x2

        u1 = self.pid.get(e1)
        u1[1] = 0.
        u2 = self.pid.get(e2)
        u2[1] = 0.

        u = np.vstack((u1.reshape(-1,1), u2.reshape(-1,1)))
        return u


def main():
    env = Env()
    env.logger = logging.Logger()
    x = env.reset()
    show_interval = 100
    n_step = 0

    while True:
        action1 = 0.5 * np.vstack((-1/np.sqrt(2), 0, -1/np.sqrt(2)-4))
        action2 = 0.5 * np.vstack((1/np.sqrt(2), 0, -1/np.sqrt(2)-4))
        action = np.vstack((action1, action2))
        xn, r, done = env.step(action)
        if done:
            break
        if n_step % show_interval == 0:
            pos = xn[0:3]
            vel = xn[3:6]
            q1 = xn[6:9]
            w1 = xn[9:12]
            q2 = xn[12:15]
            w2 = xn[15:18]
            x1 = pos - 0.5 * q1
            x2 = pos - 0.5 * q2
            distance1 = np.linalg.norm(x1 - pos)
            distance2 = np.linalg.norm(x2 - pos)
            print("payload pos.: {}, \
                    quad1 pos.: {}, \
                    quad2 pos.: {}, \
                  distance1 : {}, \
                  distance2 : {}".format(pos, x1, x2, distance1, distance2))
        n_step += 1

    env.close()
    print("no error")

if __name__ == "__main__":
    main()








