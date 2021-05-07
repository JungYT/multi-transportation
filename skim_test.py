"""
Reference:
https://ocw.mit.edu/courses/aeronautics-and-astronautics/16-07-dynamics-fall-2009/lecture-notes/MIT16_07F09_Lec05.pdf
"""
import numpy as np
from types import SimpleNamespace as SN
from pathlib import Path
from matplotlib import pyplot as plt
from celluloid import Camera

from fym.core import BaseEnv, BaseSystem, Sequential
import fym.logging

import logs

cfg = SN()


def load_config():
    cfg.eps = 1e-8
    cfg.g = 9.81
    cfg.gvec = cfg.g * np.vstack((0, 0, -1))

    cfg.env = SN()
    cfg.env.solver = "rk4"
    cfg.env.dt = 0.1
    cfg.env.max_t = 10
    cfg.env.ode_step_len = 10

    cfg.load = SN()
    cfg.load.mass = 10

    cfg.string = SN()
    cfg.string.length = 1

    cfg.quad = SN()
    cfg.quad.num = 3
    cfg.quad.mass = 5
    cfg.quad.Tmax = 2 * cfg.quad.mass * cfg.g


class Quadrotor(BaseEnv):
    name = "quad"

    def __init__(self, pos=np.zeros((3, 1)), vel=np.zeros((3, 1))):
        super().__init__()
        self.pos = BaseSystem(pos)
        self.vel = BaseSystem(vel)

    def set_dot(self, tvec):
        pos, vel = self.observe_list()
        T = np.linalg.norm(tvec)
        n = tvec / (T + cfg.eps)
        tvec = min(T, cfg.quad.Tmax) * n
        self.pos.dot = vel
        self.vel.dot = tvec / cfg.quad.mass + cfg.gvec

    def get_thrust(self, t):
        return np.vstack((0, 0, 100))


class Load(BaseEnv):
    name = "load"

    def __init__(self, pos=np.zeros((3, 1)), vel=np.zeros((3, 1))):
        super().__init__()
        self.pos = BaseSystem(pos)
        self.vel = BaseSystem(vel)

    def set_dot(self, force):
        pos, vel = self.observe_list()
        self.pos.dot = vel
        self.vel.dot = force / cfg.load.mass + cfg.gvec


class String:
    def __init__(self, load, quad):
        self.load = load
        self.quad = quad
        self.quad.string = self

    def length(self, X=None, x=None):
        if X is None and x is None:
            dx = self.load.pos.state - self.quad.pos.state
        else:
            dx = X - x
        return np.linalg.norm(dx)

    def tension(self):
        return self._tension


def fixed_deposit(length):
    deposit = np.stack([
        np.vstack((0.5, 0, 0)),
        np.vstack((-0.5, 0, 0)),
        np.vstack((0, 0.5, 0)),
    ])
    # deposit = length * deposit / np.linalg.norm(deposit, axis=1)[:, None]
    return deposit


def random_equi_len_deposit(length, num):
    strings = np.random.rand(num, 3, 1)
    deposit = length * strings / np.linalg.norm(strings, axis=1)[:, None]
    return deposit


class Env(BaseEnv):
    I = np.eye(3)
    Z = np.zeros((3, 3))

    def __init__(self):
        super().__init__(**vars(cfg.env))
        self.load = Load()

        deposit = fixed_deposit(cfg.string.length)
        self.quads = Sequential(*[Quadrotor(pos=p) for p in deposit])

        self.strings = [String(self.load, quad) for quad in self.quads.systems]

    def step(self):
        info = self.observation()
        *_, done = self.update()
        return info, done

    def observation(self):
        t = self.clock.get()
        X = self.load.pos.state
        xs = []
        for quad in self.quads.systems:
            xs.append(quad.pos.state)
        xs = np.stack(xs)
        active_idx = self.get_active_index(X, xs)
        return dict(t=t, load_pos=X, quads_pos=xs, active_idx=active_idx)

    def get_active_index(self, X, xs):
        return [np.linalg.norm(X - x) >= 1 for x in xs]

    def set_dot(self, t):
        X = self.load.pos.state
        Xdot = self.load.vel.state

        # Get active quads
        active_quads = [q for q in self.quads.systems
                        if q.string.length() >= cfg.string.length]
        inactive_quads = [q for q in self.quads.systems
                          if q.string.length() < cfg.string.length]

        n = len(active_quads)
        M = cfg.load.mass
        I = self.I
        Z = self.Z

        LHS = [np.vstack([M * I, *([Z] * n), *([I] * n)])]
        RHS = [M * cfg.gvec]

        for i, quad in enumerate(active_quads):
            x = quad.pos.state
            xdot = quad.vel.state

            l = quad.string.length()
            m = cfg.quad.mass

            dx = x - X
            dxdot = xdot - Xdot

            phi = np.arcsin(dx[2, 0] / l)
            theta = np.arctan2(dx[1, 0], dx[0, 0])
            cphi = np.cos(phi)
            sphi = np.sin(phi)
            ctheta = np.cos(theta)
            stheta = np.sin(theta)

            er = np.vstack((ctheta * cphi, stheta * cphi, sphi))
            etheta = np.vstack((-stheta, ctheta, 0))
            ephi = np.vstack((-ctheta * sphi, -stheta * sphi, cphi))

            quad.er = er

            # Reset x
            quad.pos.state = X + l * er

            E1 = np.hstack((er, etheta, ephi))
            E2 = np.diag((1, l * cphi, l))
            E = E1.dot(E2)
            dotvec = np.linalg.inv(E).dot(dxdot)
            thetadot, phidot = dotvec[1:, 0]

            # Reset xdot
            quad.vel.state = Xdot + E.dot(np.vstack((0, dotvec[1:])))

            # phidot = dxdot[-1, 0] / (l * cphi)
            # if np.isclose(ctheta, 0):
            #     thetadot = (
            #         (dxdot[1, 0] + l * phidot * ctheta * sphi)
            #         / (- l * cphi * stheta))
            # else:
            #     thetadot = (
            #         (dxdot[1, 0] + l * phidot * stheta * sphi)
            #         / (l * cphi * ctheta))

            column1 = np.vstack([
                Z,
                *([Z] * i), m * I, *([Z] * (n - i - 1)),
                *([Z] * i), -I, *([Z] * (n - i - 1)),
            ])

            matrix1 = np.hstack([er, np.zeros((3, 2))])
            matrix2 = np.hstack([
                np.zeros((3, 1)), l * cphi * etheta, l * ephi])
            column2 = np.vstack([
                - matrix1,
                *([Z] * i), matrix1, *([Z] * (n - i - 1)),
                *([Z] * i), matrix2, *([Z] * (n - i - 1)),
            ])

            LHS.append(column1)
            LHS.append(column2)

            thrust = quad.get_thrust(t)
            rhs1 = m * cfg.gvec + thrust
            rhs2 = l * (
                (thetadot**2 * cphi**2 + phidot**2) * er
                + 2 * thetadot * phidot * sphi * etheta
                - phidot**2 * sphi * cphi * ephi
            )

            RHS.append(rhs1)
            RHS.append(rhs2)

        if active_quads == []:
            force = 0
            self.load.set_dot(force)
        else:
            LHS = np.hstack(LHS)
            RHS = np.vstack(RHS)
            sol = np.linalg.inv(LHS).dot(RHS)
            load_dot, *dots = np.split(sol, np.cumsum([3, *([6] * (n - 1))]))

            force = M * load_dot
            self.load.set_dot(force)

            for quad, dot in zip(active_quads, dots):
                # t = dot[3, 0]
                # T = t * quad.er
                # force = quad.get_thrust(t) - T quad.set_dot(force)
                quad_dot = dot[:3]
                force = cfg.quad.mass * (quad_dot - cfg.gvec)
                quad.set_dot(force)

        for quad in inactive_quads:
            force = quad.get_thrust(t)
            quad.set_dot(force)


def run():
    env = Env()

    logger = fym.logging.Logger(Path(cfg.datadir, "env.h5"))
    logger.set_info(cfg=cfg)

    env.reset()

    while True:
        env.render()
        info, done = env.step()
        logger.record(**info)

        if done:
            break

    env.close()
    logger.close()


def exp1():
    # Init the experiment
    cfg.expdir = Path("data/exp1")
    logs.set_logger(cfg.expdir, "train.log")

    # ------ Data 001 ------ #
    load_config()  # Load the experiment default configuration
    cfg.datadir = Path(cfg.expdir, "data-001")

    # ------ Train ------ #
    run()


def exp1_plot():
    # ------ Exp Setup ------ #
    expdir = Path("data", "exp1")

    # ------ Data 001 ------ #
    datadir = Path(expdir, "data-001")
    data, info = fym.logging.load(Path(datadir, "env.h5"), with_info=True)
    cfg = info["cfg"]

    # ------ Animation ------ #
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    camera = Camera(fig)

    for i, (load_pos, quads_pos, active_idx) in enumerate(zip(
            data["load_pos"].squeeze(),
            data["quads_pos"].squeeze(), data["active_idx"])):

        # Load
        ax.plot3D(*load_pos, alpha=0.6, c="k", marker="o")

        for quad_pos in quads_pos:
            # Strings
            ax.plot3D(*zip(load_pos, quad_pos), alpha=0.6, c="k")

            # Quadrotors
            ax.plot3D(*quad_pos, alpha=0.6, c="r", marker="X")

        # axis limit
        ax.set_xlim3d(-3, 3)
        ax.set_ylim3d(-3, 3)
        ax.set_zlim3d(-1, 5)

        camera.snap()

    ani = camera.animate(interval=1000*cfg.env.dt, blit=True)
    ani.save(Path(datadir, "ani.mp4"))
    plt.close("all")


if __name__ == "__main__":
    exp1()
    exp1_plot()
