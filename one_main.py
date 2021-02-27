import numpy as np
from fym.core import BaseEnv, BaseSystem
from fym.models.quadrotor import Quadrotor
import fym.logging as logging
import conf
import fym.utils.rot as rot
import fym.utils.linearization as lin
import fym.agents.LQR as lqr

x0 = np.array([0, 0, 0])
quad = Quadrotor(x0, x0, conf.QUADROTOR["dcm"], conf.QUADROTOR["omega"])


class Quad(BaseEnv):
    def __init__(self):
        super().__init__()
        # System
        self.x1 = BaseSystem()
        self.x2 = BaseSystem()

    def step(self):
        *_, done = self.update()
        t = self.clock.get()
        u = self.test_controller()
        info = {
            "input": u,
        }
        self.logger.record(**info)

        return done

    def set_dot(self, t):
        x = self.state
        u = -K.dot(x)
        self.dot = self.deriv(x, u)

        self.x1.dot = -x1 + u[0]
        self.x2.dot = x1 + u[1]

    def test_controller(self):
        u = [5, 10, 5, 10]
        return u

    def LQR(self):
        def func_temp(state, control):
            pos, vel, dcm, omega = state
            return np.vstack(quad.deriv(pos, vel, dcm, omega, control))

        jacob_A = lin.jacob_analytic(func_temp)
        jacob_B = lin.jacob_analytic(func_temp, i=1)

    def actuator_input(self, force_moment):
        mat = np.array(
            [
                [1, 1, 1, 1],
                [0, -quad.d, 0, quad.d],
                [quad.d, 0, -quad.d, 0],
                [-c, c, -c, c],
            ]
        )
        invmat = np.linalg.inv(mat)
        actuator = invat.dot(force_moment)
        return actuator


class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.01, max_t=10)
        self.quad = Quad()

    def step(self):
        *_, done = self.update()
        t = self.clock.get()
        u = self.test_controller()
        info = {
            "input": u,
        }
        self.logger.record(**info)

        return done

    def set_dot(self, t, action):
        self.quad.set_dot(t, X0, Omega0)


def main():
    env = Env()
    env.logger = logging.Logger()
    env.reset()

    while True:
        env.render()
        done = env.step()
        if done:
            break

    env.close()
    print("running")


if __name__ == "__main__":
    main()
