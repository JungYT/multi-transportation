import numpy as np
from fym.core import BaseEnv, BaseSystem
import fym.core as core
import fym.logging as logging
import conf
import fym.utils.rot as rot
import fym.utils.linearization as lin
import fym.agents.LQR as lqr


g = conf.PARAM["g"]
n = len(conf.PARAM["rho"])

class Payload(BaseEnv):
    m = conf.PAYLOAD["m"]
    J = conf.PAYLOAD["J"]

    def __init__(self, pos, vel, omega, dcm):
        super().__init__()
        self.pos = BaseSystem(pos)
        self.vel = BaseSystem(vel)
        self.omega = BaseSystem(omega)
        self.dcm = BaseSystem(dcm)

    def set_dot(self, A, B, C, Mq, S1, S2):
        pos, vel, omega, dcm = self.state

        self.pos.dot = vel
        self.vel.dot = np.linalg.inv(C).dot(Mq.dot(g) + S1 + B.dot(S2))
        self.omega.dot = A.dot(self.vel.dot) + B
        self.dcm.dot = dcm.dot(hat(omega))


class Link(BaseEnv):

    def __init__(self, q, omega, rho):
        super().__init__()
        self.q = BaseSystem(q)
        self.omega = BaseSystem(omega)
        self.rho = rho
        self.l = conf.LINK["l"]

    def set_dot(self, A, B, C, Mq, S1, S2, R0, omega0, u):
        q, omega = self.state
        D = R0.dot(hat(omega0).dot(hat(omega0).dot(self.rho))) \
            - g - u / conf.QUADROTOR["m"]
        ddx0 = np.linalg.inv(C).dot(Mq.dot(g) + S1 + B.dot(S2))
        domega0 = A.dot(ddx0) + B

        self.q.dot = hat(omega).dot(q)
        self.omega.dot = hat(q).dot(ddx0 + D
                                    - R0.dot(hat(self.rho)).dot(domega0)) \
                                    / self.l


class Quadrotor(BaseEnv):
    J = conf.QUADROTOR["J"]

    def __init__(self, omega, dcm):
        super().__init__()
        self.omega = BaseSystem(omega)
        self.dcm = BaseSystem(dcm)

    def set_dot(self, M):
        omega, dcm = self.state

        self.omgea.dot = np.linalg.inv(self.J).dot(M
                                    - np.cross(omega, J.dot(omega), axsis=0))
        self.dcm.dot = dcm.dot(hat(omega))


def hat(v):
    v1, v2, v3 = v.squeeze()
    return np.array([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0]
    ])


class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=conf.PARAM["step"], max_t=conf.PARAM["final"])
        self.payload = Payload(conf.PAYLOAD["pos"], conf.PAYLOAD["vel"],
                               conf.PAYLOAD["omega"], conf.PAYLOAD["dcm"])
        self.link = core.Sequential(**{f"link{i:02d}": Link(conf.LINK["q"],
                                                            conf.LINK["omega"],
                                                            conf.LINK["rho"][i]
                                                            ) for i in range(n)
                                       })
        self.quad =core.Sequential(**{f"quad{i:02d}":
                                      Quadrotor(conf.QUADROTOR["omega"],
                                                conf.QUADROTOR["dcm"])
                                      for i in range(n)})

    def step(self):
        *_, done = self.update()

        info = {
            "input": u,
        }
        self.logger.record(**info)

        return done

    def set_dot(self, t, u):
"""
To Do
calculate A, B, C, Mq, S1, S2, M etc..
get state (ex: R0, omega0 = self.load.state)
design controller
"""
        self.payload.set_dot(A, B, C, Mq, S1, S2)
        for system in self.link.systems:
            system.set_dot(A, B, C, Mq, S1, S2, R0, omega0, u)
        for system in self.quad.systems:
            system.set_dot(M)


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
    print('running')


if __name__ == "__main__":
    main()
