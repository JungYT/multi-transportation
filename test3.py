import numpy as np
from fym.core import BaseEnv, BaseSystem
import fym.core as core
import fym.logging as logging

x0 = [0, 10, 20]


class Model(BaseEnv):
    def __init__(self, pos):
        super().__init__()
        self.pos = BaseSystem(pos)
        self.vel = BaseSystem(pos)
        self.m = 1.0

    def set_dot(self, action):
        pos = self.state

        self.pos.dot = action
        self.vel.dot = 0


class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.1, max_t=1)

        self.model = Model(x0[0])
        self.sys = core.Sequential(**{f"sys_{i:02d}": Model(x0[i]) for i in range(3)})

    def step(self):
        *_, done = self.update()
        print(self.state)

        return done

    def set_dot(self, t):
        self.model.set_dot(0)
        for system in self.sys.systems:
            pos, vel = system.state
            system.set_dot(pos)


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
