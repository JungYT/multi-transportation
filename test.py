import numpy as np
import os
from datetime import datetime
from multiprocessing import Pool

from fym.core import BaseEnv, BaseSystem
import fym.core as core
import fym.logging as logging


class Test(BaseEnv):
    def __init__(self, a, b):
        super().__init__()
        self.pos = BaseSystem(np.array([1]))
        self.vel = BaseSystem(np.array([0]))
        self.a = a
        self.b = b

    def set_dot(self, u):
        pos = self.pos.state
        vel = self.pos.state
        self.pos.dot = vel
        self.vel.dot = -self.a * vel - self.b * pos + u

    def reset(self):
        super().reset()

    def step(self):
        *_, done = self.update(u=0.5)
        info = {
            'pos': self.pos.state,
            'vel': self.vel.state,
        }
        return done, info


def main(a, b, num_parallel):
    path = os.path.join('log', datetime.today().strftime('%Y%m%d-%H%M%S'), f"{num_parallel}")
    train_path = os.path.join(path, 'train')
    parameter_logger = logging.Logger(
        log_dir=path, file_name="parameters.h5"
    )
    parameters  = {
        'a': a,
        'b': b,
        'num_parallel': num_parallel,
    }
    parameter_logger.record(**parameters)
    parameter_logger.close()

    env = Test(a, b)
    env.reset()
    logger = logging.Logger(
        log_dir=train_path, file_name=f"data_{num_parallel:05d}.h5"
    )
    while True:
        done, info = env.step()
        logger.record(**info)
        if done:
            break

    logger.close()
    env.close()


if __name__ == "__main__":
    param1 = [1, 1, 0]
    param2 = [2, 1, 1]
    param = [param1, param2]
    map(main, param)

