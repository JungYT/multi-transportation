import numpy as np
import os
from datetime import datetime
from multiprocessing import Pool

from fym.core import BaseEnv, BaseSystem
import fym.core as core
import fym.logging as logging


class Test(BaseEnv):
    def __init__(self):
        super().__init__()
        self.pos = BaseSystem(np.array([1]))
        self.vel = BaseSystem(np.array([0]))

    def set_dot(self, u):
        pos = self.pos.state
        vel = self.pos.state
        self.pos.dot = vel
        self.vel.dot = -a * vel - b * pos

    def reset(self):
        super().reset()

    def step(self):
        *_, done = self.update()
        info = {
            'pos': self.pos.state,
            'vel': self.vel.state,
        }
        return done, info


def main():
    env = Test()
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

def run_main(parameters):
    for k, v in parameters.items():
        exec('{key} = {value}'.format(key=k, value=v))
    path = os.path.join('log', datetime.today().strftime('%Y%m%d-%H%M%S'), f"{num_parallel}")
    train_path = os.path.join(path, 'train')
    parameter_logger = logging.Logger(
        log_dir=path, file_name="parameters.h5"
    )
    parameter_logger.record(**parameters)
    parameter_logger.close()

    main()

if __name__ == "__main__":
    p1 = {'a': 1, 'b': 2, 'num_parallel':0}
    p2 = {'a': 2, 'b': 3, 'num_parallel':1}
    param = [p1, p2]
    pool = Pool()
    pool.map(run_main, param)

