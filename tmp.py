import numpy as np
import random
from types import SimpleNamespace as SN
from pathlib import Path

from tqdm import tqdm
from datetime import datetime
from collections import deque
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import fym.logging as logging
from fym.utils import rot
from dynamics import MultiQuadSlungLoad
from utils import draw_plot, Animator, split_int, compare_animation

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def load_config():
    cfg = SN()
    cfg.epi_train = 2
    cfg.epi_eval = 1
    cfg.dt = 0.1
    cfg.max_t = 2.
    cfg.solver = 'odeint'
    cfg.ode_step_len = 10
    cfg.dir = Path('log', datetime.today().strftime('%Y%m%d-%H%M%S'))

    cfg.animation = SN()
    cfg.animation.quad_size = 1.5
    cfg.animation.rotor_size = 0.8
    cfg.animation.view_angle = [None, None]

    cfg.quad = SN()
    cfg.quad.num = 3
    cfg.quad.mass = 0.755
    cfg.quad.J = np.diag([0.0820, 0.0845, 0.1377])
    cfg.quad.iscollision = 0.5

    cfg.load = SN()
    cfg.load.pos_bound = [[-5, 5], [-5, 5], [1, 10]]
    cfg.load.att_bound = [
        [-np.pi/4, np.pi/4],
        [-np.pi/4, np.pi/4],
        [-np.pi, np.pi]
    ]
    cfg.load.pos_init = np.vstack((0., 0., 3.))
    cfg.load.dcm_init = rot.angle2dcm(0., 0., 0.).T
    cfg.load.mass = 4
    cfg.load.J = np.diag([0.2, 0.2, 0.2])
    cfg.load.cg = np.vstack((0., 0., -3.))
    cfg.load.size = 3.

    cfg.link = SN()
    cfg.link.len = cfg.quad.num * [3.]
    cfg.link.anchor = [
        np.vstack((
            cfg.load.size * np.cos(i*2*np.pi/cfg.quad.num),
            cfg.load.size * np.sin(i*2*np.pi/cfg.quad.num),
            0
        )) - cfg.load.cg for i in range(cfg.quad.num)
    ]
    cfg.link.uvec_bound = [[-np.pi, np.pi], [0, np.pi/4]]

    cfg.controller = SN()
    cfg.controller.Ke = 20.
    cfg.controller.Ks = 80.
    cfg.controller.chattering_bound = 0.5
    cfg.controller.unc_max = 0.1

    cfg.ddpg = SN()
    cfg.ddpg.P = np.diag([1., 1., 1., 1., 1., 1.])

    cfg.noise = SN()
    cfg.noise.rho = 0.15
    cfg.noise.mu = 0.
    cfg.noise.sigma = 0.2
    cfg.noise.size = 9

    return cfg


def train_MultiQuadSlungLoad(cfg):
    load_pos_des = np.vstack((0., 0., 5.))
    load_att_des = np.vstack((0., 0., 0.))
    env = MultiQuadSlungLoad(cfg)
    x = env.reset(load_pos_des, load_att_des)
    psi_des = 3*[0]
    action = np.hstack(3*[24., 0., 0.])
    while True:
        xn, r, done = env.step(load_pos_des, load_att_des, action, psi_des)
        if done:
            break
    env.close()


def eval_MultiQuadSlungLoad(cfg, path_data):
    load_pos_des = np.vstack((0., 0., 5.))
    load_att_des = np.vstack((0., 0., 0.))
    env = MultiQuadSlungLoad(cfg)
    env.logger = logging.Logger(path_data)
    env.logger.set_info(cfg=cfg)
    x = env.reset(load_pos_des, load_att_des, fixed_init=True)
    psi_des = 3*[0]
    action = np.hstack(3*[24., 0., 0.])
    while True:
        xn, r, done = env.step(load_pos_des, load_att_des, action, psi_des)
        if done:
            break
    env.close()


def main():
    cfg = load_config()

    for epi_num in tqdm(range(cfg.epi_train)):
        train_MultiQuadSlungLoad(cfg)
        if (epi_num+1) % cfg.epi_eval == 0:
            eval_MultiQuadSlungLoad(
                cfg,
                Path(cfg.dir, f"env_data/{epi_num+1:05d}_epi.h5")
            )
            Path(cfg.dir, f"fig_{epi_num+1:05d}_epi").mkdir(
                parents=True,
                exist_ok=True
            )
            draw_plot(
                Path(cfg.dir, f"env_data/{epi_num+1:05d}_epi.h5"),
                Path(cfg.dir, f"fig_{epi_num+1:05d}_epi")
            )


if __name__ == "__main__":
    main()

    past = -1
    dir_list = [x for x in Path('log').glob("*")]
    file_list = [x for x in Path(dir_list[past], 'env_data').glob("*")]
    compare_animation(file_list, dir_list[past])





