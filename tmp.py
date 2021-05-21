import numpy as np
import random
from types import SimpleNamespace as SN
from pathN import Path

from tqdm import tqdm
from datetime import datetime
from collections import deque
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from fym.core import BaseEnv, BaseSystem
import fym.core as core
import fym.logging as logging
from fym.utils import rot

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

cfg = SN()
def load_config():
    cfg.epi_train = 1
    cfg.epi_eval = 1
    cfg.dt = 0.1
    cfg.max_t = 1.
    cfg.ode_step = 10
    cfg.dir = Path('log', datetime.today().strftime('%Y%m%d-%H%M%S'))





