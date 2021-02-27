import fym.utils.linearization as lin
import numpy as np


class temp:
    def __init__(self):
        self.a = 1

    def func(self, pos, vel, dcm, control):
        dpos = pos
        dvel = pos * vel
        ddcm = pos * dcm + control
        return dpos, dvel, ddcm


tmp = temp()


def func2(state, control):
    pos, vel, dcm = state
    return np.vstack(tmp.func(pos, vel, dcm, control))


jacob = lin.jacob_analytic(func2)
jacob2 = lin.jacob_analytic(func2, i=1)
pos = 1
vel = 2
dcm = 3
control = 4
A = jacob([pos, vel, dcm], control)
B = jacob2([pos, vel, dcm], control)
