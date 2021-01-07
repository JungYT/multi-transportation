import numpy as np
import fym.utils.rot as rot


PARAM = {
            "step" : 0.01,
            "final" : 10,
            "g" : 9.81 * np.vstack((0, 0, 1)),
            "rho" : [np.vstack((1, 1, -1)), np.vstack((1, -1, -1)),
                     np.vstack((-1, 1, -1)), np.vstack((-1, -1, -1))],
}

PAYLOAD = {
            "pos" : np.vstack((0, 0, 0)),
            "vel" : np.vstack((0, 0, 0)),
            "dcm" : rot.angle2dcm(0, 0, 0),
            "omega" : np.vstack((0, 0, 0)),
            "m" : 10        # kg
            "J" : np.diag([0.5, 0.5, 0.5]),
}

LINK = {
            "q" : np.vstack((0, 0, 1)),
            "omega" : np.vstack((0, 0, 0)),
            "l" : 0.3,
            "rho" : np.vstack((0, 0, 0)),
}

QUADROTOR = {
            "omega" : np.vstack((0, 0, 0)),
            "dcm" : rot.angle2dcm(0, 0, 0),
            "J" : np.diag([0.0820, 0.0845, 0.1377]),
            "m" : 4.34,
            "d" : 0.0315,
            "c" : 8.004e-4,
}


if __name__ == "__main__":
    pass
