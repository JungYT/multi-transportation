import numpy as np
import fym.logging as logging
import fym.plotting as plotting
import matplotlib.pyplot as plt
import os
import conf
import fym.utils.rot as rot

# import keyboard

# Data load
past = -1
loglist = sorted(os.listdir("./log"))
load_path = os.path.join("log", loglist[past], "data.h5")
data = logging.load(load_path)
save_path = os.path.join("log", loglist[past])

# Data arrangement
for key in data["state"].keys():
    data[key] = np.squeeze(data["state"][key])

data["euler"] = []
for dcm in data["dcm"]:
    data["euler"].append(rot.quat2angle(rot.dcm2quat(dcm)))
data["euler"] = np.array(data["euler"])


# Setting
draw_dict = {
    "position": {  # figure name
        "projection": "2d",  # 2d or 3d
        "plot": [["time", "pos"]],  # if 2d, [[x, y]], if 3d, [pos]
        "type": None,  # none or scatter
        "label": None,  # legend
        "c": "b",  # color
        "alpha": None,  # transparent
        "xlabel": "time [s]",
        "ylabel": ["x [m]", "y [m]", "z [m]"],
        "xlim": None,
        "ylim": None,
        "axis": None,  # none or equal
        "grid": True,
    },
    "velocity": {
        "projection": "2d",  # 2d or 3d
        "plot": [["time", "vel"]],  # if 2d, [x, y], if 3d, [pos]
        "type": None,  # none or scatter
        "label": None,  # legend
        "c": "b",  # color
        "alpha": None,  # transparent
        "xlabel": "time [s]",
        "ylabel": ["$v_x$ [m/s]", "$v_y$ [m/s]", "$v_z$ [m/s]"],
        "xlim": None,
        "ylim": None,
        "axis": None,  # none or equal
        "grid": True,
    },
    "$\omega$": {
        "projection": "2d",  # 2d or 3d
        "plot": [["time", "omega"]],  # if 2d, [x, y], if 3d, [pos]
        "type": None,  # none or scatter
        "label": None,  # legend
        "c": "b",  # color
        "alpha": None,  # transparent
        "xlabel": "time [s]",
        "ylabel": ["x-dir [deg/s]", "y-dir [deg/s]", "z-dir [deg/s]"],
        "xlim": None,
        "ylim": None,
        "axis": None,  # none or equal
        "grid": True,
    },
    "euler": {
        "projection": "2d",  # 2d or 3d
        "plot": [["time", "euler"]],  # if 2d, [x, y], if 3d, [pos]
        "type": None,  # none or scatter
        "label": None,  # legend
        "c": "b",  # color
        "alpha": None,  # transparent
        "xlabel": "time [s]",
        "ylabel": ["roll [deg]", "pitch [deg]", "yaw [deg]"],
        "xlim": None,
        "ylim": None,
        "axis": None,  # none or equal
        "grid": True,
    },
    "trajectory": {
        "projection": "3d",
        "plot": ["pos"],
        "type": None,  # none or scatter
        "label": None,  # legend
        "c": "b",  # color
        "alpha": None,  # transparent
        "xlabel": "x [m]",
        "ylabel": "y [m]",
        "zlabel": "z [m]",
        "xlim": None,
        "ylim": None,
        "zlim": None,
        "axis": None,  # none or equal
    },
    "input": {  # figure name
        "projection": "2d",  # 2d or 3d
        "plot": [["time", "input"]],  # if 2d, [[x, y]], if 3d, [pos]
        "type": None,  # none or scatter
        "label": None,  # legend
        "c": "b",  # color
        "alpha": None,  # transparent
        "xlabel": "time [s]",
        "ylabel": ["f1 [N]", "f2 [N]", "f3 [N]", "f4 [N]"],
        "xlim": None,
        "ylim": None,
        "axis": None,  # none or equal
        "grid": True,
    },
}

weight_dict = {
    "omega": 180 / np.pi * np.ones(3),
    "euler": 180 / np.pi * np.ones(3),
}

option = {
    "savefig": {
        "onoff": True,
        "dpi": 150,  # resolution
        "transparent": False,
        "bbox_inches": "tight",  # None or tight,
        "format": None,  # file format. png(default), pdf, svg, ...
    },
    "showfig": {
        "onoff": True,
        "showkey": [],
    },
}
# Plot
plotting.plot(data, draw_dict, weight_dict, save_dir=save_path, option=option)

## Close window
# if "showfig" in option and "onoff" in option["showfig"] and option["showfig"]\
#    ["onoff"] is True:
#    while True:
#        if keyboard.is_pressed('q'):
#            plt.close('all')
#            break
