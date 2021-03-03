import numpy as np
from fym.core import BaseEnv, BaseSystem
import fym.core as core
import fym.logging as logging
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from fym.agents import PID
import os
from matplotlib import pyplot as plt
import time
import timeit

torch.manual_seed(0)
np.random.seed(0)

g = 9.81 * np.vstack((0.0, 0.0, 1.0))


class Payload(BaseEnv):
    def __init__(self):
        super().__init__()
        self.pos = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.vel = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.m = config["env"]["payload.m"]

    def set_dot(self, xddot):
        vel = self.vel.state

        self.pos.dot = vel
        self.vel.dot = xddot


class Link(BaseEnv):
    def __init__(self):
        super().__init__()
        self.q = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.w = BaseSystem(np.vstack((0.0, 0.0, 0.0)))
        self.m = config["env"]["quad.m"]
        self.l = config["env"]["quad.l"]

    def set_dot(self, u, xddot):
        q = self.q.state
        w = self.w.state
        self.q.dot = hat(w).dot(q)
        self.w.dot = hat(q).dot(xddot - g - u / self.m) / self.l


class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=config["env"]["dt"], max_t=config["env"]["max_t"])
        self.payload = Payload()
        self.links = core.Sequential(
            **{f"link{i:02d}": Link() for i in range(2)}
        )
        # controller
        self.n_quad = config["env"]["n_quad"]
        self.u_set = deque(maxlen=self.n_quad)
        self.K = config["env"]["K"]
        self.epsilon = config["env"]["epsilon"]
        self.input_saturation = config["env"]["input_saturation"]
        self.bubble = config["env"]["bubble"]
        self.e_set = deque(maxlen=self.n_quad)

    def reset(self, fixed_init=False):
        super().reset()
        if fixed_init:
            phi = np.pi / 4
            self.payload.pos.state = np.vstack((0.0, 0.0, -1.0))
            self.links.link00.q.state = np.vstack(
                (np.sin(phi), 0.0, np.cos(phi))
            )
            self.links.link01.q.state = np.vstack(
                (-np.sin(phi), 0.0, np.cos(phi))
            )
        else:
            self.payload.pos.state = np.vstack(
                (
                    np.random.uniform(low=-2, high=2),
                    0.0,
                    np.random.uniform(low=-1, high=-9),
                )
            )
            phi = np.random.uniform(low=0.1, high=np.pi / 2)

            self.links.link00.q.state = np.vstack(
                (np.sin(phi), 0.0, np.cos(phi))
            )
            self.links.link01.q.state = np.vstack(
                (-np.sin(phi), 0.0, np.cos(phi))
            )

        x = self.observation()

        return x

    def step(self, action, reference):
        """
        action_converted = np.vstack((3-np.cos(np.pi/4), 0, -3-np.sin(np.pi/4),
                                      3+np.cos(np.pi/4), 0, -3-np.sin(np.pi/4)))
        """
        #action_converted = self.action_convert_to_pos(action)
        #u = self.controller(action_converted)
        u = self.controller(action)
        *_, done = self.update(u=u)
        quad_pos = self.convert_to_pos()
        if (
            self.payload.pos.state[2] > 0
            or np.linalg.norm(quad_pos[0] - quad_pos[1]) < self.bubble
        ):
            done = True
        r = self.reward(action, reference)

        #info = {"action_converted": action_converted}
        info = {}
        self.logger.record(**info)

        x = self.observation()

        return x, r, done, quad_pos, self.payload.pos.state, np.array(self.e_set)

    def set_dot(self, t, u):
        mt = self.payload.m + self.links.link00.m + self.links.link01.m

        Mq = mt * np.eye(3)
        S = np.vstack((0.0, 0.0, 0.0))
        i = 0
        for system in self.links.systems:
            q = system.q.state
            w = system.w.state
            Mq += system.m * hat(q).dot(hat(q))
            S += system.m * (
                hat(q).dot(hat(q)).dot(g)
                - system.l * (np.transpose(w).dot(w)) * q
            ) + (np.eye(3) + hat(q).dot(hat(q))).dot(u[3 * i : 3 * (i + 1)])
            i += 1

        S += mt * g
        xddot = np.linalg.inv(Mq).dot(S)

        self.payload.set_dot(xddot)
        self.links.link00.set_dot(u[0:3], xddot)
        self.links.link01.set_dot(u[3:6], xddot)

    def reward(self, action, reference):
        load_pos = self.payload.pos.state
        quad_pos = self.convert_to_pos()
        if load_pos[2] > 0:
            r = -np.array([250])
        elif np.linalg.norm(quad_pos[0] - quad_pos[1]) < self.bubble:
            r = -np.array([250])
        else:
            e = load_pos - np.reshape(reference, (-1, 1))
            r = -16 * (e[0] ** 2) - (e[2] ** 2) - 0.01 * (action[2] ** 2)

        return r

    def controller(self, action):
        i = 0
        m0 = self.payload.m
        for system in self.links.systems:
            q = system.q.state
            w = system.w.state
            m = system.m
            l = system.l
            x0 = self.payload.pos.state
            x0dot = self.payload.vel.state
            e = x0 - l * q - action[3 * i : 3 * (i + 1)]
            vel = x0dot - l * hat(w).dot(q)
            delta_max = 0.5 * m0 * 9.81 * np.vstack((1, 1, 1))
            s = self.K * e + vel
            s_clip = np.clip(s / self.epsilon, -1, 1)

            mu = -self.K * vel - (delta_max + np.vstack((1, 1, 1))) * s_clip
            u = m * (-g - m0 / m * (np.transpose(g).dot(q)) * q + mu)

            u_norm = np.linalg.norm(u)
            if u_norm > self.input_saturation:
                u = self.input_saturation * u / u_norm

            self.u_set.append(u)
            i += 1
            self.e_set.append(np.linalg.norm(e))
        u_return = np.vstack((self.u_set[0], self.u_set[1]))

        return u_return

    def convert_to_pos(self):
        quad_pos = []
        for system in self.links.systems:
            q = system.q.state
            w = system.w.state
            m = system.m
            l = system.l
            x0 = self.payload.pos.state
            x = x0 - l * q
            quad_pos.append(x)
        return quad_pos

    def observation(self):
        load_pos = np.delete(self.payload.pos.state, 1)
        load_vel = np.delete(self.payload.vel.state, 1)
        theta = np.arctan2(
            self.links.link00.q.state[0], self.links.link00.q.state[2]
        )
        theta_dot = self.links.link00.w.state[1]
        x_converted = np.hstack((load_pos, load_vel, theta, theta_dot))
        return x_converted

    def action_convert_to_pos(self, u):
        load_pos = self.payload.pos.state
        u_converted = []
        for i in range(self.n_quad):
            R = self.links.link00.l
            phi = u[2]
            pos_x = ((-1) ** (i+1)) * R * np.sin(phi)
            pos_y = 0
            pos_z = -1 * R * np.cos(phi)
            pos = np.vstack((pos_x, pos_y, pos_z))
            desired_pos = load_pos + np.vstack((u[0], 0, u[1])) + pos
            u_converted.append(desired_pos)
        u_converted = np.vstack(u_converted)
        return u_converted


class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()

        """
        self.lin_load = nn.Linear(4, 16)
        for i in range(1, n_quad+1):
            setattr(self, f"lin_quad{i}", nn.Linear(2, 16))
        """

        n_quad = config["env"]["n_quad"]
        action_size = config["ddpg"]["action_size"]
        self.reference_size = len(config["simulation"]["reference"])
        self.action_pos_bound = config["ddpg"]["action_pos_bound"]
        self.lin1 = nn.Linear(6, 128)
        self.lin2 = nn.Linear(128 + self.reference_size, 64)
        self.lin3 = nn.Linear(64, 32)
        self.lin4 = nn.Linear(32, 16)
        self.lin5 = nn.Linear(16, action_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.l = config["env"]["quad.l"]

    def forward(self, xr):
        x, r = xr
        x = self.relu(self.lin1(x.view(-1, 6)))
        x = self.relu(self.lin2(
            torch.cat([x, r.view(-1, self.reference_size)], 1)
        ))
        x = self.relu(self.lin3(x))
        x = self.relu(self.lin4(x))
        x = self.tanh(self.lin5(x))
        # scaling
        x = x * torch.Tensor(
            [self.action_pos_bound, self.action_pos_bound, np.pi / 4]
        ) + torch.Tensor([0, 0, np.pi / 4])

        return x


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()

        n_quad = config["env"]["n_quad"]
        action_size = config["ddpg"]["action_size"]
        self.lin1 = nn.Linear(6, 64)
        self.lin2 = nn.Linear(64 + action_size, 32)
        self.lin3 = nn.Linear(32, 16)
        self.lin4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, xa):
        x, a = xa
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(torch.cat([x, a], 1)))
        x = self.relu(self.lin3(x))
        x = self.lin4(x)

        return x


class DDPG:
    def __init__(self):
        # learning hyper parameters
        self.gamma = config["ddpg"]["gamma"]
        self.actor_lr = config["ddpg"]["actor_lr"]
        self.critic_lr = config["ddpg"]["critic_lr"]
        self.tau = config["ddpg"]["tau"]

        # memory parameters
        self.memory_size = config["ddpg"]["memory_size"]
        self.batch_size = config["ddpg"]["batch_size"]

        # setting
        self.memory = deque(maxlen=self.memory_size)
        self.behavior_actor = ActorNet().float()
        self.behavior_critic = CriticNet().float()
        self.target_actor = ActorNet().float()
        self.target_critic = CriticNet().float()
        self.actor_optim = optim.Adam(
            self.behavior_actor.parameters(), lr=self.actor_lr
        )
        self.critic_optim = optim.Adam(
            self.behavior_critic.parameters(), lr=self.critic_lr
        )
        self.mse = nn.MSELoss()
        hard_update(self.target_actor, self.behavior_actor)
        hard_update(self.target_critic, self.behavior_critic)

    def action(self, x, reference, use="behavior"):
        with torch.no_grad():
            if use == "behavior":
                u = self.behavior_actor(
                    [torch.FloatTensor(x), torch.FloatTensor(reference)]
                )
            else:
                u = self.target_actor(
                    [torch.FloatTensor(x), torch.FloatTensor(reference)]
                )
        return np.array(np.squeeze(u))

    def memorize(self, item):
        self.memory.append(item)

    def sample(self):
        sample = random.sample(self.memory, self.batch_size)
        x, u, r, xn, done, reference = zip(*sample)
        x = torch.FloatTensor(x)
        u = torch.FloatTensor(u)
        r = torch.FloatTensor(r)
        xn = torch.FloatTensor(xn)
        reference = torch.FloatTensor(reference)
        done = torch.FloatTensor(done).view(-1, 1)

        return x, u, r, xn, done, reference

    def train(self):
        x, u, r, xn, done, reference = self.sample()

        with torch.no_grad():
            action = self.target_actor([xn, reference])
            Qn = self.target_critic([xn, action])
            target = r + (1 - done) * self.gamma * Qn

        Q_noise = self.behavior_critic([x, u])

        self.critic_optim.zero_grad()
        critic_loss = self.mse(Q_noise, target)
        critic_loss.backward()
        self.critic_optim.step()

        Q = self.behavior_critic([x, self.behavior_actor([x, reference])])

        self.actor_optim.zero_grad()
        actor_loss = torch.sum(-Q)
        actor_loss.backward()
        self.actor_optim.step()

        soft_update(self.target_actor, self.behavior_actor, self.tau)
        soft_update(self.target_critic, self.behavior_critic, self.tau)


class OrnsteinUhlenbeckNoise:
    def __init__(self, x0=None):
        self.rho = config["noise"]["rho"]
        self.mu = config["noise"]["mu"]
        self.sigma = config["noise"]["sigma"]
        self.dt = config["noise"]["dt"]
        self.x0 = x0
        self.size = config["ddpg"]["action_size"]
        self.reset()

    def reset(self):
        self.x = self.x0 if self.x0 is not None else np.zeros(self.size)

    def sample(self):
        x = (
            self.x
            + self.rho * (self.mu - self.x) * self.dt
            + np.sqrt(self.dt) * self.sigma * np.random.normal(size=self.size)
        )
        self.x = x

        return x


def hat(v):
    v1, v2, v3 = v.squeeze()
    return np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])


def wrap(angle):
    angle_wrap = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle_wrap


def soft_update(target, behavior, tau):
    for t_param, b_param in zip(target.parameters(), behavior.parameters()):
        t_param.data.copy_(t_param.data * (1.0 - tau) + b_param.data * tau)


def hard_update(target, behavior):
    for t_param, b_param in zip(target.parameters(), behavior.parameters()):
        t_param.data.copy_(b_param.data)


config = {
    "ddpg": {
        "gamma": 0.95,
        "actor_lr": 0.0001,
        "critic_lr": 0.001,
        "tau": 0.001,
        "memory_size": 20000,
        "batch_size": 64,
        "action_size": 3,
        "action_pos_bound": 1,
    },
    "env": {
        "payload.m": 1.0,
        "quad.m": 1.0,
        "quad.l": 1.0,
        "max_t": 10,
        "dt": 0.01,
        "n_quad": 2,
        "K": 5,
        "epsilon": 0.1,
        "input_saturation": 50,
        "bubble": 0.2,
    },
    "noise": {"rho": 0.15, "mu": 0, "sigma": 0.2, "dt": 0.1},
    "simulation": {
        "n_epi": 2000,
        "n_test": 20,
        "n_test_epi": 10,
        "animation": False,
        "reference": np.array([0.0, 0.0, -5.0]),
        "fixed_init": True,
        "reach": 0.05,
    },
}

#np.set_printoptions(precision=2)


def main():
    # environment
    env = Env()
    env.logger = logging.Logger()

    # agent parameters
    agent = DDPG()

    # noise parameters
    noise = OrnsteinUhlenbeckNoise()

    # display
    score = 0
    n_data = 0
    score_save = []
    data_save = deque()

    # reference signal
    reference = config["simulation"]["reference"]

    # test variables
    action_bound_min = [-config["ddpg"]["action_pos_bound"],
                        -config["ddpg"]["action_pos_bound"],
                        -np.pi/2]
    action_bound_max = [config["ddpg"]["action_pos_bound"],
                        config["ddpg"]["action_pos_bound"],
                        np.pi/2]
    #u = np.array([1, 1, np.pi/3])

    for n_epi in range(config["simulation"]["n_epi"]):
        x = env.reset()
        noise.reset()
        print("episode {}...".format(n_epi))
        where = 0
        while True:
            u = agent.action(x, reference) + noise.sample()
            #u = np.clip(u, action_bound_min, action_bound_max)
            u_converted = env.action_convert_to_pos(u)
            reach = 0
            show = 0
            while True:
            #for i in range(5000):
                xn, r, done, quad_pos, _, e_set = env.step(u_converted, reference)
                """
                if show % 1 == 0: 
                    print("quad1 pos error: {}, quad2 pos error: {}".format(
                        np.linalg.norm(u_converted[0:3]-quad_pos[0]),
                        np.linalg.norm(u_converted[3:6]-quad_pos[1])
                    ))
                    u_controller = env.controller(u_converted)
                    print("control input1: {}, control input2: {}".format(
                        u_controller[0:3], u_controller[3:6]))
                    print("quad1 pos: {}, quad2 pos: {}".format(
                        quad_pos[0], quad_pos[1]))
                    print("payload position: {}, velocity: {}".format(
                        xn[0:2], xn[2:4]))
                show += 1
                """
                if all(e_set < config["simulation"]["reach"]):
                    reach = 1
                if reach == 1 or done:
                    break
            if reach == 1:
                r_train = (r + 100) / 100
                agent.memorize((x, u, r_train, xn, done, reference))
                x = xn
                n_data += 1
                where += 1
                #print(where)
            if done:
                break
            if n_data > 1000:
                print('on training..')
                agent.train()


        if (n_epi * config["simulation"]["n_test"]) % config["simulation"][
            "n_epi"
        ] == 0 and n_epi != 0:
            for test_epi in range(config["simulation"]["n_test_epi"]):
                x = env.reset(fixed_init=config["simulation"]["fixed_init"])
                # x = env.reset()
                while True:
                    u = agent.action(x, reference, use="behavior")
                    u_converted = env.action_convert_to_pos(u)
                    while True:
                        x, r, done, quad_pos, load_pos, e_set = env.step(
                            u_converted, reference
                        )
                        if all(e_set < config["simulation"]["reach"]):
                            break
                    r_train = (r + 100) / 100
                    score += r_train
                    """
                    print("quad1 to load distance: {}, \
                            quad2 to load distance: {}, \
                            quad1 y position: {}, \
                            quad2 y position: {}, \
                            load y position: {}".format(
                                np.linalg.norm(load_pos - quad_pos[0]),
                                np.linalg.norm(load_pos - quad_pos[1]),
                                quad_pos[0][1], quad_pos[1][1], load_pos[1]))
                    """
                    if test_epi == 0 and config["simulation"]["animation"]:
                        if n_epi == config["simulation"]["n_epi"] - 100:
                            data_save.append((x, r, quad_pos, load_pos, u))
                        plt.cla()
                        plt.axis([-10, 10, -1, 15])
                        plt.plot(
                            config["simulation"]["reference"][0],
                            config["simulation"]["reference"][2],
                            "go",
                        )
                        plt.plot(
                            [x[0], quad_pos[0][0]],
                            [-x[1], -quad_pos[0][2]],
                            "k",
                            linewidth=0.5,
                        )
                        plt.plot(
                            [x[0], quad_pos[1][0]],
                            [-x[1], -quad_pos[1][2]],
                            "k",
                            linewidth=0.5,
                        )
                        plt.scatter(x[0], -x[1], color="r")
                        plt.scatter(quad_pos[0][0], -quad_pos[0][2], color="b")
                        plt.scatter(quad_pos[1][0], -quad_pos[1][2], color="b")
                        plt.title(f"{n_epi} episodes are learned")
                        plt.grid(True)
                        plt.pause(config["env"]["dt"])

                    if done:
                        break
                if test_epi == 0 and config["simulation"]["animation"]:
                    plt.show(block=False)
                    time.sleep(3)
                    plt.close("all")

            avg_score = score / config["simulation"]["n_test_epi"]

            print(
                "# of episode: {},\
                    avg score: {}".format(
                    n_epi, avg_score
                )
            )
            score_save.append(np.hstack((n_epi, avg_score)))
            score = 0

    env.close()
    return np.vstack(score_save), data_save


def figure(score, data_save):
    past = -1
    loglist = sorted(os.listdir("./log"))
    load_path = os.path.join("log", loglist[past], "data.h5")
    data = logging.load(load_path)
    save_path = os.path.join("log", loglist[past])

    pos = data["state"]["payload"]["pos"]
    quad_pos = pos \
        - config["env"]["quad.l"] * data["state"]["links"]["link00"]["q"]
    action_pos = data["action_converted"][:, 0:3, :]
    time = data["time"]

    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(time, quad_pos[:, 0], "r--", time, action_pos[:, 0], "b")
    ax[1].plot(time, quad_pos[:, 2], "r--", time, action_pos[:, 2], "b")
    ax[0].set_title("Position of quadrotor 1")
    fig2, ax2 = plt.subplots(nrows=2, ncols=1)
    ax2[0].plot(
        time,
        pos[:, 0],
        "r--",
        time,
        config["simulation"]["reference"][0] * np.ones(len(time)),
        "b",
    )
    ax2[1].plot(
        time,
        pos[:, 2],
        "r--",
        time,
        config["simulation"]["reference"][2] * np.ones(len(time)),
        "b",
    )
    ax2[0].set_title("Position of payload")
    fig3, ax3 = plt.subplots(nrows=1, ncols=1)
    ax3.plot(score[:, 0], score[:, 1])
    ax3.set_title("Return")
    ax3.set_xlabel("episode number")
    plt.show()


if __name__ == "__main__":
    score, data_save = main()
    #figure(score, data_save)
