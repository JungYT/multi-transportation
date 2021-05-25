"""
Dynamics of multi-quadrotors with slung load.
Reference: Taeyoun Lee, "Geometric Control of Quadrotor UAVs Transporting
a Cable-Suspended Rigid Body," IEEE TCST, 2018.
Note
1. Rotation matrix(dcm) in the paper and code is the one used in Robotics,
which means the dcm should be transposed to use ``rot.angle2dcm`` package.
"""
import numpy as np
import numpy.linalg as nla
from collections import deque

from fym.core import BaseEnv, BaseSystem
import fym.core as core
from fym.utils import rot
from utils import hat

# np.random.seed(0)

class Load(BaseEnv):
    def __init__(self, pos_bound, att_bound, mass, J):
        super().__init__()
        self.pos = BaseSystem(np.vstack((
            np.random.uniform(
                low=pos_bound[0][0],
                high=pos_bound[0][1]
            ),
            np.random.uniform(
                low=pos_bound[1][0],
                high=pos_bound[1][1]
            ),
            np.random.uniform(
                low=pos_bound[2][0],
                high=pos_bound[2][1]
            )
        )))
        self.vel = BaseSystem(np.vstack((0., 0., 0.)))
        self.dcm = BaseSystem(rot.angle2dcm(
            np.random.uniform(
                low=att_bound[2][0],
                high=att_bound[2][1]
            ),
            np.random.uniform(
                low=att_bound[1][0],
                high=att_bound[1][1]
            ),
            np.random.uniform(
                low=att_bound[0][0],
                high=att_bound[0][1]
            )
        ).T)
        self.omega = BaseSystem(np.vstack((0., 0., 0.)))
        self.mass = mass
        self.J = J

    def set_dot(self, acc, ang_acc):
        self.pos.dot = self.vel.state
        self.vel.dot = acc
        self.omega.dot = ang_acc
        self.dcm.dot = self.dcm.state.dot(hat(self.omega.state))


class Link(BaseEnv):
    def __init__(self, length, anchor, uvec_bound):
        super().__init__()
        self.uvec = BaseSystem(rot.spherical2cartesian(
            1,
            np.random.uniform(
                low=uvec_bound[0][0],
                high=uvec_bound[0][1]
            ),
            np.random.uniform(
                low=uvec_bound[1][0],
                high=uvec_bound[1][1]
            )
        ))
        self.omega = BaseSystem(np.vstack((0., 0., 0.)))
        self.len = length
        self.anchor = anchor

    def set_dot(self, ang_acc):
        self.uvec.dot = hat(self.omega.state).dot(self.uvec.state)
        self.omega.dot = ang_acc


class Quadrotor(BaseEnv):
    def __init__(self, mass, J):
        super().__init__()
        self.dcm = BaseSystem(np.eye(3))
        self.omega = BaseSystem(np.vstack((0., 0., 0.)))
        self.mass = mass
        self.J = J

    def set_dot(self, moment):
        self.dcm.dot = self.dcm.state.dot(hat(self.omega.state))
        self.omega.dot = nla.inv(self.J).dot(
            moment - hat(self.omega.state).dot(self.J.dot(self.omega.state))
        )


class MultiQuadSlungLoad(BaseEnv):
    def __init__(self, cfg):
        super().__init__(dt=cfg.dt, max_t=cfg.max_t, solver=cfg.solver,
                         ode_step_len=cfg.ode_step_len)
        self.quad_num = cfg.quad.num
        self.load = Load(
            cfg.load.pos_bound,
            cfg.load.att_bound,
            cfg.load.mass,
            cfg.load.J
        )
        self.links = core.Sequential(**{f"link{i:02d}": Link(
            cfg.link.len[i],
            cfg.link.anchor[i],
            cfg.link.uvec_bound
        ) for i in range(self.quad_num)})
        self.quads = core.Sequential(**{f"quad{i:02d}": Quadrotor(
            cfg.quad.mass,
            cfg.quad.J
        ) for i in range(self.quad_num)})
        self.g = np.vstack((0., 0., -9.81))
        self.cfg = cfg

        self.S1_set = deque(maxlen=self.quad_num)
        self.S2_set = deque(maxlen=self.quad_num)
        self.S3_set = deque(maxlen=self.quad_num)
        self.S4_set = deque(maxlen=self.quad_num)
        self.S5_set = deque(maxlen=self.quad_num)
        self.S6_set = deque(maxlen=self.quad_num)
        self.S7_set = deque(maxlen=self.quad_num)

        self.iscollision = False
        self.P = cfg.ddpg.P
        self.Ke = cfg.controller.Ke
        self.Ks = cfg.controller.Ks
        self.chattering_bound = cfg.controller.chattering_bound
        self.unc_max = cfg.controller.unc_max

    def reset(self, load_pos_des, load_att_des, fixed_init=False):
        super().reset()
        if fixed_init:
            self.load.pos.state = self.cfg.load.pos_init
            self.load.dcm.state = self.cfg.load.dcm_init

            for link in self.links.systems:
                link.uvec.state = np.vstack((0., 0., -1.))
        obs = self.observe(load_pos_des, load_att_des)
        return obs

    def set_dot(self, t, quad_att_des, f_des):
        m_T = self.load.mass
        R0 = self.load.dcm.state
        omega = self.load.omega.state
        omega_hat = hat(omega)
        omega_hat_square = omega_hat.dot(omega_hat)

        for i,(link, quad) in enumerate(
            zip(self.links.systems, self.quads.systems)
        ):
            l = link.len
            rho = link.anchor
            q = link.uvec.state
            w = link.omega.state
            m = quad.mass
            R = quad.dcm.state
            u = f_des[i] * R.dot(np.vstack((0., 0., 1.)))

            m_T += m
            q_hat_square = (hat(q)).dot(hat(q))
            q_qT = np.eye(3) + q_hat_square
            rho_hat = hat(rho)
            rhohat_R0T = rho_hat.dot(R0.T)
            w_norm = nla.norm(w)
            l_w_square_q = l * w_norm * w_norm * q
            R0_omega_square_rho = R0.dot(omega_hat_square.dot(rho))

            S1_temp = q_qT.dot(u - m*R0_omega_square_rho) - m*l_w_square_q
            S2_temp = m * q_qT.dot(rhohat_R0T.T)
            S3_temp = m * rhohat_R0T.dot(q_qT.dot(rhohat_R0T.T))
            S4_temp = m * q_hat_square
            S5_temp = m * rhohat_R0T.dot(q_qT)
            S6_temp = rhohat_R0T.dot(
                q_qT.dot(u + m*self.g) \
                - m*q_hat_square.dot(R0_omega_square_rho) \
                - m*l_w_square_q
            )
            S7_temp = m * rho_hat.dot(rho_hat)
            self.S1_set.append(S1_temp)
            self.S2_set.append(S2_temp)
            self.S3_set.append(S3_temp)
            self.S4_set.append(S4_temp)
            self.S5_set.append(S5_temp)
            self.S6_set.append(S6_temp)
            self.S7_set.append(S7_temp)
        S1 = sum(self.S1_set)
        S2 = sum(self.S2_set)
        S3 = sum(self.S3_set)
        S4 = sum(self.S4_set)
        S5 = sum(self.S5_set)
        S6 = sum(self.S6_set)
        S7 = sum(self.S7_set)

        J_bar = self.load.J - S7
        J_hat = self.load.J + S3
        J_hat_inv = np.linalg.inv(J_hat)
        Mq = m_T*np.eye(3) + S4
        A = -J_hat_inv.dot(S5)
        B = J_hat_inv.dot(S6 - omega_hat.dot(J_bar.dot(omega)))
        C = Mq + S2.dot(A)

        load_acc = nla.inv(C).dot(Mq.dot(self.g) + S1 - S2.dot(B))
        load_ang_acc = A.dot(load_acc) + B
        self.load.set_dot(load_acc, load_ang_acc)

        for i, (link, quad) in enumerate(
                zip(self.links.systems, self.quads.systems)
        ):
            l = link.len
            rho = link.anchor
            q = link.uvec.state
            q_hat = hat(q)
            m = quad.mass
            R = quad.dcm.state
            u = f_des[i] * R.dot(np.vstack((0., 0., 1.)))
            R0_omega_square_rho = R0.dot(omega_hat_square.dot(rho))
            D = R0.dot(hat(rho).dot(load_ang_acc)) + self.g + u/m

            link_ang_acc = q_hat.dot(load_acc + R0_omega_square_rho - D) / l
            link.set_dot(link_ang_acc)
            M = self.control_attitude(
                quad_att_des[i],
                R,
                quad.omega.state,
                quad.J
            )
            quad.set_dot(M)

    def step(self, load_pos_des, load_att_des, action):
        quad_att_des = 3*[np.vstack((0., 0., 0.))]
        f_des = 3*[25]
        *_, time_out = self.update(quad_att_des=quad_att_des, f_des=f_des)
        done = self.terminate(time_out)
        obs = self.observe(load_pos_des, load_att_des)
        reward = self.get_reward(load_pos_des, load_att_des)
        return obs, reward, done

    def logger_callback(self, t, y, i, t_hist, ode_hist,
                        quad_att_des, f_des):
        states = self.observe_dict(y)
        load = states['load']
        links = states['links']
        quads = states['quads']
        M = [self.control_attitude(
            quad_att_des[i],
            states['quads'][f'quad{i:02d}']['dcm'],
            states['quads'][f'quad{i:02d}']['omega'],
            quad.J
        ) for i, quad in enumerate(self.quads.systems)]
        load_att = np.vstack(rot.dcm2angle(load['dcm'].T)[::-1])

        quad_pos = []
        quad_vel = []
        quad_att = []
        anchor_pos = []
        # check_dynamics = []
        for i, link in enumerate(self.links.systems):
            quad_pos.append(
                load['pos'] + load['dcm'].dot(link.anchor) \
                - link.len*links[f'link{i:02d}']['uvec']
            )
            quad_vel.append(
                load['vel'] \
                + load['dcm'].dot(hat(load['omega']).dot(link.anchor)) \
                - link.len*hat(links[f'link{i:02d}']['omega']).dot(
                    links[f'link{i:02d}']['uvec']
                )
            )
            quad_att.append(
                np.array(rot.dcm2angle(quads[f'quad{i:02d}']['dcm']))[::-1]
            )
            anchor_pos.append(load['pos'] + load['dcm'].dot(link.anchor))
            # check_dynamics.append(np.dot(
            #     links[f'link{i:02d}']['uvec'].reshape(-1, ),
            #     links[f'link{i:02d}']['omega'].reshape(-1,)
            # )
            #                       )
        distance_btw_quads = self.check_collision(quad_pos)
        distance_btw_quad2anchor = [nla.norm(quad_pos[i]-anchor_pos[i])
                    for i in range(self.quad_num)]
        return dict(time=t, **states, load_att=load_att, quad_moment=M,
                    quad_att_des=quad_att_des, f_des=f_des,
                    quad_pos=quad_pos, quad_vel=quad_vel, quad_att=quad_att,
                    anchor_pos=anchor_pos,
                    distance_btw_quads=distance_btw_quads,
                    distance_btw_quad2anchor=distance_btw_quad2anchor)

    def check_collision(self, quads_pos):
        distance = [nla.norm(quads_pos[i]-quads_pos[i+1])
                      for i in range(self.quad_num-1)]
        distance.append(nla.norm(quads_pos[-1] - quads_pos[0]))
        if not self.iscollision and any(
            i < self.cfg.quad.iscollision for i in distance
        ):
            self.iscollision = True
        return distance

    def terminate(self, done):
        load_posz = self.load.pos.state[2]
        done = 1. if (load_posz < 0 or done or self.iscollision) else 0.
        return done

    def control_attitude(self, quad_att_des, quad_dcm, quad_omega, J):
        quad_att = np.vstack(
            rot.dcm2angle(quad_dcm.T)[::-1]
        )
        phi, theta, _ = quad_att.squeeze()
        omega = quad_omega
        omega_hat = hat(omega)
        wx, wy, wz = quad_omega.squeeze()

        L = np.array([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ])
        L2 = np.array([
            [wy*np.cos(phi)*np.tan(theta) - wz*np.sin(phi)*np.tan(theta),
            wy*np.sin(phi)/(np.cos(theta))**2 \
             + wz*np.cos(phi)/(np.cos(theta))**2,
            0],
            [-wy*np.sin(phi) - wz*np.cos(phi), 0, 0],
            [wy*np.cos(phi)/np.cos(theta) - wz*np.sin(phi)*np.cos(theta),
            wy*np.sin(phi)*np.tan(theta)/np.cos(theta) \
             - wz*np.cos(phi)*np.tan(theta)/np.cos(theta),
            0]
        ])
        b = np.vstack((wx, 0., 0.))
        e2 = L.dot(omega)
        e1 = quad_att - quad_att_des
        s = self.Ke*e1 + e2
        s_clip = np.clip(s/self.chattering_bound, -1, 1)
        M = (J.dot(nla.inv(L))).dot(
            -self.Ke*e2 - b - L2.dot(e2) - s_clip*(self.unc_max + self.Ks)
        ) + omega_hat.dot(J.dot(omega))
        return M

    def observe(self, load_pos_des, load_att_des):
        load_pos = self.load.pos.state
        load_att = np.vstack(rot.dcm2angle(self.load.dcm.state.T))[::-1]
        e_load_pos = load_pos - load_pos_des
        e_load_att = load_att - load_att_des
        return np.vstack((e_load_pos, e_load_att))

    def get_reward(self, load_pos_des, load_att_des):
        obs = self.observe(load_pos_des, load_att_des)
        load_pos = self.load.pos.state
        if (load_pos[2] < 0 or self.iscollision):
            r = -np.array([50])
        else:
            r = -np.transpose(obs).dot(self.P.dot(obs)).reshape(-1,)
        return r




