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
from utils import hat, unhat, block_diag

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
        self.uvec = BaseSystem(rot.sph2cart2(
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
        self.cfg = cfg
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
        ) for i in range(cfg.quad.num)})
        self.quads = core.Sequential(**{f"quad{i:02d}": Quadrotor(
            cfg.quad.mass,
            cfg.quad.J
        ) for i in range(cfg.quad.num)})
        self.g = cfg.g
        self.e3 = np.vstack((0., 0., 1.))
        self.eye = np.eye(3)

        self.iscollision = False
        self.P_anchor = np.block([
            [np.eye(3)]*self.cfg.quad.num,
            [hat(self.cfg.link.anchor[i]) for i in range(self.cfg.quad.num)]
        ])
        self.P_anchor_sudo = self.P_anchor.T.dot(
            nla.inv(self.P_anchor.dot(self.P_anchor.T))
        )

    def reset(self, des, fixed_init=False):
        load_pos_des, load_att_des, _ = des
        super().reset()
        if fixed_init:
            self.load.pos.state = self.cfg.load.pos_init
            self.load.dcm.state = self.cfg.load.dcm_init

            uvec_init = np.vstack((0., 0., -1.))
            for link in self.links.systems:
                link.uvec.state = uvec_init
        obs = self.observe(load_pos_des, load_att_des)
        return obs

    def set_dot(self, t, quad_att_des, f_des):
        m_T = self.load.mass
        R0 = self.load.dcm.state
        omega = self.load.omega.state
        omega_hat = hat(omega)
        omega_hat_square = omega_hat.dot(omega_hat)

        S1_set = [None] * self.cfg.quad.num
        S2_set = [None] * self.cfg.quad.num
        S3_set = [None] * self.cfg.quad.num
        S4_set = [None] * self.cfg.quad.num
        S5_set = [None] * self.cfg.quad.num
        S6_set = [None] * self.cfg.quad.num
        S7_set = [None] * self.cfg.quad.num

        # load_pos = self.load.pos.state
        # load_vel = self.load.vel.state

        # F_d, M_d = self.design_desired_FM(load_pos_des, load_dcm_des, load_pos,
        #                                   R0, omega, load_vel)
        # tension_des = self.calculate_desired_tension(F_d, M_d, R0)



        for i, (link, quad) in enumerate(
            zip(self.links.systems, self.quads.systems)
        ):
            l = link.len
            rho = link.anchor
            q = link.uvec.state
            w = link.omega.state
            m = quad.mass
            R = quad.dcm.state
            u = f_des[i] * R.dot(self.e3)

            m_T += m
            q_hat_square = (hat(q)).dot(hat(q))
            q_qT = self.eye + q_hat_square
            rho_hat = hat(rho)
            rhohat_R0T = rho_hat.dot(R0.T)
            w_norm = np.sqrt(w[0]**2 + w[1]**2 + w[2]**2)
            l_w_square_q = l * w_norm * w_norm * q
            R0_omega_square_rho = R0.dot(omega_hat_square.dot(rho))

            # mu_des = tension_des[3*i:3*(i+1)]
            # mu = (q.dot(q.transpose())).dot(mu_des)
            # anchor_acc = self.load.vel.dot - self.g + R0_omega_square_rho \
            #     - R0.dot(hat(rho).dot(self.load.omega.dot))

            # u_parallel = mu + m*l*w_norm*w_norm*q \
            #     + m*(q.dot(q.transpose()).dot(anchor_acc))

            # q_des = -mu_des/np.sqrt(mu_des[0]**2 + mu_des[1]**2 + mu_des[2]**2)
            # w_des = q_des
            # e_q = hat(q_des).dot(q)
            # e_w = w + q_hat_square.dot(

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
            S1_set[i] = S1_temp
            S2_set[i] = S2_temp
            S3_set[i] = S3_temp
            S4_set[i] = S4_temp
            S5_set[i] = S5_temp
            S6_set[i] = S6_temp
            S7_set[i] = S7_temp
        S1 = sum(S1_set)
        S2 = sum(S2_set)
        S3 = sum(S3_set)
        S4 = sum(S4_set)
        S5 = sum(S5_set)
        S6 = sum(S6_set)
        S7 = sum(S7_set)

        J_bar = self.load.J - S7
        J_hat = self.load.J + S3
        J_hat_inv = nla.inv(J_hat)
        Mq = m_T*self.eye + S4
        A = -J_hat_inv.dot(S5)
        B = J_hat_inv.dot(S6 - omega_hat.dot(J_bar.dot(omega)))
        C = Mq + S2.dot(A)

        load_acc = nla.inv(C).dot(Mq.dot(self.g) + S1 - S2.dot(B))
        load_ang_acc = A.dot(load_acc) + B
        self.load.set_dot(load_acc, load_ang_acc)

        M = [None]*self.cfg.quad.num
        for i, (link, quad) in enumerate(
                zip(self.links.systems, self.quads.systems)
        ):
            l = link.len
            rho = link.anchor
            q = link.uvec.state
            q_hat = hat(q)
            m = quad.mass
            R = quad.dcm.state
            u = f_des[i] * R.dot(self.e3)
            R0_omega_square_rho = R0.dot(omega_hat_square.dot(rho))
            D = R0.dot(hat(rho).dot(load_ang_acc)) + self.g + u/m
            link_ang_acc = q_hat.dot(load_acc + R0_omega_square_rho - D) / l
            link.set_dot(link_ang_acc)

            M[i] = self.control_attitude(
                quad_att_des[i],
                R,
                quad.omega.state,
                quad.J
            )
            quad.set_dot(M[i])

        return dict(quad_att_des=quad_att_des, quad_moment=M, f_des=f_des)

    def step(self, action, des):
        # quad_att_des = 3*[np.vstack((np.pi/12., 0., 0.))]
        # quad_att_des = 3*[np.vstack((0., 0., 0.))]
        # f_des = 3*[25]
        load_pos_des, load_att_des, psi_des = des
        quad_att_des, f_des = self.transform_action2des(action, psi_des)
        *_, time_out = self.update(quad_att_des=quad_att_des, f_des = f_des)
        # *_, time_out = self.update(load_pos_des=np.vstack((0., 0., 0.)),
        #                            load_dcm_des = np.eye(3))
        done = self.terminate(time_out)
        obs = self.observe(load_pos_des, load_att_des)
        reward = self.get_reward(load_pos_des, load_att_des)
        info = {
            'time': self.clock.get(),
            'reward': reward,
            'action': action,
        }
        return obs, reward, done, info

    def logger_callback(self, t, quad_att_des, f_des):
        load_att = np.vstack(rot.dcm2angle(self.load.dcm.state.T)[::-1])
        quad_pos = [None]*self.cfg.quad.num
        quad_vel = [None]*self.cfg.quad.num
        quad_att = [None]*self.cfg.quad.num
        anchor_pos = [None]*self.cfg.quad.num
        distance_btw_quad2anchor = [None]*self.cfg.quad.num
        # check_dynamics = [None]*self.cfg.quad.num
        for i, (link, quad) in enumerate(
                zip(self.links.systems, self.quads.systems)
        ):
            quad_pos[i] = self.load.pos.state \
                + self.load.dcm.state.dot(link.anchor) \
                - link.len*link.uvec.state
            quad_vel[i] = self.load.vel.state \
                + self.load.dcm.state.dot(
                    hat(self.load.omega.state).dot(link.anchor)
                ) \
                - link.len*hat(link.omega.state).dot(link.uvec.state)
            quad_att[i] = np.array(rot.dcm2angle(quad.dcm.state.T))[::-1]
            anchor_pos[i] = self.load.pos.state \
                + self.load.dcm.state.dot(link.anchor)
            distance_btw_quad2anchor[i] = \
                np.sqrt(
                    (quad_pos[i][0][0] - anchor_pos[i][0][0])**2 \
                    + (quad_pos[i][1][0] - anchor_pos[i][1][0])**2 \
                    + (quad_pos[i][2][0] - anchor_pos[i][2][0])**2
                )
            if distance_btw_quad2anchor[i] < 0.1 or distance_btw_quad2anchor[i] > 1.:
                print('problem!')
            # check_dynamics[i] = np.dot(
            #     links[f'link{i:02d}']['uvec'].reshape(-1, ),
            #     links[f'link{i:02d}']['omega'].reshape(-1,)
            # )
        distance_btw_quads = self.check_collision(quad_pos)
        return dict(time=t, **self.observe_dict(), load_att=load_att,
                    anchor_pos=anchor_pos, quad_vel=quad_vel,
                    quad_att=quad_att, quad_pos=quad_pos,
                    distance_btw_quads=distance_btw_quads,
                    distance_btw_quad2anchor=distance_btw_quad2anchor)

    def design_desired_FM(self, load_pos_des, load_dcm_des, load_pos, load_dcm,
                          load_omega, load_vel):
        e_pos = load_pos - load_pos_des
        e_vel = load_vel
        e_dcm = 0.5 * unhat(
            load_dcm_des.T.dot(load_dcm) - load_dcm.T.dot(load_dcm_des)
        )
        e_omega = load_omega
        F_d = self.load.mass * (
            -self.cfg.controller.Kpos * e_pos \
            -self.cfg.controller.Kvel * e_vel \
            - self.g
        )
        M_d = -self.cfg.controller.Kdcm * e_dcm \
            - self.cfg.controller.Komega * e_omega
        return F_d, M_d

    def calculate_desired_tension(self, F_d, M_d, load_dcm):
        load_dcm_block = block_diag(load_dcm, self.cfg.quad.num)
        tension_des = load_dcm_block.dot(
            self.P_anchor_sudo.dot(np.vstack((load_dcm.T.dot(F_d), M_d)))
        )
        return tension_des

    def design_control_input(self, load_pos_des, load_dcm_des, load_pos,
                             load_dcm, load_omega, load_vel, link_uvec):
        F_d, M_d = self.design_desired_FM(load_pos_des, load_dcm_des, load_pos,
                                          load_dcm, load_omega, load_vel)
        tension_des = self.calculate_desired_tension(F_d, M_d, load_dcm)


    def check_collision(self, quads_pos):
        distance = [
            np.sqrt(
                (quads_pos[i-1][0][0]-quads_pos[i][0][0])**2 \
                    + (quads_pos[i-1][1][0]-quads_pos[i][1][0])**2 \
                    + (quads_pos[i-1][2][0]-quads_pos[i][2][0])**2
            ) for i in range(self.cfg.quad.num)
        ]
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
        s = self.cfg.controller.Ke*e1 + e2
        s_clip = np.clip(s/self.cfg.controller.chattering_bound, -1, 1)
        M = (J.dot(nla.inv(L))).dot(
            -self.cfg.controller.Ke*e2 - b - L2.dot(e2) \
            - s_clip*(self.cfg.controller.unc_max + self.cfg.controller.Ks)
        ) + omega_hat.dot(J.dot(omega))
        return M

    def observe(self, load_pos_des, load_att_des):
        obs = [np.array(rot.cart2sph2(link.uvec.state))[1::]
               for link in self.links.systems]
        load_pos = self.load.pos.state
        load_att = np.vstack(rot.dcm2angle(self.load.dcm.state.T))[::-1]
        e_load_pos = load_pos - load_pos_des
        e_load_att = load_att - load_att_des
        obs.append(e_load_pos.reshape(-1,))
        obs.append(e_load_att.reshape(-1,))
        return np.hstack(obs)

    def get_reward(self, load_pos_des, load_att_des):
        error = self.observe(load_pos_des, load_att_des)[0:6]
        load_pos = self.load.pos.state
        if (load_pos[2] < 0 or self.iscollision):
            r = -np.array([self.cfg.ddpg.reward_max])
        else:
            r = -np.transpose(error).dot(
                self.cfg.ddpg.P.dot(error)
            ).reshape(-1,)
        r_scaled = (r/(self.cfg.ddpg.reward_max/2)+1)*10
        return r_scaled

    def transform_action2des(self, action, psi_des):
        f_des = [None]*self.cfg.quad.num
        quad_att_des = [None]*self.cfg.quad.num
        for i in range(self.cfg.quad.num):
            chi, gamma = action[3*i+1:3*i+3]
            u_des = rot.sph2cart2(1, chi, gamma)
            phi, theta = self.find_euler(u_des, psi_des[i])
            quad_att_des[i] = np.vstack((phi, theta, psi_des[i]))
            f_des[i] = action[3*i]
        return quad_att_des, f_des

    def find_euler(self, vec, psi):
        vec_n = rot.angle2dcm(psi, 0, 0).dot(vec)
        theta = np.arctan2(vec_n[0], vec_n[2]).item()
        phi = np.arctan2(
            -vec_n[1]*vec_n[2],
            np.cos(theta)*(1-vec_n[1]**2)
        ).item()
        return phi, theta

