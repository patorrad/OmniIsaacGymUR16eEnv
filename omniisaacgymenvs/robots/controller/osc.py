
# isaac-universal_robots
from omni.isaac.universal_robots import UR10

import numpy as np
import torch

def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)

class Controller_osc():
    def __init__(
        self, 
        robots,
        device='cuda:0'
    ) -> None:
        
        self.device = device
        # TODO Need to find in articulation_view
        self._franka_effort_limits = torch.tensor([330., 330., 150.,  56.,  56.,  56.], device='cuda:0')
        self._num_dof = 6 #robots.num_dof
        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [0.06, -2.5, 2.03, 0.58, 1.67, 1.74], device=self.device
        )

        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * self._num_dof, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0)
        
        return

    # def _compute_osc_torques(self, dpose):
    #     # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
    #     # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
    #     q, qd = self._q[:, :6], self._qd[:, :6]
    #     mm_inv = torch.inverse(self._mm)
    #     m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
    #     m_eef = torch.inverse(m_eef_inv)

    #     # Transform our cartesian action `dpose` into joint torques `u`
    #     u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
    #             self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

    #     # Nullspace control torques `u_null` prevents large changes in joint configuration
    #     # They are added into the nullspace of OSC so that the end effector orientation remains constant
    #     # roboticsproceedings.org/rss07/p31.pdf
    #     j_eef_inv = m_eef @ self._j_eef @ mm_inv
    #     u_null = self.kd_null * -qd + self.kp_null * (
    #             (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
    #     u_null[:, self.num_franka_dofs:] *= 0
    #     u_null = self._mm @ u_null.unsqueeze(-1)
    #     u += (torch.eye(self.num_franka_dofs, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

    #     # Clip the values to be within valid effort range
    #     u = tensor_clamp(u.squeeze(-1),
    #                      -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))

    #     return u
    
    def _compute_osc_torques(self, robots, dpose=0.):
        q = robots.get_joint_positions(clone=True)
        qd = robots.get_joint_velocities(clone=True)
        _mm = robots.get_mass_matrices(clone=True)
        _j_eef = torch.squeeze(robots.get_jacobians(clone=True)[:,6:,:,:])
        # print('$$$$$$$$$$$$$$$$$$$$$$$$$, ', _j_eef.shape)
        # b = _j_eef[:,6:,:,:]
        # print('$$$$$$$$$$$$$$$$$$$$$$$$$, ', b.shape)
        # b = torch.squeeze(b)
        # print('$$$$$$$$$$$$$$$$$$$$$$$$$, ', b.shape)

        if _mm is not None:
            mm_inv = torch.inverse(_mm)
            m_eef_inv = _j_eef @ mm_inv @ torch.transpose(_j_eef, 1, 2)
            m_eef = torch.inverse(m_eef_inv)
            
            # print(robots._body_indices)
            eef_vel = robots.get_velocities() #robots.get_body_index("ee_link"))
            # print('$$$$$$$$$$$$$$$$$$$$$$$$$ eef_vel.shape, ', eef_vel.shape)
            # print('$$$$$$$$$$$$$$$$$$$$$$$$$ eef_vel, ', eef_vel)
            # Transform our cartesian action `dpose` into joint torques `u`
            u = torch.transpose(_j_eef, 1, 2) @ m_eef @ (
                    self.kp * dpose - self.kd * eef_vel).unsqueeze(-1)
            
            # # Nullspace control torques `u_null` prevents large changes in joint configuration
            # # They are added into the nullspace of OSC so that the end effector orientation remains constant
            # # roboticsproceedings.org/rss07/p31.pdf
            # j_eef_inv = m_eef @ _j_eef @ mm_inv
            # u_null = self.kd_null * -qd + self.kp_null * (
            #         (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
            # u_null[:, self._num_dof:] *= 0
            # u_null = _mm @ u_null.unsqueeze(-1)
            # u += (torch.eye(self._num_dof, device=self.device).unsqueeze(0) - torch.transpose(_j_eef, 1, 2) @ j_eef_inv) @ u_null

            # Clip the values to be within valid effort range
            u = tensor_clamp(u.squeeze(-1),
                            -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))
            # print('$$$$$$$$$$$$$$$$$$$$$$$$$, ', u)
            return u
