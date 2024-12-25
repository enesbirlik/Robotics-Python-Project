import numpy as np
from typing import Tuple, Optional
from .solver_base import SolverBase

class DLSSolver(SolverBase):
    """Damped Least Squares inverse kinematics solver"""
    
    def solve(self, 
             target_position: np.ndarray, 
             max_iter: int = 100,
             tolerance: float = 1e-3,
             lambda_val: float = 0.1,
             **kwargs) -> Tuple[Optional[np.ndarray], int]:
        """
        Solve IK using DLS method
        
        Args:
            target_position: Target end-effector position
            max_iter: Maximum iterations
            tolerance: Convergence tolerance
            lambda_val: Damping factor
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (joint angles, iterations used)
        """
        self._pre_solve()
        
        try:
            q = np.zeros(self.robot.num_joints)
            I = np.eye(3)  # 3x3 identity matrix (for position only)
            
            for iteration in range(max_iter):
                current_pose = self.robot.forward_kinematics(q)
                if current_pose is None:
                    continue
                    
                current_pos = current_pose.t
                error = target_position - current_pos
                error_norm = np.linalg.norm(error)
                
                if error_norm < tolerance:
                    self._post_solve(q, target_position, iteration)
                    return q, iteration
                
                # Calculate Jacobian
                J = self.robot.robot.jacob0(q)[:3, :]  # Only position components
                
                # DLS solution
                JT = J.T
                dq = np.dot(JT, np.linalg.solve(np.dot(J, JT) + lambda_val**2 * I, error))
                
                # Update joint angles
                q = q + dq
                
                # Apply joint limits
                for i in range(len(q)):
                    q[i] = np.clip(q[i], self.robot.qlim[i][0], self.robot.qlim[i][1])
            
            self._post_solve(q, target_position, max_iter)
            return q, max_iter
            
        except Exception as e:
            print(f"DLS solver error: {str(e)}")
            self._post_solve(None, target_position, max_iter)
            return None, max_iter