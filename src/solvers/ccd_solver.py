import numpy as np
from typing import Tuple, Optional
from .solver_base import SolverBase

class CCDSolver(SolverBase):
    """Cyclic Coordinate Descent solver implementation"""
    
    def solve(self, 
             target_position: np.ndarray, 
             max_iter: int = 100,
             tolerance: float = 1e-3,
             **kwargs) -> Tuple[Optional[np.ndarray], int]:
        """
        Solve IK using CCD method
        
        Args:
            target_position: Target end-effector position
            max_iter: Maximum iterations
            tolerance: Convergence tolerance
            **kwargs: Additional parameters (unused in CCD)
            
        Returns:
            Tuple of (joint angles, iterations used)
        """
        self._pre_solve()
        
        try:
            q = np.zeros(self.robot.num_joints)
            
            for iteration in range(max_iter):
                current_pose = self.robot.forward_kinematics(q)
                if current_pose is None:
                    continue
                    
                end_effector = current_pose.t
                error = np.linalg.norm(end_effector - target_position)
                
                if error < tolerance:
                    self._post_solve(q, target_position, iteration)
                    return q, iteration
                
                # Backward iteration through joints
                for i in range(len(q)-1, -1, -1):
                    # Get current joint position
                    pivot = self.robot.forward_kinematics(q[:i+1]).t if i >= 0 else np.zeros(3)
                    
                    # Calculate vectors
                    current_to_target = target_position - pivot
                    current_to_end = end_effector - pivot
                    
                    # Check vector magnitudes
                    if np.linalg.norm(current_to_target) < 1e-6 or np.linalg.norm(current_to_end) < 1e-6:
                        continue
                    
                    # Normalize vectors
                    current_to_target = current_to_target / np.linalg.norm(current_to_target)
                    current_to_end = current_to_end / np.linalg.norm(current_to_end)
                    
                    # Calculate rotation angle
                    dot_product = np.clip(np.dot(current_to_end, current_to_target), -1.0, 1.0)
                    angle = np.arccos(dot_product)
                    
                    # Determine rotation direction
                    cross_product = np.cross(current_to_end, current_to_target)
                    rotation_axis = np.array([0, 0, 1])  # Z-axis rotation
                    if np.dot(cross_product, rotation_axis) < 0:
                        angle = -angle
                    
                    # Apply joint limits and update
                    q[i] = np.clip(q[i] + angle, self.robot.qlim[i][0], self.robot.qlim[i][1])
                    
                    # Update end effector position
                    new_pose = self.robot.forward_kinematics(q)
                    if new_pose is not None:
                        end_effector = new_pose.t
            
            self._post_solve(q, target_position, max_iter)
            return q, max_iter
            
        except Exception as e:
            print(f"CCD solver error: {str(e)}")
            self._post_solve(None, target_position, max_iter)
            return None, max_iter