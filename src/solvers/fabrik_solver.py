import numpy as np
from typing import Tuple, Optional, List
from .solver_base import SolverBase

class FABRIKSolver(SolverBase):
    """Forward And Backward Reaching Inverse Kinematics solver implementation"""
    
    def solve(self, 
             target_position: np.ndarray, 
             max_iter: int = 100,
             tolerance: float = 1e-3,
             **kwargs) -> Tuple[Optional[np.ndarray], int]:
        """
        Solve IK using FABRIK method
        
        Args:
            target_position: Target end-effector position
            max_iter: Maximum iterations
            tolerance: Convergence tolerance
            **kwargs: Additional parameters (unused in FABRIK)
            
        Returns:
            Tuple of (joint angles, iterations used)
        """
        self._pre_solve()
        
        try:
            num_joints = self.robot.num_joints
            q = np.zeros(num_joints)
            
            # Calculate initial joint positions
            joint_positions: List[np.ndarray] = []
            initial_pose = self.robot.forward_kinematics(q)
            if initial_pose is None:
                return None, max_iter
                
            base_pos = np.zeros(3)
            end_effector = initial_pose.t
            
            # Calculate link lengths
            link_lengths = []
            for i in range(num_joints):
                pos1 = self.robot.forward_kinematics(q[:i+1]).t if i > 0 else base_pos
                pos2 = self.robot.forward_kinematics(q[:i+2]).t if i < num_joints-1 else end_effector
                link_lengths.append(np.linalg.norm(pos2 - pos1))
            
            # Check if target is reachable
            total_length = sum(link_lengths)
            target_distance = np.linalg.norm(target_position - base_pos)
            if target_distance > total_length:
                print("Target position is unreachable")
                return None, max_iter
            
            # FABRIK iterations
            for iteration in range(max_iter):
                current_pos = end_effector
                error = np.linalg.norm(current_pos - target_position)
                
                if error < tolerance:
                    # Calculate final angles
                    for i in range(num_joints-1):
                        v1 = joint_positions[i+1] - joint_positions[i]
                        v2 = joint_positions[i+2] - joint_positions[i+1]
                        
                        if np.linalg.norm(v1) > 1e-10 and np.linalg.norm(v2) > 1e-10:
                            angle = np.arctan2(np.cross(v1, v2)[2], np.dot(v1, v2))
                            q[i] = np.clip(angle, self.robot.qlim[i][0], self.robot.qlim[i][1])
                    
                    self._post_solve(q, target_position, iteration)
                    return q, iteration
                
                # Backward reaching
                new_positions = [target_position]
                for i in range(num_joints-1, -1, -1):
                    direction = (new_positions[0] - current_pos)
                    distance = link_lengths[i]
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction) * distance
                    new_positions.insert(0, new_positions[0] - direction)
                
                # Forward reaching
                new_positions[0] = base_pos
                for i in range(1, len(new_positions)):
                    direction = (new_positions[i] - new_positions[i-1])
                    distance = link_lengths[i-1]
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction) * distance
                    new_positions[i] = new_positions[i-1] + direction
                
                joint_positions = new_positions
                
                # Calculate new angles
                for i in range(num_joints):
                    if i < num_joints-1:
                        v1 = joint_positions[i+1] - joint_positions[i]
                        v2 = joint_positions[i+2] - joint_positions[i+1]
                        if np.linalg.norm(v1) > 1e-10 and np.linalg.norm(v2) > 1e-10:
                            angle = np.arctan2(np.cross(v1, v2)[2], np.dot(v1, v2))
                            q[i] = np.clip(angle, self.robot.qlim[i][0], self.robot.qlim[i][1])
            
            self._post_solve(q, target_position, max_iter)
            return q, max_iter
            
        except Exception as e:
            print(f"FABRIK solver error: {str(e)}")
            self._post_solve(None, target_position, max_iter)
            return None, max_iter