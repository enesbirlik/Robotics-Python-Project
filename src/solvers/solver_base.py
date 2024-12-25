from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional, Dict, Any
from ..core.robot_base import RobotBase
import time

class SolverBase(ABC):
    """Base class for inverse kinematics solvers"""
    
    def __init__(self, robot: RobotBase):
        """
        Initialize solver with robot instance
        
        Args:
            robot: RobotBase instance
        """
        self.robot = robot
        self._metrics: Dict[str, Any] = {}

    @abstractmethod
    def solve(self, 
             target_position: np.ndarray, 
             max_iter: int = 100,
             tolerance: float = 1e-3,
             **kwargs) -> Tuple[Optional[np.ndarray], int]:
        """
        Solve inverse kinematics
        
        Args:
            target_position: Target end-effector position
            max_iter: Maximum iterations
            tolerance: Convergence tolerance
            **kwargs: Additional solver-specific parameters
            
        Returns:
            Tuple of (joint angles, iterations used)
        """
        pass

    def _pre_solve(self) -> None:
        """Pre-solve initialization"""
        self._metrics = {
            'start_time': time.time(),
            'iterations': 0,
            'error': float('inf'),
            'joint_angles': None,
            'success': False
        }

    def _post_solve(self, 
                   joint_angles: Optional[np.ndarray], 
                   target_position: np.ndarray,
                   iterations: int) -> None:
        """
        Post-solve metrics calculation
        
        Args:
            joint_angles: Final joint angles
            target_position: Target position
            iterations: Number of iterations used
        """
        self._metrics['end_time'] = time.time()
        self._metrics['time'] = self._metrics['end_time'] - self._metrics['start_time']
        self._metrics['iterations'] = iterations
        
        if joint_angles is not None:
            final_pose = self.robot.forward_kinematics(joint_angles)
            if final_pose is not None:
                self._metrics['error'] = np.linalg.norm(final_pose.t - target_position)
                self._metrics['joint_angles'] = joint_angles
                self._metrics['success'] = True

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get solver metrics"""
        return self._metrics