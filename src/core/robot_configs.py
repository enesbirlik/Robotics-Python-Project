import numpy as np
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass

@dataclass
class RobotConfig:
    """Robot configuration dataclass"""
    dh_params: List[List[float]]
    joint_types: str
    qlim: List[List[float]]
    name: str
    description: str = ""

class RobotConfigs:
    """Robot configurations storage class"""
    
    @staticmethod
    def get_ur5_config() -> RobotConfig:
        """Get UR5 robot configuration"""
        # [a, alpha, d, offset]
        dh_params = [
            [0,     np.pi/2,     89.2,     0],    # Link 1 (Revolute)
            [-425,  0,           0,        0],    # Link 2 (Revolute)
            [-392,  0,           0,        0],    # Link 3 (Revolute)
            [0,     np.pi/2,     109.3,    0],    # Link 4 (Revolute)
            [0,    -np.pi/2,     94.75,    0],    # Link 5 (Revolute)
            [0,     0,           82.5,     0]     # Link 6 (Revolute)
        ]

        joint_types = "RRRRRR"  # R: Revolute, P: Prismatic
        
        qlim = [
            [-2*np.pi, 2*np.pi],  # Joint 1
            [-2*np.pi, 2*np.pi],  # Joint 2
            [-2*np.pi, 2*np.pi],  # Joint 3
            [-2*np.pi, 2*np.pi],  # Joint 4
            [-2*np.pi, 2*np.pi],  # Joint 5
            [-2*np.pi, 2*np.pi]   # Joint 6
        ]

        return RobotConfig(
            dh_params=dh_params,
            joint_types=joint_types,
            qlim=qlim,
            name="UR5",
            description="Universal Robots UR5 6-DOF Robot"
        )
    
    @staticmethod
    def add_custom_robot(dh_params: List[List[float]], 
                        joint_types: str,
                        qlim: List[List[float]],
                        name: str,
                        description: str = "") -> RobotConfig:
        """
        Create a custom robot configuration
        
        Args:
            dh_params: DH parameters [a, alpha, d, offset]
            joint_types: Joint types string (R for revolute, P for prismatic)
            qlim: Joint limits [[min, max], ...]
            name: Robot name
            description: Robot description
            
        Returns:
            RobotConfig object
        """
        return RobotConfig(
            dh_params=dh_params,
            joint_types=joint_types,
            qlim=qlim,
            name=name,
            description=description
        )