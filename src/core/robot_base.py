import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
from spatialmath import SE3
from typing import List, Tuple, Optional, Dict, Any

class RobotBase:
    """Base robot class that handles basic robot operations"""
    
    def __init__(self, 
                 dh_params: List[List[float]], 
                 joint_types: str, 
                 qlim: List[List[float]], 
                 name: str = "Generic_Robot"):
        """
        Initialize robot with DH parameters
        
        Args:
            dh_params: List of DH parameters [a, alpha, d, offset]
            joint_types: String of joint types ('R' for revolute, 'P' for prismatic)
            qlim: List of joint limits [[min, max], ...]
            name: Name of the robot
        """
        self.DH = np.array(dh_params)
        self.joint_types = joint_types
        self.qlim = np.array(qlim)
        self.name = name
        self.robot = self._create_robot()
        self._solution_metrics: Dict[str, Any] = {}

    def _create_robot(self) -> Optional[DHRobot]:
        """Create robot using robotics-toolbox"""
        try:
            links = []
            for i, (dh_params, joint_type, joint_limits) in enumerate(
                zip(self.DH, self.joint_types, self.qlim)):
                
                a, alpha, d, offset = dh_params
                
                if joint_type == "R":
                    link = RevoluteDH(
                        a=a, 
                        alpha=alpha, 
                        d=d, 
                        offset=offset, 
                        qlim=joint_limits
                    )
                elif joint_type == "P":
                    link = PrismaticDH(
                        a=a, 
                        alpha=alpha, 
                        theta=offset,
                        qlim=joint_limits
                    )
                else:
                    raise ValueError(f"Invalid joint type '{joint_type}' at position {i}")
                    
                links.append(link)
                
            return DHRobot(links, name=self.name)
            
        except Exception as e:
            print(f"Error creating robot: {str(e)}")
            return None

    def forward_kinematics(self, joint_angles: np.ndarray) -> Optional[SE3]:
        """
        Calculate forward kinematics
        
        Args:
            joint_angles: Joint angles in radians
            
        Returns:
            SE3 transformation matrix if successful, None otherwise
        """
        try:
            if not isinstance(joint_angles, np.ndarray):
                joint_angles = np.array(joint_angles)
                
            if joint_angles.size < len(self.joint_types):
                q = np.zeros(len(self.joint_types))
                q[:joint_angles.size] = joint_angles
            else:
                q = joint_angles
                
            return self.robot.fkine(q)
            
        except Exception as e:
            print(f"Forward kinematics error: {str(e)}")
            return None

    def get_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Calculate robot Jacobian
        
        Args:
            joint_angles: Joint angles in radians
            
        Returns:
            Jacobian matrix
        """
        return self.robot.jacob0(joint_angles)

    @property
    def num_joints(self) -> int:
        """Get number of joints"""
        return len(self.joint_types)

    @property
    def solution_metrics(self) -> Dict[str, Any]:
        """Get solution metrics"""
        return self._solution_metrics

    @solution_metrics.setter
    def solution_metrics(self, metrics: Dict[str, Any]):
        """Set solution metrics"""
        self._solution_metrics = metrics

    def visualize(self, joint_angles: np.ndarray, block: bool = True):
        """
        Visualize robot at given joint angles
        
        Args:
            joint_angles: Joint angles in radians
            block: Whether to block execution while figure is shown
        """
        try:
            self.robot.teach(joint_angles, block=block)
        except Exception as e:
            print(f"Visualization error: {str(e)}")

    def print_metrics(self):
        """Print solution metrics"""
        print("\nSolution Metrics:")
        print("-" * 20)
        for method, metrics in self._solution_metrics.items():
            print(f"\n{method.upper()} Method:")
            print(f"Iterations: {metrics['iterations']}")
            print(f"Solve Time: {metrics['time']:.4f} seconds")
            print(f"Target Error: {metrics['error']:.4f} units")
            print(f"Joint Angles (deg): {np.degrees(metrics['joint_angles'])}")