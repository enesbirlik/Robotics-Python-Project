import numpy as np
from abc import ABC, abstractmethod

class IKSolver(ABC):
    def __init__(self, robot):
        self.robot = robot
    
    @abstractmethod
    def solve(self, target_pose, initial_guess=None):
        pass
    
    def check_convergence(self, current_pos, target_pos, tolerance):
        return np.linalg.norm(current_pos - target_pos) < tolerance