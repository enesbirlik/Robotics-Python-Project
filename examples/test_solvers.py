import sys
import os

# Proje kök dizinini sys.path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')  # Matplotlib arka ucunu değiştir

from src.core.robot_configs import RobotConfigs
from src.core.robot_base import RobotBase
from src.solvers.ccd_solver import CCDSolver
from src.solvers.fabrik_solver import FABRIKSolver
from src.solvers.jacobian_solver import JacobianSolver
from src.solvers.dls_solver import DLSSolver
from src.visualization.visualizer import Visualizer

def test_solvers():
    robot_config = RobotConfigs.get_ur5_config()
    robot = RobotBase(
        dh_params=robot_config.dh_params,
        joint_types=robot_config.joint_types,
        qlim=robot_config.qlim,
        name=robot_config.name
    )

    visualizer = Visualizer()
    
    test_positions = [
        [50, 50, 50],
        [55, 55, 55],
        [60, 60, 60]
    ]
    
    solver_params = {
        'max_iter': 1000,
        'tolerance': 1e-3,
        'lambda_val': 0.2,
        'alpha': 0.5
    }
    
    solvers = {
        'CCD': CCDSolver(robot),
        'FABRIK': FABRIKSolver(robot),
        'Jacobian': JacobianSolver(robot),
        'DLS': DLSSolver(robot)
    }
    
    for i, target_pos in enumerate(test_positions):
        print(f"\nTest {i+1}: Hedef Pozisyon = {target_pos}")
        solutions = {}
        
        for name, solver in solvers.items():
            q, iterations = solver.solve(target_pos, **solver_params)
            if q is not None:
                solutions[name] = q
        
        visualizer.compare_solutions(robot, solutions)

if __name__ == "__main__":
    test_solvers()