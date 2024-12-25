import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
import matplotlib.pyplot as plt
from spatialmath import SE3
import time

# Add project root to Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.robot_base import RobotBase
from src.core.robot_configs import RobotConfigs
from src.solvers.ccd_solver import CCDSolver
from src.solvers.fabrik_solver import FABRIKSolver
from src.solvers.jacobian_solver import JacobianSolver
from src.solvers.dls_solver import DLSSolver

def main():
    # Get UR5 configuration
    ur5_config = RobotConfigs.get_ur5_config()
    
    # Create robot instance
    robot = RobotBase(
        dh_params=ur5_config.dh_params,
        joint_types=ur5_config.joint_types,
        qlim=ur5_config.qlim,
        name=ur5_config.name
    )

    # Create target position
    target = np.array([400, 200, 200])
    
    # Test different solvers
    solvers = {
        'CCD': CCDSolver(robot),
        'FABRIK': FABRIKSolver(robot),
        'Jacobian': JacobianSolver(robot),
        'DLS': DLSSolver(robot)
    }

    # Parameters for solvers
    params = {
        'max_iter': 1000,
        'tolerance': 1e-4,
        'lambda_val': 0.1,  # for DLS
        'alpha': 0.5        # for Jacobian
    }

    # Test each solver
    for name, solver in solvers.items():
        print(f"\n{'-'*50}")
        print(f"Testing {name} solver...")
        print(f"{'-'*50}")
        
        # Solve IK
        joint_angles, iterations = solver.solve(target, **params)
        
        if joint_angles is not None:
            # Get final position
            final_pose = robot.forward_kinematics(joint_angles)
            final_pos = final_pose.t
            
            # Calculate error
            error = np.linalg.norm(target - final_pos)
            
            print(f"Success!")
            print(f"Final position: {final_pos}")
            print(f"Target position: {target}")
            print(f"Error: {error:.4f}")
            print(f"Iterations: {iterations}")
            
            # Plot robot
            env = robot.robot.plot(joint_angles, block=False)
            
            # Add text info to console
            print(f"\nSolver: {name}")
            print(f"Iterations: {iterations}")
            print(f"Error: {error:.4f}")
            
            user_input = input(f"\nPress Enter to continue to next solver (or 'q' to quit)...")
            if user_input.lower() == 'q':
                env.close()
                return
            
            env.close()
        else:
            print(f"Solver failed to find a solution")
    
    print("\nAll solvers tested.")

if __name__ == "__main__":
    main()