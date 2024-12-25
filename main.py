from robot_manipulator import RobotManipulator
from robot_config import get_ur5_parameters, get_test_positions, get_solver_parameters
from ik_solvers import IKSolver
import numpy as np
import time

def interpolate_positions(start, end, steps):
    return [start + (end - start) * t for t in np.linspace(0, 1, steps)]

def test_ik_solvers():
    # Get robot parameters
    dh_params, joint_types, qlim = get_ur5_parameters()
    robot = RobotManipulator(dh_params, joint_types, qlim)
    
    # Get test positions and solver parameters
    test_positions = get_test_positions()
    solver_params = get_solver_parameters()
    
    # Number of interpolation steps between each pair of points
    steps = 100
    
    # Interpolate positions
    interpolated_positions = []
    for i in range(len(test_positions) - 1):
        start = np.array(test_positions[i])
        end = np.array(test_positions[i + 1])
        interpolated_positions.extend(interpolate_positions(start, end, steps))
    interpolated_positions.append(test_positions[-1])
    
    # Define solvers
    solvers = {
        # 'CCD': IKSolver.ccd_solver,
        # 'FABRIK': IKSolver.fabrik_solver,
        'Jacobian': IKSolver.jacobian_solver,
        'DLS': IKSolver.dls_solver
    }
    
    # Compare solver performances
    performance_metrics = {}
    
    
    for solver_name, solver_func in solvers.items():
        print(f"\nTesting {solver_name} solver...")
        start_time = time.time()
        total_iterations = 0
        total_error = 0
        solutions = []
        
        for target_pos in interpolated_positions:
            result = robot.solve_inverse_kinematics(
                target_pos,
                method=solver_name.lower(),
                **solver_params
            )
            
            if result and 'joint_angles' in result:
                solutions.append(result['joint_angles'])
                total_iterations += result['iterations']
                total_error += result['error']
            else:
                print(f"Solver {solver_name} failed for target position {target_pos}")
                break
        
        end_time = time.time()
        
        if solutions:
            performance_metrics[solver_name] = {
                'total_time': end_time - start_time,
                'avg_iterations': total_iterations / len(solutions),
                'avg_error': total_error / len(solutions),
                'success_rate': len(solutions) / len(interpolated_positions) * 100,
                'solutions': solutions
            }
            
            # Visualize the trajectory for this solver
            print(f"\nVisualizing {solver_name} solution trajectory...")
            qtraj = np.array(solutions)
            robot.animate_trajectory(qtraj)  # Will show continuous motion
            input(f"Press Enter to continue to next solver...")
    
    # Print comparison results
    print("\nSolver Performance Comparison:")
    print("=" * 50)
    for solver_name, metrics in performance_metrics.items():
        print(f"\n{solver_name}:")
        print(f"Total Time: {metrics['total_time']:.4f} seconds")
        print(f"Average Iterations: {metrics['avg_iterations']:.2f}")
        print(f"Average Error: {metrics['avg_error']:.6f}")
        print(f"Success Rate: {metrics['success_rate']:.1f}%")

def main():
    print("UR5 Robot Inverse Kinematics Solution Test Starting...")
    print("=" * 50)
    test_ik_solvers()

if __name__ == "__main__":
    main()