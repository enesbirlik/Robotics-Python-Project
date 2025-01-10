from robot_manipulator import RobotManipulator
from robot_config import get_scara_parameters, get_test_positions, get_solver_parameters,get_custom_robot_parameters
from ik_solvers import IKSolver
import numpy as np
import time

def interpolate_positions(start, end, steps):
    return [start + (end - start) * t for t in np.linspace(0, 1, steps)]

def test_ik_solvers():
    # Get robot parameters
    dh_params, joint_types, qlim = get_scara_parameters()
    robot = RobotManipulator(dh_params, joint_types, qlim)
    
    print("\nRobot Configuration:")
    print("=" * 50)
    print(f"Joint Types: {joint_types}")
    print("\nDH Parameters:")
    for i, params in enumerate(dh_params):
        print(f"Joint {i}: {params}")
    print("\nJoint Limits:")
    for i, limits in enumerate(qlim):
        print(f"Joint {i}: {limits}")
    
    # Test forward kinematics at zero configuration
    q_zero = np.zeros(len(joint_types))
    fk_zero = robot.forward_kinematics(q_zero)
    print("\nInitial Configuration:")
    print(f"Joint angles (zero): {q_zero}")
    print(f"Initial end-effector position: {fk_zero.t}")
    
    # Get test positions and solver parameters
    test_positions = get_test_positions()
    solver_params = get_solver_parameters()
    
    print("\nGoal Positions:")
    for i, pos in enumerate(test_positions):
        print(f"Goal {i+1}: {pos}")
    
    # Number of interpolation steps between each pair of points
    steps = 5
    
    # Interpolate positions
    interpolated_positions = []
    for i in range(len(test_positions) - 1):
        start = np.array(test_positions[i])
        end = np.array(test_positions[i + 1])
        interpolated_positions.extend(interpolate_positions(start, end, steps))
    interpolated_positions.append(test_positions[-1])
    
    print("\nInterpolated Positions:")
    for i, pos in enumerate(interpolated_positions):
        print(f"Point {i+1}: {pos}")
    
    # Define solvers
    solvers = {
        'Jacobian': IKSolver.jacobian_solver,
        'DLS': IKSolver.dls_solver,
        'Newton': IKSolver.newton_raphson_solver,
        # 'CCD': IKSolver.ccd_cozer_solver,
        # 'FABRIK': IKSolver.fabrik_solver,
    }
    
    # Compare solver performances
    performance_metrics = {}
    
    for solver_name, solver_func in solvers.items():
        print(f"\nTesting {solver_name} solver...")
        print("=" * 50)
        start_time = time.time()
        total_iterations = 0
        total_error = 0
        solutions = []
        
        for i, target_pos in enumerate(interpolated_positions):
            print(f"\nTarget {i+1}: {target_pos}")
            
            result = robot.solve_inverse_kinematics(
                target_pos,
                method=solver_name.lower(),
                **solver_params
            )
            
            if result and 'joint_angles' in result:
                q_sol = result['joint_angles']
                solutions.append(q_sol)
                total_iterations += result['iterations']
                total_error += result['error']
                
                # Forward kinematics check
                fk_result = robot.forward_kinematics(q_sol)
                print(f"Joint angles: {np.degrees(q_sol) if isinstance(q_sol, np.ndarray) else q_sol}")
                print(f"Forward kinematics result: {fk_result.t}")
                print(f"Position error: {np.linalg.norm(fk_result.t - target_pos):.6f}")
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
            
            # Print final positions
            print(f"\nFinal Results for {solver_name}:")
            print("-" * 30)
            for i, q_sol in enumerate(solutions):
                fk_result = robot.forward_kinematics(q_sol)
                print(f"\nSolution {i+1}:")
                print(f"Target: {interpolated_positions[i]}")
                print(f"Joint angles (deg): {np.degrees(q_sol)}")
                print(f"Achieved position: {fk_result.t}")
                print(f"Error: {np.linalg.norm(fk_result.t - interpolated_positions[i]):.6f}")
            
            # Visualize the trajectory
            print(f"\nVisualizing {solver_name} solution trajectory...")
            qtraj = np.array(solutions)
            robot.animate_trajectory(qtraj)
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
    print("Robot Inverse Kinematics Solution Test Starting...")
    print("=" * 50)
    test_ik_solvers()

if __name__ == "__main__":
    main()