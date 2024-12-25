from robot_manipulator import RobotManipulator
from robot_config import get_ur5_parameters, get_test_positions, get_solver_parameters, get_my_robot_parameters
from ik_solvers import IKSolver

def test_ik_solvers():
    # Get robot parameters
    dh_params, joint_types, qlim = get_ur5_parameters()
    robot = RobotManipulator(dh_params, joint_types, qlim)
    
    # Get test positions and solver parameters
    test_positions = get_test_positions()
    solver_params = get_solver_parameters()
    
    # Test each position
    for i, target_pos in enumerate(test_positions):
        print(f"\nTest {i+1}: Target Position = {target_pos}")
        print("=" * 50)
        
        # Solve using all methods
        results = robot.solve_inverse_kinematics(
            target_pos,
            **solver_params
        )
        
        # Print results
        robot.print_solution_metrics()
        
        # Visualize best solution
        if results:
            best_method = min(results.items(), key=lambda x: x[1]['error'])[0]
            print(f"\nVisualizing best solution ({best_method})...")
            robot.visualize(results[best_method]['joint_angles'])

def main():
    print("UR5 Robot Inverse Kinematics Solution Test Starting...")
    print("=" * 50)
    
    test_ik_solvers()

if __name__ == "__main__":
    main()
