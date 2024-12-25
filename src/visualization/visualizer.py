# src/visualization/visualizer.py
import matplotlib.pyplot as plt
from spatialmath import SE3

class Visualizer:
    @staticmethod
    def plot_solution(robot, q, title="Robot Configuration"):
        plt.figure()
        robot.visualize(q)
        end_effector_pose = robot.forward_kinematics(q)
        if end_effector_pose:
            end_effector_pos = end_effector_pose.t
            plt.scatter(end_effector_pos[0], end_effector_pos[1], end_effector_pos[2], color='r', label='End-Effector')
            plt.text(end_effector_pos[0], end_effector_pos[1], end_effector_pos[2], f"({end_effector_pos[0]:.2f}, {end_effector_pos[1]:.2f}, {end_effector_pos[2]:.2f})", color='r')
        plt.title(title)
        plt.legend()
        plt.show()

    @staticmethod
    def compare_solutions(robot, solutions):
        num_solutions = len(solutions)
        fig = plt.figure(figsize=(5*num_solutions, 5))
        
        for i, (name, q) in enumerate(solutions.items(), 1):
            ax = fig.add_subplot(1, num_solutions, i, projection='3d')
            robot.visualize(q)
            end_effector_pose = robot.forward_kinematics(q)
            if end_effector_pose:
                end_effector_pos = end_effector_pose.t
                ax.scatter(end_effector_pos[0], end_effector_pos[1], end_effector_pos[2], color='r', label='End-Effector')
                ax.text(end_effector_pos[0], end_effector_pos[1], end_effector_pos[2], f"({end_effector_pos[0]:.2f}, {end_effector_pos[1]:.2f}, {end_effector_pos[2]:.2f})", color='r')
            ax.set_title(f'{name} Solution')
            ax.legend()
        
        plt.show()