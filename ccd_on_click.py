import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
from spatialmath import SE3
import numpy as np

class MyRobot:
    def __init__(self, dh_table, joint_types, qlim=None, name="Robot"):
        self.robot = self.create_robot_from_dh(dh_table, joint_types, qlim, name)
        self.joints = self.robot.links

    def create_robot_from_dh(self, DH, joint_types, qlim=None, name="Robot"):
        links = []
        for i, row in enumerate(DH):
            a, alpha, d, offset = row
            joint_limit = qlim[i] if qlim else None
            if joint_types[i] == "R":
                link = RevoluteDH(a=a, alpha=alpha, d=d, offset=offset, qlim=joint_limit)
            elif joint_types[i] == "P":
                link = PrismaticDH(a=a, alpha=alpha, theta=offset, qlim=joint_limit)
            else:
                raise ValueError(f"Invalid joint type '{joint_types[i]}' at index {i}")
            links.append(link)
        return DHRobot(links, name=name)

    def get_joint_positions(self, q):
        TM_all = self.robot.fkine_all(q)
        joint_positions = [TM.t for TM in TM_all]
        return np.array(joint_positions)

    def get_end_effector_position(self, q):
        ee_pose = self.robot.fkine(q)
        return ee_pose.t

    def solve_ccd(self, target, max_iterations=40, tolerance=1e-3, visualize=False):
        q = np.zeros(len(self.joints))  # Initial joint configuration
        iteration_data = []  # Store iteration data for manual update
        joint_positions = self.get_joint_positions(q)
        ee_position = self.get_end_effector_position(q)

        for iteration in range(max_iterations):
            if np.linalg.norm(ee_position - target) < tolerance:
                break

            for i in reversed(range(len(self.joints))):
                joint_position = joint_positions[i]

                # Direction vectors
                direction_to_effector = ee_position - joint_position
                direction_to_goal = target - joint_position

                # Normalize directions
                direction_to_effector /= np.linalg.norm(direction_to_effector)
                direction_to_goal /= np.linalg.norm(direction_to_goal)

                # Calculate rotation axis and angle
                angle = np.arccos(np.clip(
                    np.dot(direction_to_effector, direction_to_goal), -1.0, 1.0))
                axis = np.cross(direction_to_effector, direction_to_goal)

                # Skip if the axis is near zero (no rotation needed)
                if np.linalg.norm(axis) < 1e-6:
                    continue
                axis /= np.linalg.norm(axis)

                transformation_matrix = self.robot.fkine_all(q)
                rotation_axis = transformation_matrix[i].a

                # Update joint angle for revolute joints
                if isinstance(self.joints[i], RevoluteDH):
                    q[i] += angle * np.sign(np.dot(axis, rotation_axis))
                    if self.joints[i].qlim is not None:
                        q[i] = np.clip(q[i], self.joints[i].qlim[0], self.joints[i].qlim[1])

                elif isinstance(self.joints[i], PrismaticDH):
                    vector_to_target = target - joint_position
                    translation_axis = np.array([0, 0, 1])  # Assuming z-axis
                    displacement = np.dot(vector_to_target, translation_axis)
                    displacement = np.clip(displacement, -np.linalg.norm(vector_to_target), np.linalg.norm(vector_to_target))

                    new_position = q[i] + displacement
                    q[i] = np.clip(new_position, self.joints[i].qlim[0], self.joints[i].qlim[1])

                # Update joint positions and end-effector position
                joint_positions = self.get_joint_positions(q)
                ee_position = joint_positions[-1]

                # Update end effector position
                iteration_data.append((iteration, i, q.copy(), ee_position.copy()))  # Store step data

        if visualize:
            self.visualize_steps_with_button(target, iteration_data)

        return q

    def visualize_steps_with_button(self, target, iteration_data):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.step_index = 0

        def on_click(event):
            if self.step_index < len(iteration_data):
                iteration, joint_index, q, ee_position = iteration_data[self.step_index]
                self._plot_robot(ax, target, f"Iteration {iteration}, Step {joint_index}",
                                 ee_position, q, active_joint_index=joint_index)
                self.step_index += 1
            else:
                print("All steps completed.")

        ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])  # Define button position
        btn = Button(ax_button, 'Next Step')
        btn.on_clicked(on_click)

        plt.show()

    def _plot_robot(self, ax, target, title, end_effector_position, q, active_joint_index=None):
        ax.clear()
        joint_positions = self.get_joint_positions(q)

        # Plot all joints and links
        for i in range(len(joint_positions) - 1):
            color = 'blue' if i == active_joint_index else 'orange'
            ax.plot([joint_positions[i, 0], joint_positions[i + 1, 0]],
                    [joint_positions[i, 1], joint_positions[i + 1, 1]],
                    [joint_positions[i, 2], joint_positions[i + 1, 2]], marker='o', color=color, linewidth=2)

        # Plot the target and end-effector positions
        ax.scatter(*target, color='r', label='Target', s=50)
        ax.scatter(*end_effector_position, color='g', label='End Effector', s=50)
        
        # Set plot limits and title
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_title(title)
        ax.legend()
        plt.draw()

    def add_robot_from_dh(self, dh_table):
        current_transform = np.eye(4)
        # Add your implementation here

# Example usage
# UR5 DH Table: [a, alpha, d, offset]
dh_table = np.array([
    [0, np.pi/2, 0.089159, 0],         # Joint 1
    [-0.425, 0, 0, 0],                 # Joint 2
    [-0.39225, 0, 0, 0],               # Joint 3
    [0, np.pi/2, 0.10915, 0],          # Joint 4
    [0, -np.pi/2, 0.09465, 0],         # Joint 5
    [0, 0, 0.0823, 0]                  # Joint 6
])
joint_types = "RRRRRR"
qlim = [
    [np.pi, np.pi],  # Joint 1 limits
    [-np.pi, np.pi],  # Joint 2 limits
    [-np.pi, np.pi],  # Joint 3 limits
    [-np.pi, np.pi],  # Joint 4 limits
    [-np.pi, np.pi],  # Joint 5 limits
    [-np.pi, np.pi]  # Joint 6 limits
]

robot = MyRobot(dh_table, joint_types, qlim)

# Example target position
target_position = np.array([0.1, 0.1, 0.1])

# Solve CCD and visualize
q_sol = robot.solve_ccd(target_position, visualize=True)
print("Solved Joint Angles:", q_sol)

print("End Effector TF:")
print(robot.robot.fkine(q_sol))

# Visualize the robot
robot.robot.plot(q_sol, block=True)
