import matplotlib.pyplot as plt
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

    def solve_ccd(self, target, max_iterations=900, tolerance=1e-3, visualize=False):
        q = np.zeros(len(self.joints))  # Initial joint configuration
        joint_positions = self.get_joint_positions(q)
        ee_position = joint_positions[-1]  # Use the last joint position as the end-effector position

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
                    joint_position = joint_positions[i+1]
                    
                    # Calculate the vector from the joint to the target
                    vector_to_target = target - ee_position
                    
                    # Translate to the joint's local Z-axis (assume joint's translation axis is the Z-axis)
                    translation_axis = transformation_matrix[i].a  # Extract Z-axis from transformation matrix

                    # Calculate displacement along the joint's translation axis
                    displacement = np.dot(vector_to_target, translation_axis)

                    # Update the joint position with the displacement
                    new_position = q[i] + displacement

                    # Clip the position within joint limits
                    if self.joints[i].qlim is not None:
                        q[i] = np.clip(new_position, self.joints[i].qlim[0], self.joints[i].qlim[1])
                    else:
                        q[i] = new_position


                # Update joint positions and end-effector position
                joint_positions = self.get_joint_positions(q)
                ee_position = joint_positions[-1]

            if visualize and iteration % 1 == 0:
                ax = plt.gca(projection='3d')
                self._plot_robot(ax, target, iteration, ee_position, q)

        if visualize:
            plt.ioff()
            plt.show()

        return q


    def _plot_robot(self, ax, target, iteration, end_effector_position, q):
        ax.clear()
        joint_positions = self.get_joint_positions(q)
        ax.plot(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], marker='o', color='orange', linewidth=2)
        ax.scatter(*target, color='r', label='Target', s=50)
        ax.scatter(*end_effector_position, color='g', label='End Effector', s=50)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_title(f"Iteration {iteration}")
        ax.legend()
        plt.draw()
        plt.pause(0.01)

    def add_robot_from_dh(self, dh_table):
        current_transform = np.eye(4)
        # Add your implementation here


# Example usage
import numpy as np

# DH Table: [a, alpha, d, offset]
dh_table = np.array([
    [0.3, 0, 0, np.pi/2],         # Joint 1
    [0.4, 0, 0, 0],               # Joint 2
    [0.3, 0, 0, 0],               # Joint 3
    [0, -np.pi/2, -0.2, np.pi/2], # Joint 4 (Prismatic)
    [0, np.pi/2, 0, np.pi/2],     # Joint 5
    [0.2, 0, 0, 0],               # Joint 6
    [0, np.pi/2, 0, np.pi/2],     # Joint 7
    [0, -np.pi/2, 0, np.pi/2],    # Joint 8
    [-0.25, 0, 0.1, 0],           # Joint 9 (Prismatic)
    [-0.3, 0, 0, 0],              # Joint 10
])

# Joint types: "R" for revolute, "P" for prismatic
joint_types = "RRRPRRRRPR"

# Joint limits
qlim = [
    [-np.pi, np.pi],       # Joint 1 limits
    [-np.pi/2, np.pi/2],   # Joint 2 limits
    [-np.pi, np.pi],       # Joint 3 limits
    [0.0, 0.4],            # Joint 4 limits (Prismatic, 0 to 0.4 meters)
    [-np.pi/2, np.pi/2],   # Joint 5 limits
    [-np.pi, np.pi],       # Joint 6 limits
    [-np.pi, np.pi],       # Joint 7 limits
    [-np.pi, np.pi],       # Joint 8 limits
    [0.0, 0.2],            # Joint 9 limits (Prismatic, 0 to 0.2 meters)
    [-np.pi, np.pi],       # Joint 10 limits
]


# qlim = [None, None, None, None, None, None] # No joint limits

robot = MyRobot(dh_table, joint_types, qlim)

# Example target position
target_position = np.array([0.9, 1.1, 1.0])

# Solve CCD and visualize
q_sol = robot.solve_ccd(target_position, visualize=True)
print("Solved Joint Angles:", q_sol)
print("End Effector Position:")
print(robot.robot.fkine(q_sol))

# Visualize the robot
robot.robot.plot(q_sol, block=True)
