import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from robot_manipulator import RobotManipulator
from robot_config import get_ur5_parameters, get_solver_parameters

class RobotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Inverse Kinematics Solver")
        
        # Initialize with default parameters
        self.dh_params, self.joint_types, self.qlim = get_ur5_parameters()
        self.num_joints = len(self.joint_types)
        self.solver_params = get_solver_parameters()
        
        # Configure grid weights
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Create main containers
        self.left_panel = ttk.Frame(self.root)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.create_robot_config_panel()
        self.create_control_panel()
        self.create_visualization_panel()
        self.create_metrics_panel()
        
        # Initialize robot
        self.init_robot()
        
    def create_robot_config_panel(self):
        """Create panel for robot configuration"""
        config_frame = ttk.LabelFrame(self.left_panel, text="Robot Configuration", padding="5")
        config_frame.pack(fill="x", expand=False, padx=5, pady=5)
        
        # Joint count selection
        ttk.Label(config_frame, text="Number of Joints:").grid(row=0, column=0, sticky="w")
        self.joint_count = tk.IntVar(value=6)
        joint_spin = ttk.Spinbox(config_frame, from_=2, to=10, width=5,
                                textvariable=self.joint_count,
                                command=self.update_dh_table)
        joint_spin.grid(row=0, column=1, sticky="w", padx=5)
        
        # DH parameter table
        self.dh_frame = ttk.LabelFrame(config_frame, text="DH Parameters", padding="5")
        self.dh_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=5)
        
        # Create DH table headers
        headers = ['Joint Type', 'a', 'alpha', 'd', 'offset', 'q_min', 'q_max']
        for i, header in enumerate(headers):
            ttk.Label(self.dh_frame, text=header).grid(row=0, column=i, padx=2)
        
        self.dh_entries = []
        self.create_dh_table()
        
        # Update button
        ttk.Button(config_frame, text="Update Robot", 
                  command=self.update_robot).grid(row=2, column=0, columnspan=2, pady=5)
    
    def create_dh_table(self):
        """Create input fields for DH parameters"""
        # Clear existing entries
        for widget in self.dh_frame.grid_slaves():
            if int(widget.grid_info()["row"]) > 0:  # Preserve headers
                widget.destroy()
        self.dh_entries.clear()
        
        for i in range(self.num_joints):
            row_entries = {}
            
            # Joint type selection
            joint_type = ttk.Combobox(self.dh_frame, values=['R', 'P'], width=5)
            joint_type.set(self.joint_types[i])
            joint_type.grid(row=i+1, column=0, padx=2)
            row_entries['type'] = joint_type
            
            # DH parameters
            dh = self.dh_params[i]
            for j, value in enumerate(dh):
                entry = ttk.Entry(self.dh_frame, width=8)
                entry.insert(0, f"{value:.2f}")
                entry.grid(row=i+1, column=j+1, padx=2)
                row_entries[f'dh{j}'] = entry
            
            # Joint limits
            for j, value in enumerate(self.qlim[i]):
                entry = ttk.Entry(self.dh_frame, width=8)
                entry.insert(0, f"{value:.2f}")
                entry.grid(row=i+1, column=j+5, padx=2)
                row_entries[f'limit{j}'] = entry
            
            self.dh_entries.append(row_entries)
    
    def create_control_panel(self):
        """Create control panel with solver selection and target position"""
        control_frame = ttk.LabelFrame(self.left_panel, text="Control Panel", padding="5")
        control_frame.pack(fill="x", expand=False, padx=5, pady=5)
        
        # Solver selection
        ttk.Label(control_frame, text="Solver:").grid(row=0, column=0, sticky="w")
        self.solver_var = tk.StringVar(value="jacobian")
        solver_combo = ttk.Combobox(control_frame, textvariable=self.solver_var)
        solver_combo['values'] = ('jacobian', 'dls', 'ccd', 'fabrik', 'newton')
        solver_combo.grid(row=0, column=1, sticky="ew", padx=5)
        
        # Target position inputs
        ttk.Label(control_frame, text="Target Position:").grid(row=1, column=0, sticky="w", pady=5)
        pos_frame = ttk.Frame(control_frame)
        pos_frame.grid(row=1, column=1, sticky="ew", pady=5)
        
        self.target_x = tk.DoubleVar(value=300)
        self.target_y = tk.DoubleVar(value=300)
        self.target_z = tk.DoubleVar(value=300)
        
        ttk.Label(pos_frame, text="X:").grid(row=0, column=0)
        ttk.Entry(pos_frame, textvariable=self.target_x, width=8).grid(row=0, column=1)
        ttk.Label(pos_frame, text="Y:").grid(row=0, column=2)
        ttk.Entry(pos_frame, textvariable=self.target_y, width=8).grid(row=0, column=3)
        ttk.Label(pos_frame, text="Z:").grid(row=0, column=4)
        ttk.Entry(pos_frame, textvariable=self.target_z, width=8).grid(row=0, column=5)
        
        # Control buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        ttk.Button(btn_frame, text="Solve IK", command=self.solve_ik).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Animate", command=self.animate_solution).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Reset", command=self.reset_robot).pack(side="left", padx=5)
    
    def create_visualization_panel(self):
        """Create robot visualization panel"""
        viz_frame = ttk.LabelFrame(self.root, text="Robot Visualization", padding="5")
        viz_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=5, pady=5)
        
        # Create figure and canvas for matplotlib
        self.figure = Figure(figsize=(8, 8))
        self.ax = self.figure.add_subplot(111, projection='3d')
        
        # Create canvas and add it to frame
        self.canvas = FigureCanvasTkAgg(self.figure, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_metrics_panel(self):
        """Create performance metrics panel"""
        metrics_frame = ttk.LabelFrame(self.left_panel, text="Performance Metrics", padding="5")
        metrics_frame.pack(fill="x", expand=False, padx=5, pady=5)
        
        # Metrics labels
        self.iterations_var = tk.StringVar(value="Iterations: -")
        self.time_var = tk.StringVar(value="Time: -")
        self.error_var = tk.StringVar(value="Error: -")
        
        ttk.Label(metrics_frame, textvariable=self.iterations_var).pack(anchor="w")
        ttk.Label(metrics_frame, textvariable=self.time_var).pack(anchor="w")
        ttk.Label(metrics_frame, textvariable=self.error_var).pack(anchor="w")
    
    def update_dh_table(self):
        """Update DH table when joint count changes"""
        old_num_joints = len(self.dh_params)
        self.num_joints = self.joint_count.get()
        
        # If increasing joints, extend parameters
        if self.num_joints > old_num_joints:
            for _ in range(self.num_joints - old_num_joints):
                self.dh_params.append([0, 0, 0, 0])
                self.joint_types += "R"
                self.qlim.append([-2*np.pi, 2*np.pi])
        # If decreasing joints, truncate parameters
        else:
            self.dh_params = self.dh_params[:self.num_joints]
            self.joint_types = self.joint_types[:self.num_joints]
            self.qlim = self.qlim[:self.num_joints]
        
        # Recreate table
        self.create_dh_table()
    
    def update_robot(self):
        """Update robot with current DH parameters"""
        # Get values from entries
        new_dh_params = []
        new_joint_types = ""
        new_qlim = []
        
        for entries in self.dh_entries:
            # Get joint type
            new_joint_types += entries['type'].get()
            
            # Get DH parameters
            dh_row = [float(entries[f'dh{i}'].get()) for i in range(4)]
            new_dh_params.append(dh_row)
            
            # Get joint limits
            limits = [float(entries[f'limit{i}'].get()) for i in range(2)]
            new_qlim.append(limits)
        
        # Update robot
        self.dh_params = new_dh_params
        self.joint_types = new_joint_types
        self.qlim = new_qlim
        self.init_robot()
    
    def init_robot(self):
        """Initialize or reinitialize robot"""
        self.robot = RobotManipulator(self.dh_params, self.joint_types, self.qlim)
        self.reset_robot()
    
    def solve_ik(self):
        """Solve inverse kinematics with selected solver"""
        target_pos = np.array([
            self.target_x.get(),
            self.target_y.get(),
            self.target_z.get()
        ])
        
        result = self.robot.solve_inverse_kinematics(
            target_pos,
            method=self.solver_var.get(),
            **self.solver_params
        )
        
        if result:
            self.update_metrics(result)
            self.visualize_solution(result['joint_angles'])
    
    def animate_solution(self):
        """Animate the current solution"""
        if hasattr(self, 'current_solution'):
            self.robot.animate_trajectory(self.current_solution, gui=self)
    
    def reset_robot(self):
        """Reset robot to initial position"""
        self.robot.visualize(np.zeros(len(self.robot.joint_types)), gui=self)
        self.update_metrics(None)
    
    def update_metrics(self, result):
        """Update performance metrics display"""
        if result:
            self.iterations_var.set(f"Iterations: {result['iterations']}")
            self.time_var.set(f"Time: {result['time']:.4f} s")
            self.error_var.set(f"Error: {result['error']:.6f} mm")
            self.current_solution = result['joint_angles']
        else:
            self.iterations_var.set("Iterations: -")
            self.time_var.set("Time: -")
            self.error_var.set("Error: -")
    
    def visualize_solution(self, joint_angles):
        """Visualize robot configuration"""
        self.robot.visualize(joint_angles, gui=self)

def main():
    root = tk.Tk()
    app = RobotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()