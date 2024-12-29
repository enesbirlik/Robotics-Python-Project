import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import numpy as np
from robot_manipulator import RobotManipulator
from robot_config import get_ur5_parameters, get_solver_parameters, get_custom_robot_parameters
import json
import os

class ModernRobotGUI:
    def __init__(self):
        # Ana tema ayarları
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Ana pencere ayarları
        self.root = ctk.CTk()
        self.root.title("Modern Robot IK Solver")
        self.root.geometry("1200x800")
        
        # Default robot parametreleri
        self.dh_params, self.joint_types, self.qlim = get_custom_robot_parameters()
        self.num_joints = len(self.joint_types)
        self.solver_params = get_solver_parameters()
        self.trajectory_points = []
        
        # Ana layout
        self.create_main_layout()
        
        # Robot'u başlat
        self.init_robot()

    def create_main_layout(self):
        # Ana container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Sol panel (Robot Config ve Control Panel)
        self.left_panel = ctk.CTkFrame(self.main_container)
        self.left_panel.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Sağ panel (Metrikler ve Log)
        self.right_panel = ctk.CTkFrame(self.main_container)
        self.right_panel.pack(side="right", fill="both", padx=5, pady=5)
        
        self.create_robot_config_panel()
        self.create_control_panel()
        self.create_metrics_panel()
        self.create_log_panel()

    def create_robot_config_panel(self):
        config_frame = ctk.CTkFrame(self.left_panel)
        config_frame.pack(fill="x", padx=5, pady=5)
        
        # Başlık
        title = ctk.CTkLabel(config_frame, text="Robot Configuration",
                            font=ctk.CTkFont(size=16, weight="bold"))
        title.pack(pady=5)
        
        # Joint sayısı seçimi
        joint_frame = ctk.CTkFrame(config_frame)
        joint_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(joint_frame, text="Number of Joints:").pack(side="left", padx=5)
        self.joint_count = ctk.CTkOptionMenu(joint_frame,
                                           values=[str(i) for i in range(2, 11)],
                                           command=self.update_dh_table)
        self.joint_count.pack(side="left", padx=5)
        self.joint_count.set("6")
        
        # DH Parameter tablosu
        self.dh_frame = ctk.CTkScrollableFrame(config_frame, height=200)
        self.dh_frame.pack(fill="x", padx=5, pady=5)
        
        # Tablo başlıkları
        headers = ['Joint Type', 'a', 'alpha', 'd', 'offset', 'q_min', 'q_max']
        for j, header in enumerate(headers):
            ctk.CTkLabel(self.dh_frame, text=header,
                        font=ctk.CTkFont(weight="bold")).grid(row=0, column=j, padx=2)
        
        self.create_dh_table()
        
        # Update ve Save/Load butonları
        button_frame = ctk.CTkFrame(config_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkButton(button_frame, text="Update Robot",
                     command=self.update_robot).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Save Config",
                     command=self.save_config).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Load Config",
                     command=self.load_config).pack(side="left", padx=5)

    def create_control_panel(self):
        control_frame = ctk.CTkFrame(self.left_panel)
        control_frame.pack(fill="x", padx=5, pady=5)
        
        # Başlık
        title = ctk.CTkLabel(control_frame, text="Control Panel",
                            font=ctk.CTkFont(size=16, weight="bold"))
        title.pack(pady=5)
        
        # Solver seçimi
        solver_frame = ctk.CTkFrame(control_frame)
        solver_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(solver_frame, text="Solver:").pack(side="left", padx=5)
        self.solver_var = ctk.CTkOptionMenu(solver_frame,
                                          values=["jacobian", "dls", "ccd", "fabrik", "newton"])
        self.solver_var.pack(side="left", padx=5)
        
        # Target position frame
        target_frame = ctk.CTkFrame(control_frame)
        target_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(target_frame, text="Target Position",
                    font=ctk.CTkFont(weight="bold")).pack(pady=5)
        
        # X, Y, Z inputs
        self.target_entries = {}
        for coord, default in zip(['X', 'Y', 'Z'], [300, 300, 300]):
            coord_frame = ctk.CTkFrame(target_frame)
            coord_frame.pack(fill="x", padx=5, pady=2)
            
            ctk.CTkLabel(coord_frame, text=f"{coord}:").pack(side="left", padx=5)
            entry = ctk.CTkEntry(coord_frame, width=100)
            entry.pack(side="left", padx=5)
            entry.insert(0, str(default))
            self.target_entries[coord.lower()] = entry
        
        # Point Management Frame
        point_frame = ctk.CTkFrame(control_frame)
        point_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(point_frame, text="Trajectory Points",
                    font=ctk.CTkFont(weight="bold")).pack(pady=5)
        
        # Point listbox
        self.point_listbox = ctk.CTkTextbox(point_frame, height=150)
        self.point_listbox.pack(fill="x", padx=5, pady=5)
        
        # Point control buttons
        point_btn_frame = ctk.CTkFrame(point_frame)
        point_btn_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkButton(point_btn_frame, text="Add Point",
                     command=self.add_point).pack(side="left", padx=5)
        ctk.CTkButton(point_btn_frame, text="Clear Points",
                     command=self.clear_points).pack(side="left", padx=5)
        
        # Main control buttons
        button_frame = ctk.CTkFrame(control_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkButton(button_frame, text="Solve IK",
                     command=self.solve_ik).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Animate Trajectory",
                     command=self.animate_trajectory).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Reset",
                     command=self.reset_robot).pack(side="left", padx=5)

    def create_metrics_panel(self):
        metrics_frame = ctk.CTkFrame(self.right_panel)
        metrics_frame.pack(fill="x", padx=5, pady=5)
        
        # Başlık
        title = ctk.CTkLabel(metrics_frame, text="Performance Metrics",
                            font=ctk.CTkFont(size=16, weight="bold"))
        title.pack(pady=5)
        
        # Metrik labelları
        self.iterations_var = ctk.CTkLabel(metrics_frame, text="Iterations: -")
        self.iterations_var.pack(pady=2)
        
        self.time_var = ctk.CTkLabel(metrics_frame, text="Time: -")
        self.time_var.pack(pady=2)
        
        self.error_var = ctk.CTkLabel(metrics_frame, text="Error: -")
        self.error_var.pack(pady=2)

    def create_log_panel(self):
        log_frame = ctk.CTkFrame(self.right_panel)
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Başlık
        title = ctk.CTkLabel(log_frame, text="Log",
                            font=ctk.CTkFont(size=16, weight="bold"))
        title.pack(pady=5)
        
        # Log text area
        self.log_text = ctk.CTkTextbox(log_frame, height=200)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

    def create_dh_table(self):
        self.dh_entries = []
        num_joints = int(self.joint_count.get())
        
        for i in range(num_joints):
            row_entries = {}
            
            # Joint type selection
            joint_type = ctk.CTkOptionMenu(self.dh_frame, values=['R', 'P'], width=70)
            joint_type.grid(row=i+1, column=0, padx=2, pady=2)
            joint_type.set(self.joint_types[i] if i < len(self.joint_types) else 'R')
            row_entries['type'] = joint_type
            
            # DH parameters
            for j in range(4):
                entry = ctk.CTkEntry(self.dh_frame, width=70)
                entry.grid(row=i+1, column=j+1, padx=2, pady=2)
                value = self.dh_params[i][j] if i < len(self.dh_params) else 0.0
                entry.insert(0, f"{value:.2f}")
                row_entries[f'dh{j}'] = entry
            
            # Joint limits
            for j in range(2):
                entry = ctk.CTkEntry(self.dh_frame, width=70)
                entry.grid(row=i+1, column=j+5, padx=2, pady=2)
                value = self.qlim[i][j] if i < len(self.qlim) else (-6.28 if j == 0 else 6.28)
                entry.insert(0, f"{value:.2f}")
                row_entries[f'limit{j}'] = entry
            
            self.dh_entries.append(row_entries)

    def update_dh_table(self, _=None):
        # Mevcut tabloyu temizle
        for widget in self.dh_frame.grid_slaves():
            if int(widget.grid_info()["row"]) > 0:
                widget.destroy()
        
        self.num_joints = int(self.joint_count.get())
        self.create_dh_table()

    def update_robot(self):
        try:
            new_dh_params = []
            new_joint_types = ""
            new_qlim = []
            
            for entries in self.dh_entries:
                new_joint_types += entries['type'].get()
                dh_row = [float(entries[f'dh{i}'].get()) for i in range(4)]
                new_dh_params.append(dh_row)
                limits = [float(entries[f'limit{i}'].get()) for i in range(2)]
                new_qlim.append(limits)
            
            self.dh_params = new_dh_params
            self.joint_types = new_joint_types
            self.qlim = new_qlim
            self.init_robot()
            
            self.log_message("Robot updated successfully")
            
        except ValueError as e:
            self.log_message(f"Error updating robot: Invalid parameter value")

    def add_point(self):
        try:
            point = [
                float(self.target_entries['x'].get()),
                float(self.target_entries['y'].get()),
                float(self.target_entries['z'].get())
            ]
            self.trajectory_points.append(point)
            
            point_str = f"Point {len(self.trajectory_points)}: ({point[0]}, {point[1]}, {point[2]})\n"
            self.point_listbox.insert("end", point_str)
            self.log_message(f"Added point: {point}")
            
        except ValueError:
            self.log_message("Invalid coordinate values")

    def clear_points(self):
        self.trajectory_points = []
        self.point_listbox.delete("1.0", "end")
        self.log_message("Cleared all trajectory points")

    def interpolate_trajectory(self, steps=50):
        if len(self.trajectory_points) < 2:
            self.log_message("Need at least 2 points for trajectory")
            return None
            
        interpolated_points = []
        for i in range(len(self.trajectory_points) - 1):
            start = np.array(self.trajectory_points[i])
            end = np.array(self.trajectory_points[i + 1])
            
            for t in np.linspace(0, 1, steps):
                point = start + (end - start) * t
                interpolated_points.append(point)
                
        return interpolated_points

    def solve_trajectory(self):
        interpolated_points = self.interpolate_trajectory()
        if not interpolated_points:
            return None
            
        solutions = []
        for point in interpolated_points:
            result = self.robot.solve_inverse_kinematics(
                point,
                method=self.solver_var.get(),
                **self.solver_params
            )
            
            if result and 'joint_angles' in result:
                solutions.append(result['joint_angles'])
            else:
                self.log_message(f"Failed to solve IK for point {point}")
                return None
                
        return np.array(solutions)

    def init_robot(self):
        self.robot = RobotManipulator(self.dh_params, self.joint_types, self.qlim)
        self.reset_robot()
        self.log_message("Robot initialized")

    def solve_ik(self):
        try:
            target_pos = np.array([
                float(self.target_entries['x'].get()),
                float(self.target_entries['y'].get()),
                float(self.target_entries['z'].get())
            ])
            
            result = self.robot.solve_inverse_kinematics(
                target_pos,
                method=self.solver_var.get(),
                **self.solver_params
            )
            
            if result:
                self.update_metrics(result)
                self.visualize_solution(result['joint_angles'])
                self.log_message("IK solved successfully")
            else:
                self.log_message("No solution found")
                
        except Exception as e:
            self.log_message(f"Error solving IK: {str(e)}")

    def animate_trajectory(self):
        if not self.trajectory_points:
            self.log_message("No trajectory points defined")
            return
            
        self.log_message("Solving trajectory...")
        solutions = self.solve_trajectory()
        
        if solutions is not None:
            self.log_message("Animating trajectory...")
            self.robot.animate_trajectory(solutions)
        else:
            self.log_message("Failed to solve trajectory")

    def reset_robot(self):
        self.robot.visualize(np.zeros(len(self.robot.joint_types)))
        self.update_metrics(None)
        self.log_message("Robot reset to initial position")

    def update_metrics(self, result):
        if result:
            self.iterations_var.configure(text=f"Iterations: {result['iterations']}")
            self.time_var.configure(text=f"Time: {result['time']:.4f} s")
            self.error_var.configure(text=f"Error: {result['error']:.6f} mm")
        else:
            self.iterations_var.configure(text="Iterations: -")
            self.time_var.configure(text="Time: -")
            self.error_var.configure(text="Error: -")

    def visualize_solution(self, joint_angles):
        self.robot.visualize(joint_angles)
        
    def save_config(self):
        try:
            config = {
                'dh_params': self.dh_params,
                'joint_types': self.joint_types,
                'qlim': self.qlim
            }
            with open('robot_config.json', 'w') as f:
                json.dump(config, f)
            self.log_message("Configuration saved successfully")
        except Exception as e:
            self.log_message(f"Error saving configuration: {str(e)}")

    def load_config(self):
        try:
            if os.path.exists('robot_config.json'):
                with open('robot_config.json', 'r') as f:
                    config = json.load(f)
                self.dh_params = config['dh_params']
                self.joint_types = config['joint_types']
                self.qlim = config['qlim']
                self.num_joints = len(self.joint_types)
                self.joint_count.set(str(self.num_joints))
                self.create_dh_table()
                self.init_robot()
                self.log_message("Configuration loaded successfully")
            else:
                self.log_message("No saved configuration found")
        except Exception as e:
            self.log_message(f"Error loading configuration: {str(e)}")

    def log_message(self, message):
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")

def main():
    app = ModernRobotGUI()
    app.root.mainloop()

if __name__ == "__main__":
    main()