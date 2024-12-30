import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import numpy as np
from robot_manipulator import RobotManipulator
from robot_config import get_ur5_parameters, get_solver_parameters, get_scara_parameters, get_kr6_parameters, get_custom_robot_parameters
import json
import os

class ModernRobotGUI:
    def __init__(self):
        # Ana tema ayarları
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        
        # Ana pencere ayarları
        self.root = ctk.CTk()
        self.root.title("Python Robot IK Solver")
        self.root.geometry("1020x800")
        
        # Robot instance'ı
        self.dh_params, self.joint_types, self.qlim = get_custom_robot_parameters()
        self.robot = RobotManipulator(self.dh_params, self.joint_types, self.qlim)
            
        self.num_joints = len(self.joint_types)
        self.solver_params = get_solver_parameters()
        self.trajectory_points = []
        
        # Solution metrikleri için dictionary
        self.solution_metrics = {}
        
        # Ana layout
        self.create_main_layout()
        
        # İlk görselleştirme
        self.init_robot()

    def create_main_layout(self):
        # Ana container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Sol panel
        self.left_panel = ctk.CTkFrame(self.main_container, width=600)
        self.left_panel.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.left_panel.pack_propagate(False)
        
        # Sağ panel
        self.right_panel = ctk.CTkFrame(self.main_container, width=300)
        self.right_panel.pack(side="right", fill="y", padx=5, pady=5)
        self.right_panel.pack_propagate(False)
        
        self.create_robot_config_panel()
        self.create_control_panel()
        self.create_metrics_panel()
        self.create_log_panel()

    def create_robot_config_panel(self):
        config_frame = ctk.CTkFrame(self.left_panel)
        config_frame.pack(fill="x", padx=5, pady=5)
        
        # Üst kısım - Robot seçimi ve joint sayısı
        top_frame = ctk.CTkFrame(config_frame)
        top_frame.pack(fill="x", padx=5, pady=5)
        
        # Robot seçimi
        robot_frame = ctk.CTkFrame(top_frame)
        robot_frame.pack(side="left", padx=5)
        
        ctk.CTkLabel(robot_frame, text="Robot Type:",
                    font=ctk.CTkFont(size=14)).pack(side="left", padx=5)
        self.robot_type = ctk.CTkOptionMenu(robot_frame,
                                        values=['Custom', 'UR5', 'SCARA', 'KR6'],
                                        command=self.load_robot_config,
                                        font=ctk.CTkFont(size=14),
                                        width=120)
        self.robot_type.pack(side="left", padx=5)
        self.robot_type.set('Custom')
        
        # Joint sayısı seçimi
        joint_frame = ctk.CTkFrame(top_frame)
        joint_frame.pack(side="left", padx=5)
        
        ctk.CTkLabel(joint_frame, text="Number of Joints:",
                    font=ctk.CTkFont(size=14)).pack(side="left", padx=5)
        self.joint_count = ctk.CTkOptionMenu(joint_frame,
                                        values=[str(i) for i in range(2, 11)],
                                        command=self.create_dh_table,
                                        font=ctk.CTkFont(size=14),
                                        width=100)
        self.joint_count.pack(side="left", padx=5)
        self.joint_count.set(str(self.num_joints))
        
        # Tablo başlıkları için frame
        table_frame = ctk.CTkFrame(config_frame)
        table_frame.pack(fill="x", padx=5, pady=5)
        
        # Grid yapılandırması
        table_frame.grid_columnconfigure((0,1,2,3,4,5,6), weight=1)
        
        # Başlıklar
        headers = ['Joint Type', 'a', 'alpha', 'd', 'offset', 'q_min', 'q_max']
        column_widths = [70, 70, 70, 70, 70, 70, 70]  # Her sütun için genişlik
        
        for j, (header, width) in enumerate(zip(headers, column_widths)):
            label = ctk.CTkLabel(table_frame, 
                            text=header,
                            font=ctk.CTkFont(size=12, weight="bold"),
                            width=width)
            label.grid(row=0, column=j, padx=2, pady=2, sticky="ew")
        
        # DH tablosu için scrollable frame
        self.dh_frame = ctk.CTkScrollableFrame(config_frame, height=300)
        self.dh_frame.pack(fill="x", padx=5, pady=5)
        
        # DH frame için grid yapılandırması
        for i in range(7):  # 7 sütun için
            self.dh_frame.grid_columnconfigure(i, weight=1)
        
        self.create_dh_table()

    def create_control_panel(self):
        control_frame = ctk.CTkFrame(self.left_panel)
        control_frame.pack(fill="x", padx=5, pady=5)
        
        # Solver seçimi ve target position aynı satırda
        input_frame = ctk.CTkFrame(control_frame)
        input_frame.pack(fill="x", padx=5, pady=5)
        
        # Sol taraf - Solver seçimi
        solver_frame = ctk.CTkFrame(input_frame)
        solver_frame.pack(side="left", padx=5)
        
        ctk.CTkLabel(solver_frame, text="Solver:",
                    font=ctk.CTkFont(size=14)).pack(side="left", padx=5)
        self.solver_var = ctk.CTkOptionMenu(solver_frame,
                                        values=['dls', 'jacobian', 'ccd', 'fabrik', 'newton'],
                                        font=ctk.CTkFont(size=14),
                                        width=100)
        self.solver_var.pack(side="left", padx=5)
        self.solver_var.set('dls')
        
        # Sağ taraf - Target Position
        target_frame = ctk.CTkFrame(input_frame)
        target_frame.pack(side="right", fill="x", expand=True, padx=5)
        
        ctk.CTkLabel(target_frame, text="Target Position (mm)",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=5)
        
        self.target_entries = {}
        for coord in ['x', 'y', 'z']:
            coord_frame = ctk.CTkFrame(target_frame)
            coord_frame.pack(side="left", padx=5)
            
            ctk.CTkLabel(coord_frame, text=f"{coord.upper()}:",
                        font=ctk.CTkFont(size=14)).pack(side="left", padx=2)
            entry = ctk.CTkEntry(coord_frame, width=80, font=ctk.CTkFont(size=14))
            entry.pack(side="left", padx=2)
            entry.insert(0, "0")
            self.target_entries[coord] = entry
        
        # Butonlar
        button_frame = ctk.CTkFrame(control_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        # Sol taraf - Ana kontrol butonları
        main_buttons = ctk.CTkFrame(button_frame)
        main_buttons.pack(side="left", padx=5)
        
        ctk.CTkButton(main_buttons, text="Solve IK",
                    command=self.solve_ik,
                    font=ctk.CTkFont(size=14),
                    width=120).pack(side="left", padx=2)
        ctk.CTkButton(main_buttons, text="Reset Robot",
                    command=self.reset_robot,
                    font=ctk.CTkFont(size=14),
                    width=120).pack(side="left", padx=2)
        
        # Sağ taraf - Trajectory butonları
        traj_buttons = ctk.CTkFrame(button_frame)
        traj_buttons.pack(side="right", padx=5)
        
        ctk.CTkButton(traj_buttons, text="Add Point",
                    command=self.add_point,
                    font=ctk.CTkFont(size=14),
                    width=120).pack(side="left", padx=2)
        ctk.CTkButton(traj_buttons, text="Clear Points",
                    command=self.clear_points,
                    font=ctk.CTkFont(size=14),
                    width=120).pack(side="left", padx=2)
        ctk.CTkButton(traj_buttons, text="Animate",
                    command=self.animate_trajectory,
                    font=ctk.CTkFont(size=14),
                    width=120).pack(side="left", padx=2)
        
        # Trajectory noktaları listesi
        traj_list_frame = ctk.CTkFrame(control_frame)
        traj_list_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(traj_list_frame, text="Trajectory Points",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        # İki sütunlu liste görünümü
        list_container = ctk.CTkFrame(traj_list_frame)
        list_container.pack(fill="both", expand=True)
        
        # Sol taraf - Nokta numaraları
        self.point_numbers = ctk.CTkTextbox(list_container, height=150,
                                        width=50, font=ctk.CTkFont(size=12))
        self.point_numbers.pack(side="left", fill="y", padx=(5,0), pady=5)
        
        # Sağ taraf - Koordinatlar
        self.point_listbox = ctk.CTkTextbox(list_container, height=150,
                                        font=ctk.CTkFont(size=12))
        self.point_listbox.pack(side="left", fill="both", expand=True, padx=5, pady=5)

    def create_metrics_panel(self):
        metrics_frame = ctk.CTkFrame(self.right_panel)
        metrics_frame.pack(fill="x", padx=5, pady=5)
        
        # Başlık
        title = ctk.CTkLabel(metrics_frame, text="Performance Metrics",
                            font=ctk.CTkFont(size=16, weight="bold"))
        title.pack(pady=10)
        
        # Metrik labelları
        self.iterations_var = ctk.CTkLabel(metrics_frame, text="Iterations: -",
                                         font=ctk.CTkFont(size=14))
        self.iterations_var.pack(pady=5)
        
        self.time_var = ctk.CTkLabel(metrics_frame, text="Time: -",
                                    font=ctk.CTkFont(size=14))
        self.time_var.pack(pady=5)
        
        self.error_var = ctk.CTkLabel(metrics_frame, text="Error: -",
                                     font=ctk.CTkFont(size=14))
        self.error_var.pack(pady=5)

    def create_log_panel(self):
        log_frame = ctk.CTkFrame(self.right_panel)
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Başlık
        title = ctk.CTkLabel(log_frame, text="Log",
                            font=ctk.CTkFont(size=16, weight="bold"))
        title.pack(pady=5)
        
        # Log text area
        self.log_text = ctk.CTkTextbox(log_frame, height=400,
                                      font=ctk.CTkFont(size=12))
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

    def create_dh_table(self, event=None):
        # Önce mevcut girişleri temizle
        for widget in self.dh_frame.winfo_children():
            widget.destroy()
                
        self.dh_entries = []
        num_joints = int(self.joint_count.get())
            
        for i in range(num_joints):
            row_entries = {}
                
            # Joint type selection
            joint_type = ctk.CTkOptionMenu(self.dh_frame, values=['R', 'P'], width=70)
            joint_type.grid(row=i+1, column=0, padx=2, pady=2)
            joint_type.set(self.joint_types[i] if i < len(self.joint_types) else 'R')
            joint_type.configure(command=lambda t=joint_type, idx=i: self.update_joint_limits(t, idx))
            row_entries['type'] = joint_type
                
            # DH parameters
            for j in range(4):
                entry = ctk.CTkEntry(self.dh_frame, width=70)
                entry.grid(row=i+1, column=j+1, padx=2, pady=2)
                value = self.dh_params[i][j] if i < len(self.dh_params) else 0.0
                entry.insert(0, f"{value:.2f}")
                entry.bind('<FocusOut>', lambda e, idx=i: self.update_robot_config())
                row_entries[f'dh{j}'] = entry
                
            # Joint limits
            for j in range(2):
                entry = ctk.CTkEntry(self.dh_frame, width=70)
                entry.grid(row=i+1, column=j+5, padx=2, pady=2)
                value = self.qlim[i][j] if i < len(self.qlim) else (-6.28 if j == 0 else 6.28)
                entry.insert(0, f"{value:.2f}")
                entry.bind('<FocusOut>', lambda e, idx=i: self.update_robot_config())
                row_entries[f'limit{j}'] = entry
                
            self.dh_entries.append(row_entries)

    def update_robot_config(self):
        """DH tablosundaki değişiklikleri robota uygula"""
        try:
            new_dh_params = []
            new_joint_types = ""
            new_qlim = []
                
            for entries in self.dh_entries:
                # Joint tipi
                new_joint_types += entries['type'].get()
                    
                # DH parametreleri
                dh_row = []
                for i in range(4):
                    try:
                        value = float(entries[f'dh{i}'].get())
                        dh_row.append(value)
                    except ValueError:
                        self.log_message(f"Invalid DH parameter value")
                        return
                new_dh_params.append(dh_row)
                    
                # Eklem limitleri
                limits = []
                for i in range(2):
                    try:
                        value = float(entries[f'limit{i}'].get())
                        limits.append(value)
                    except ValueError:
                        self.log_message(f"Invalid joint limit value")
                        return
                new_qlim.append(limits)
                
            # Değerleri güncelle
            self.dh_params = new_dh_params
            self.joint_types = new_joint_types
            self.qlim = new_qlim
                
            # Robot'u güncelle
            self.robot = RobotManipulator(self.dh_params, self.joint_types, self.qlim)
            self.reset_robot()
                
            self.log_message("Robot configuration updated successfully")
                
        except Exception as e:
            self.log_message(f"Error updating robot configuration: {str(e)}")


    def update_joint_limits(self, joint_type_menu, index):
        """Joint tipine göre limit birimlerini güncelle"""
        try:
            joint_type = joint_type_menu.get()
            entries = self.dh_entries[index]
                
            if joint_type == 'R':
                # Revolute joint için radyan cinsinden default limitler
                if float(entries['limit0'].get()) == 0.0 and float(entries['limit1'].get()) == 100.0:
                    entries['limit0'].delete(0, 'end')
                    entries['limit1'].delete(0, 'end')
                    entries['limit0'].insert(0, str(-np.pi))
                    entries['limit1'].insert(0, str(np.pi))
            else:
                # Prismatic joint için mm cinsinden default limitler
                if abs(float(entries['limit0'].get())) > 10 or abs(float(entries['limit1'].get())) > 10:
                    entries['limit0'].delete(0, 'end')
                    entries['limit1'].delete(0, 'end')
                    entries['limit0'].insert(0, "0.0")
                    entries['limit1'].insert(0, "100.0")
                    
            # Robot konfigürasyonunu güncelle
            self.update_robot_config()
                
        except Exception as e:
            self.log_message(f"Error updating joint limits: {str(e)}")

    def load_robot_config(self, robot_type):
        try:
            if robot_type == 'Custom':
                self.dh_params, self.joint_types, self.qlim = get_custom_robot_parameters()
            elif robot_type == 'UR5':
                self.dh_params, self.joint_types, self.qlim = get_ur5_parameters()
            elif robot_type == 'SCARA':
                self.dh_params, self.joint_types, self.qlim = get_scara_parameters()
            elif robot_type == 'KR6':
                self.dh_params, self.joint_types, self.qlim = get_kr6_parameters()
                
            # Joint sayısını güncelle
            self.num_joints = len(self.joint_types)
            self.joint_count.set(str(self.num_joints))
                
            # Önce DH tablosunu güncelle
            self.create_dh_table()
                
            # Robot'u güncelle
            self.robot = RobotManipulator(self.dh_params, self.joint_types, self.qlim)
            self.reset_robot()
                
            self.log_message(f"Loaded {robot_type} configuration")
                
        except Exception as e:
            self.log_message(f"Error loading robot configuration: {str(e)}")

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
            
        except Exception as e:
            self.log_message(f"Error updating robot: {str(e)}")

    def init_robot(self):
        """Initialize or reinitialize robot"""
        try:
            self.robot = RobotManipulator(self.dh_params, self.joint_types, self.qlim)
            self.reset_robot()
            self.log_message("Robot initialized")
        except Exception as e:
            self.log_message(f"Error initializing robot: {str(e)}")

    def reset_robot(self):
        """Reset robot to initial position"""
        try:
            initial_angles = np.zeros(len(self.joint_types))
            self.robot.visualize(initial_angles)
            self.update_metrics(None)
            self.log_message("Robot reset to initial position")
        except Exception as e:
            self.log_message(f"Error resetting robot: {str(e)}")

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
            
            if result and 'joint_angles' in result:
                joint_values = result['joint_angles']
                self.robot.visualize(joint_values, block=False)
                
                # Metrikleri güncelle
                self.update_metrics(result)
                
                # Log mesajını joint tiplerine göre oluştur
                log_msg = "Solution found:\n"
                for i, (val, jtype) in enumerate(zip(joint_values, self.robot.joint_types)):
                    if jtype == 'R':
                        log_msg += f"Joint {i+1} (R): {np.degrees(val):.2f}°\n"
                    else:
                        log_msg += f"Joint {i+1} (P): {val:.2f}mm\n"
                
                self.log_message(log_msg)
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

    def update_metrics(self, result):
        if result:
            self.iterations_var.configure(text=f"Iterations: {result['iterations']}")
            if 'time' in result:
                self.time_var.configure(text=f"Time: {result['time']:.4f} s")
            if 'error' in result:
                self.error_var.configure(text=f"Error: {result['error']:.2f} mm")
        else:
            self.iterations_var.configure(text="Iterations: -")
            self.time_var.configure(text="Time: -")
            self.error_var.configure(text="Error: -")

    def add_point(self):
        try:
            point = [
                float(self.target_entries['x'].get()),
                float(self.target_entries['y'].get()),
                float(self.target_entries['z'].get())
            ]
            self.trajectory_points.append(point)
            
            # Nokta numarası ve koordinatları ayrı ayrı ekle
            point_num = len(self.trajectory_points)
            self.point_numbers.insert("end", f"#{point_num}\n")
            self.point_listbox.insert("end", f"({point[0]}, {point[1]}, {point[2]}) mm\n")
            
            self.log_message(f"Added point {point_num}: {point} mm")
            
        except ValueError:
            self.log_message("Invalid coordinate values")

    def clear_points(self):
        self.trajectory_points = []
        self.point_numbers.delete("1.0", "end")
        self.point_listbox.delete("1.0", "end")
        self.log_message("Cleared all trajectory points")

    def interpolate_trajectory(self, base_steps=50):
        if len(self.trajectory_points) < 2:
            self.log_message("Need at least 2 points for trajectory")
            return None
            
        interpolated_points = []
        distances = []
        
        # Calculate distances between consecutive points
        for i in range(len(self.trajectory_points) - 1):
            start = np.array(self.trajectory_points[i])
            end = np.array(self.trajectory_points[i + 1])
            distance = np.linalg.norm(end - start)
            distances.append(distance)
        
        # Calculate total path length
        total_distance = sum(distances)
        
        # Interpolate points with proportional steps
        for i in range(len(self.trajectory_points) - 1):
            start = np.array(self.trajectory_points[i])
            end = np.array(self.trajectory_points[i + 1])
            
            # Calculate proportional steps based on segment distance
            segment_steps = max(
                int((distances[i] / total_distance) * base_steps),
                30  # Minimum steps per segment
            )
            
            for t in np.linspace(0, 1, segment_steps):
                point = start + (end - start) * t
                interpolated_points.append(point)
        
        self.log_message(f"Generated trajectory with {len(interpolated_points)} points")
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

    def log_message(self, message):
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")

def deg2rad(deg):
    """Dereceyi radyana çevirir"""
    return deg * np.pi / 180

def rad2deg(rad):
    """Radyanı dereceye çevirir"""
    return rad * 180 / np.pi

def main():
    app = ModernRobotGUI()
    app.root.mainloop()

if __name__ == "__main__":
    main()