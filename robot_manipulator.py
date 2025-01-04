from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
from spatialmath import SE3
import numpy as np
import time
from ik_solvers import IKSolver

class RobotManipulator:
    def __init__(self, dh_params, joint_types, qlim):
        self.DH = np.array(dh_params)
        self.joint_types = joint_types
        self.qlim = np.array(qlim)
        self.robot = self._create_robot()
        self.solution_metrics = {}
        self.units = ['deg' if jtype == 'R' else 'mm' for jtype in joint_types]

    def _create_robot(self):
        try:
            links = []
            for i, (dh_params, joint_type, joint_limits) in enumerate(
                zip(self.DH, self.joint_types, self.qlim)):
                
                d, a, alpha, offset = dh_params
                
                if joint_type == "R":
                    link = RevoluteDH(
                        d=d,           # Link offset
                        a=a,          # Link length
                        alpha=alpha,  # Link twist
                        offset=offset,# Joint offset
                        qlim=joint_limits
                    )
                elif joint_type == "P":
                    link = PrismaticDH(
                        theta=offset, # Fixed rotation
                        a=a,         # Link length
                        alpha=alpha, # Link twist
                        qlim=joint_limits
                    )
                else:
                    raise ValueError(f"Invalid joint type '{joint_type}' at position {i}")
                    
                links.append(link)
            
            return DHRobot(links, name="Robot")
                
        except Exception as e:
            print(f"Robot creation error: {str(e)}")
            return None

    def forward_kinematics(self, joint_angles): 
            return self.robot.fkine(joint_angles)

    def solve_inverse_kinematics(self, target_position, method='all', **kwargs):
        solvers = {
            'ccd': IKSolver.ccd_solver,
            'fabrik': IKSolver.fabrik_solver,
            'jacobian': IKSolver.jacobian_solver,
            'dls': IKSolver.dls_solver,
            'newton': IKSolver.newton_raphson_solver    
        }
    
        if method not in solvers:
            raise ValueError(f"Invalid solution method: {method}")
    
        solver_func = solvers[method]
        start_time = time.time()
        
        solver_kwargs = {}
        if 'max_iter' in kwargs:
            solver_kwargs['max_iter'] = kwargs['max_iter']
        if 'tolerance' in kwargs:
            solver_kwargs['tolerance'] = kwargs['tolerance']
        
        if method == 'dls' and 'lambda_val' in kwargs:
            solver_kwargs['lambda_val'] = kwargs['lambda_val']
        elif method == 'jacobian' and 'alpha' in kwargs:
            solver_kwargs['alpha'] = kwargs['alpha']
        
        try:
            result = solver_func(self, target_position, **solver_kwargs)
            end_time = time.time()
    
            if result is not None:
                joint_angles, iterations = result
                if joint_angles is not None:
                    final_pos = self.forward_kinematics(joint_angles).t
                    error = np.linalg.norm(final_pos - target_position)
    
                    metrics = {
                        'joint_angles': np.array(joint_angles),
                        'iterations': int(iterations),
                        'time': float(end_time - start_time),
                        'error': float(error)
                    }
                    
                    self.solution_metrics = metrics
                    return metrics
    
            return None
        except Exception as e:
            print(f"Error in solve_inverse_kinematics: {str(e)}")
            return None
        
    def visualize(self, joint_angles, gui=None, block=True):
        try:
            if self.robot is None:
                print("Error: Robot not initialized properly")
                return
                
            if isinstance(joint_angles, (int, float)):
                joint_angles = np.zeros(len(self.joint_types))
                
            q = np.array(joint_angles, dtype=float)
            
            # Ölçekleme kaldırıldı - Prizmatik eklemler mm cinsinden kalacak
            print("\nJoint Values:")
            for i, (val, jtype) in enumerate(zip(q, self.joint_types)):
                print(f"Joint {i+1} ({jtype}): {val}")
            
            self.robot.plot(q, 
                        block=block,
                        jointaxes=True,
                        shadow=False,
                        eeframe=True)
                    
        except Exception as e:
            print(f"Visualization error: {str(e)}")

    def animate_trajectory(self, trajectory):
        try:
            if not isinstance(trajectory, np.ndarray):
                trajectory = np.array(trajectory)
            
            # Görselleştirme için yörüngeyi kopyala
            traj = np.array(trajectory, dtype=float)
            
            # Ölçekleme işlemini kaldır
            # Önceki kod:
            # for i, jtype in enumerate(self.joint_types):
            #     if jtype == 'P':
            #         traj[:, i] = trajectory[:, i] / 100.0
            
            # Robot animasyon özelliklerini ayarla
            self.robot.plot(traj,
                        backend='pyplot',
                        dt=0.05,
                        block=False,
                        eeframe=True,
                        jointaxes=True,
                        shadow=False)
            
        except Exception as e:
            print(f"\nAnimation error: {str(e)}")

    def print_solution_metrics(self):
        print("\nSolution Metrics:")
        print("-----------------")
        
        if not self.solution_metrics:
            print("No solution metrics available")
            return
            
        if isinstance(self.solution_metrics, dict):
            if 'joint_angles' in self.solution_metrics:
                metrics = self.solution_metrics
                if metrics is not None:
                    print(f"Iterations: {metrics.get('iterations', 'N/A')}")
                    print(f"Solution Time: {metrics.get('time', 0):.4f} seconds")
                    print(f"Target Distance: {metrics.get('error', 0):.4f} mm")
                    if metrics.get('joint_angles') is not None:
                        joint_values = np.array(metrics['joint_angles'])
                        print("Joint Values:")
                        for i, (val, unit, jtype) in enumerate(zip(joint_values, self.units, self.joint_types)):
                            if jtype == 'R':
                                print(f"Joint {i+1} (R): {np.degrees(val):.2f} deg")
                            else:
                                print(f"Joint {i+1} (P): {val:.2f} mm")
                    else:
                        print("Joint Values: None")