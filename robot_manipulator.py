from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
from spatialmath import SE3
import numpy as np
import time
from ik_solvers import IKSolver  # IKSolver sınıfını içe aktar

class RobotManipulator:
    def __init__(self, dh_params, joint_types, qlim):
        self.DH = np.array(dh_params)
        self.joint_types = joint_types
        self.qlim = np.array(qlim)
        self.robot = self._create_robot()
        self.solution_metrics = {}
        
    def _create_robot(self):
        try:
            links = []
            for i, (dh_params, joint_type, joint_limits) in enumerate(
                zip(self.DH, self.joint_types, self.qlim)):
                
                a, alpha, d, offset = dh_params
                
                if joint_type == "R":
                    link = RevoluteDH(
                        a=a, alpha=alpha, d=d, offset=offset, qlim=joint_limits
                    )
                elif joint_type == "P":
                    link = PrismaticDH(
                        a=a, alpha=alpha, theta=offset, qlim=joint_limits
                    )
                else:
                    raise ValueError(f"Invalid joint type '{joint_type}' at position {i}")
                    
                links.append(link)
                
            return DHRobot(links, name="My_Robot")
            
        except Exception as e:
            print(f"Robot creation error: {str(e)}")
            return None

    def forward_kinematics(self, joint_angles):
        try:
            if not isinstance(joint_angles, np.ndarray):
                joint_angles = np.array(joint_angles)
                
            if joint_angles.size < len(self.joint_types):
                q = np.zeros(len(self.joint_types))
                q[:joint_angles.size] = joint_angles
            else:
                q = joint_angles
                
            return self.robot.fkine(q)
        except Exception as e:
            print(f"Forward kinematics error: {str(e)}")
            return None

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
        
        # Her solver için gerekli parametreleri ayır
        solver_kwargs = {}
        if 'max_iter' in kwargs:
            solver_kwargs['max_iter'] = kwargs['max_iter']
        if 'tolerance' in kwargs:
            solver_kwargs['tolerance'] = kwargs['tolerance']
        
        # Özel parametreler
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
        """Visualize a single robot configuration"""
        try:
            if isinstance(joint_angles, (int, float)):
                joint_angles = np.zeros(len(self.joint_types))
            self.robot.plot(joint_angles, block=block)
        except Exception as e:
            print(f"Visualization error: {str(e)}")

    def animate_trajectory(self, trajectory):
        """Animate robot through a trajectory of joint angles"""
        try:
            if not isinstance(trajectory, np.ndarray):
                trajectory = np.array(trajectory)
            
            self.robot.plot(trajectory,
                        backend='pyplot',
                        dt=0.05,
                        block=False,
                        eeframe=True,
                        jointaxes=True)
            
        except Exception as e:
            print(f"\nAnimation error: {str(e)}")

    def print_solution_metrics(self):
        print("\nSolution Metrics:")
        print("-----------------")
        
        if not self.solution_metrics:
            print("No solution metrics available")
            return
            
        # Handle single solution case
        if isinstance(self.solution_metrics, dict):
            if 'joint_angles' in self.solution_metrics:
                # Single solution metrics
                metrics = self.solution_metrics
                if metrics is not None:
                    print(f"Iterations: {metrics.get('iterations', 'N/A')}")
                    print(f"Solution Time: {metrics.get('time', 0):.4f} seconds")
                    print(f"Target Distance: {metrics.get('error', 0):.4f} mm")
                    if metrics.get('joint_angles') is not None:
                        joint_angles = np.array(metrics['joint_angles'])
                        print(f"Joint Angles (degrees): {np.degrees(joint_angles)}")
                    else:
                        print("Joint Angles: None")
            else:
                # Multiple solutions for different methods
                for method, metrics in self.solution_metrics.items():
                    print(f"\n{method.upper()}:")
                    if metrics is not None:
                        print(f"Iterations: {metrics.get('iterations', 'N/A')}")
                        print(f"Solution Time: {metrics.get('time', 0):.4f} seconds")
                        print(f"Target Distance: {metrics.get('error', 0):.4f} mm")
                        if metrics.get('joint_angles') is not None:
                            joint_angles = np.array(metrics['joint_angles'])
                            print(f"Joint Angles (degrees): {np.degrees(joint_angles)}")
                        else:
                            print("Joint Angles: None")