from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
from spatialmath import SE3
import numpy as np
import time

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

    def solve_inverse_kinematics(self, target_position, solvers=None, **kwargs):
        if solvers is None:
            from ik_solvers import IKSolver
            solvers = {
                'ccd': IKSolver.ccd_solver,
                'fabrik': IKSolver.fabrik_solver,
                'jacobian': IKSolver.jacobian_solver,
                'dls': IKSolver.dls_solver,
                'newton_raphson': IKSolver.newton_raphson_solver
            }

        results = {}
        for solver_name, solver_func in solvers.items():
            start_time = time.time()
            
            # Her solver için gerekli parametreleri ayır
            solver_kwargs = {}
            if 'max_iter' in kwargs:
                solver_kwargs['max_iter'] = kwargs['max_iter']
            if 'tolerance' in kwargs:
                solver_kwargs['tolerance'] = kwargs['tolerance']
            
            # Özel parametreler
            if solver_name == 'dls' and 'lambda_val' in kwargs:
                solver_kwargs['lambda_val'] = kwargs['lambda_val']
            elif solver_name == 'jacobian' and 'alpha' in kwargs:
                solver_kwargs['alpha'] = kwargs['alpha']
            
            result = solver_func(self, target_position, **solver_kwargs)
            end_time = time.time()

            if result is not None:
                joint_angles, iterations = result
                final_pos = self.forward_kinematics(joint_angles).t
                error = np.linalg.norm(final_pos - target_position)

                results[solver_name] = {
                    'joint_angles': joint_angles,
                    'iterations': iterations,
                    'time': end_time - start_time,
                    'error': error
                }

        self.solution_metrics = results
        return results

    def visualize(self, joint_angles, block=True):
        try:
            self.robot.teach(joint_angles, block=block)
        except Exception as e:
            print(f"Visualization error: {str(e)}")

    def print_solution_metrics(self):
        print("\nSolution Metrics:")
        print("-----------------")
        for method, metrics in self.solution_metrics.items():
            print(f"\n{method.upper()}:")
            print(f"Iterations: {metrics['iterations']}")
            print(f"Solution Time: {metrics['time']:.4f} seconds")
            print(f"Target Distance: {metrics['error']:.4f} mm")
            if metrics['joint_angles'] is not None:
                print(f"Joint Angles (degrees): {np.degrees(metrics['joint_angles'])}")
            else:
                print("Joint Angles: None")