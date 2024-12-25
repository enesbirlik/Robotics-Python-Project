import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
from spatialmath import SE3
from scipy.linalg import pinv
import time

    
class IKSolver:
    @staticmethod
    def ccd_solver(robot, target_position, max_iter=100, tolerance=1e-3):
        """Cyclic Coordinate Descent yöntemi"""
        try:
            q = np.zeros(len(robot.joint_types))
            
            for iteration in range(max_iter):
                current_pose = robot.forward_kinematics(q)
                if current_pose is None:
                    continue
                    
                end_effector = current_pose.t
                error = np.linalg.norm(end_effector - target_position)
                
                if error < tolerance:
                    return q, iteration
                
                # Eklemler üzerinde geriye doğru döngü
                for i in range(len(q)-1, -1, -1):
                    # Mevcut eklem pozisyonunu al
                    pivot = robot.forward_kinematics(q[:i+1]).t if i >= 0 else np.zeros(3)
                    
                    # Vektörleri hesapla
                    current_to_target = target_position - pivot
                    current_to_end = end_effector - pivot
                    
                    # Vektörlerin büyüklüğünü kontrol et
                    if np.linalg.norm(current_to_target) < 1e-6 or np.linalg.norm(current_to_end) < 1e-6:
                        continue
                    
                    # Vektörleri normalize et
                    current_to_target = current_to_target / np.linalg.norm(current_to_target)
                    current_to_end = current_to_end / np.linalg.norm(current_to_end)
                    
                    # Rotasyon açısını hesapla
                    dot_product = np.clip(np.dot(current_to_end, current_to_target), -1.0, 1.0)
                    angle = np.arccos(dot_product)
                    
                    # Rotasyon yönünü belirle
                    cross_product = np.cross(current_to_end, current_to_target)
                    rotation_axis = np.array([0, 0, 1])  # Z ekseni etrafında rotasyon
                    if np.dot(cross_product, rotation_axis) < 0:
                        angle = -angle
                    
                    # Açıyı sınırla ve uygula
                    q[i] = np.clip(q[i] + angle, robot.qlim[i][0], robot.qlim[i][1])
                    
                    # Yeni uç efektör pozisyonunu hesapla
                    new_pose = robot.forward_kinematics(q)
                    if new_pose is not None:
                        end_effector = new_pose.t
            
            return q, max_iter
            
        except Exception as e:
            print(f"CCD çözüm hatası: {str(e)}")
            return None, max_iter

    @staticmethod
    def fabrik_solver(robot, target_position, max_iter=100, tolerance=1e-3):
        """Forward And Backward Reaching Inverse Kinematics yöntemi"""
        try:
            num_joints = len(robot.joint_types)
            q = np.zeros(num_joints)
            
            # Başlangıç joint pozisyonlarını hesapla
            joint_positions = []
            initial_pose = robot.forward_kinematics(q)
            if initial_pose is None:
                return None, max_iter
                
            base_pos = np.zeros(3)
            end_effector = initial_pose.t
            
            # Link uzunluklarını hesapla
            link_lengths = []
            for i in range(num_joints):
                pos1 = robot.forward_kinematics(q[:i+1]).t if i > 0 else base_pos
                pos2 = robot.forward_kinematics(q[:i+2]).t if i < num_joints-1 else end_effector
                link_lengths.append(np.linalg.norm(pos2 - pos1))
            
            # Toplam uzunluğu hesapla
            total_length = sum(link_lengths)
            
            # Hedef noktanın ulaşılabilir olup olmadığını kontrol et
            target_distance = np.linalg.norm(target_position - base_pos)
            if target_distance > total_length:
                print("Hedef nokta ulaşılamaz mesafede")
                return None, max_iter
            
            # FABRIK iterasyonları
            for iteration in range(max_iter):
                # İleri aşama
                current_pos = end_effector
                error = np.linalg.norm(current_pos - target_position)
                
                if error < tolerance:
                    # Açıları hesapla
                    for i in range(num_joints-1):
                        v1 = joint_positions[i+1] - joint_positions[i]
                        v2 = joint_positions[i+2] - joint_positions[i+1]
                        
                        if np.linalg.norm(v1) > 1e-10 and np.linalg.norm(v2) > 1e-10:
                            angle = np.arctan2(np.cross(v1, v2)[2], np.dot(v1, v2))
                            q[i] = np.clip(angle, robot.qlim[i][0], robot.qlim[i][1])
                    
                    return q, iteration
                
                # Geriye doğru hesaplama
                new_positions = [target_position]
                for i in range(num_joints-1, -1, -1):
                    direction = (new_positions[0] - current_pos)
                    distance = link_lengths[i]
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction) * distance
                    new_positions.insert(0, new_positions[0] - direction)
                
                # İleri doğru hesaplama
                new_positions[0] = base_pos
                for i in range(1, len(new_positions)):
                    direction = (new_positions[i] - new_positions[i-1])
                    distance = link_lengths[i-1]
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction) * distance
                    new_positions[i] = new_positions[i-1] + direction
                
                # Joint pozisyonlarını güncelle
                joint_positions = new_positions
                
                # Yeni açıları hesapla
                for i in range(num_joints):
                    if i < num_joints-1:
                        v1 = joint_positions[i+1] - joint_positions[i]
                        v2 = joint_positions[i+2] - joint_positions[i+1]
                        if np.linalg.norm(v1) > 1e-10 and np.linalg.norm(v2) > 1e-10:
                            angle = np.arctan2(np.cross(v1, v2)[2], np.dot(v1, v2))
                            q[i] = np.clip(angle, robot.qlim[i][0], robot.qlim[i][1])
            
            return None, max_iter
            
        except Exception as e:
            print(f"FABRIK çözüm hatası: {str(e)}")
            return None, max_iter


    @staticmethod
    def jacobian_solver(robot, target_position, max_iter=100, tolerance=1e-3, alpha=0.5):
        """Jacobian bazlı yöntem"""
        try:
            q = np.zeros(len(robot.joint_types))
            
            for iteration in range(max_iter):
                current_pose = robot.forward_kinematics(q)
                if current_pose is None:
                    return None, max_iter
                    
                current_pos = current_pose.t
                error = target_position - current_pos
                error_norm = np.linalg.norm(error)
                
                if error_norm < tolerance:
                    return q, iteration
                
                # Jacobian hesaplama
                J = robot.robot.jacob0(q)[:3, :]
                
                # Pseudo-inverse çözüm
                J_pinv = pinv(J)
                dq = alpha * np.dot(J_pinv, error)
                
                # Açı güncelleme
                q = q + dq
                
                # Limit kontrolü
                for i in range(len(q)):
                    q[i] = np.clip(q[i], robot.qlim[i][0], robot.qlim[i][1])
            
            return q, iteration
            
        except Exception as e:
            print(f"Jacobian çözüm hatası: {str(e)}")
            return None, max_iter

    @staticmethod
    def dls_solver(robot, target_position, lambda_val=0.1, max_iter=100, tolerance=1e-3):
        """Damped Least Squares yöntemi"""
        try:
            q = np.zeros(len(robot.joint_types))
            I = np.eye(3)  # 3x3 birim matris (pozisyon için)
            
            for iteration in range(max_iter):
                current_pose = robot.forward_kinematics(q)
                if current_pose is None:
                    return None, max_iter
                    
                current_pos = current_pose.t
                error = target_position - current_pos
                error_norm = np.linalg.norm(error)
                
                if error_norm < tolerance:
                    return q, iteration
                
                # Jacobian hesaplama
                J = robot.robot.jacob0(q)[:3, :]
                
                # DLS çözümü
                JT = J.T
                dq = np.dot(JT, np.linalg.solve(np.dot(J, JT) + lambda_val**2 * I, error))
                
                # Açı güncelleme
                q = q + dq
                
                # Limit kontrolü
                for i in range(len(q)):
                    q[i] = np.clip(q[i], robot.qlim[i][0], robot.qlim[i][1])
            
            return q, iteration
            
        except Exception as e:
            print(f"DLS çözüm hatası: {str(e)}")
            return None, max_iter

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
                        a=a, 
                        alpha=alpha, 
                        d=d, 
                        offset=offset, 
                        qlim=joint_limits
                    )
                elif joint_type == "P":
                    link = PrismaticDH(
                        a=a, 
                        alpha=alpha, 
                        theta=offset,
                        qlim=joint_limits
                    )
                else:
                    raise ValueError(f"Geçersiz eklem tipi '{joint_type}' pozisyon {i}'de")
                    
                links.append(link)
                
            return DHRobot(links, name="My_Robot")
            
        except Exception as e:
            print(f"Robot oluşturma hatası: {str(e)}")
            return None

    def solve_inverse_kinematics(self, target_position, method='all', **kwargs):
        """
        Farklı ters kinematik çözüm yöntemlerini çağıran fonksiyon
        """
        def call_solver(solver_func, solver_name, target_position, kwargs):
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
                
            joint_angles, iterations = solver_func(self, target_position, **solver_kwargs)
            end_time = time.time()
            
            if joint_angles is not None:
                final_pos = self.forward_kinematics(joint_angles).t
                error = np.linalg.norm(final_pos - target_position)
                
                return {
                    'joint_angles': joint_angles,
                    'iterations': iterations,
                    'time': end_time - start_time,
                    'error': error
                }
            return None

        solvers = {
            'ccd': IKSolver.ccd_solver,
            'fabrik': IKSolver.fabrik_solver,
            'jacobian': IKSolver.jacobian_solver,
            'dls': IKSolver.dls_solver
        }
        
        if method == 'all':
            results = {}
            for solver_name, solver_func in solvers.items():
                result = call_solver(solver_func, solver_name, target_position, kwargs)
                if result is not None:
                    results[solver_name] = result
            
            self.solution_metrics = results
            return results
            
        elif method in solvers:
            result = call_solver(solvers[method], method, target_position, kwargs)
            if result is not None:
                self.solution_metrics = {method: result}
                return result
        else:
            raise ValueError(f"Geçersiz çözüm yöntemi: {method}")

    def forward_kinematics(self, joint_angles):
        try:
            # Gelen açıları numpy array'e çevir
            if not isinstance(joint_angles, np.ndarray):
                joint_angles = np.array(joint_angles)
                
            # Eksik açıları tamamla
            if joint_angles.size < len(self.joint_types):
                q = np.zeros(len(self.joint_types))
                q[:joint_angles.size] = joint_angles
            else:
                q = joint_angles
                
            return self.robot.fkine(q)
        except Exception as e:
            print(f"İleri kinematik çözüm hatası: {str(e)}")
            return None

    def visualize(self, joint_angles, block=True):
        try:
            self.robot.teach(joint_angles, block=block)
        except Exception as e:
            print(f"Görselleştirme hatası: {str(e)}")

    def print_solution_metrics(self):
        """Çözüm metriklerini yazdır"""
        print("\nÇözüm Metrikleri:")
        print("-----------------")
        for method, metrics in self.solution_metrics.items():
            print(f"\n{method.upper()} Yöntemi:")
            print(f"İterasyon Sayısı: {metrics['iterations']}")
            print(f"Çözüm Süresi: {metrics['time']:.4f} saniye")
            print(f"Hedef Noktaya Uzaklık: {metrics['error']:.4f} mm")
            print(f"Eklem Açıları (derece): {np.degrees(metrics['joint_angles'])}")

def get_robot_parameters():
    # [a, alpha, d, offset] UNIVERSAL ROBOTS UR5 PARAMETERS
    dh_params = [
        [0,     np.pi/2,     89.2,     0],    # Link 1 (Revolute)
        [-425,  0,           0,        0],    # Link 2 (Revolute)
        [-392,  0,           0,        0],    # Link 3 (Revolute)
        [0,     np.pi/2,     109.3,    0],    # Link 4 (Revolute)
        [0,    -np.pi/2,     94.75,    0],    # Link 5 (Revolute)
        [0,     0,           82.5,     0]     # Link 6 (Revolute)
    ]        

    joint_types = "RRRRRR"  # R: Revolute (Döner), P: Prismatic (Prizmatik)
    qlim = [
        [-2*np.pi, 2*np.pi],  # 1. eklem (R)
        [-2*np.pi, 2*np.pi],  # 2. eklem (R)
        [-2*np.pi, 2*np.pi],  # 3. eklem (R)
        [-2*np.pi, 2*np.pi],  # 4. eklem (R)
        [-2*np.pi, 2*np.pi],  # 5. eklem (R)
        [-2*np.pi, 2*np.pi]   # 6. eklem (R)
    ]
    return dh_params, joint_types, qlim

def test_ik_solvers():
    """Farklı test senaryoları için ters kinematik çözücüleri test et"""
    dh_params, joint_types, qlim = get_robot_parameters()
    robot = RobotManipulator(dh_params, joint_types, qlim)
    
    # Test senaryoları
    test_positions = [
        [50, 50, 50],    # Önde bir nokta
        [55, 55, 55],    # Yanda bir nokta
        [60, 60, 60],    # Yukarıda bir nokta
        [100, 100, 100], # Farklı bir nokta
    ]
    
    # Solver parametreleri
    solver_params = {
        'max_iter': 1000,
        'tolerance': 1e-3,
        'lambda_val': 0.2,  # DLS için
        'alpha': 0.5        # Jacobian için
    }
    
    for i, target_pos in enumerate(test_positions):
        print(f"\nTest {i+1}: Hedef Pozisyon = {target_pos}")
        print("=" * 50)
        
        # Tüm yöntemleri test et
        results = robot.solve_inverse_kinematics(
            target_pos, 
            method='all',
            **solver_params
        )
        
        # Sonuçları yazdır
        robot.print_solution_metrics()
        
        # En iyi çözümü görselleştir
        if results:
            best_method = min(results.items(), key=lambda x: x[1]['error'])[0]
            print(f"\nEn iyi çözüm ({best_method}) görselleştiriliyor...")
            robot.visualize(results[best_method]['joint_angles'])

def main():
    print("UR5 Robot Ters Kinematik Çözüm Testi Başlıyor...")
    print("=" * 50)
    
    test_ik_solvers()

if __name__ == "__main__":
    main()