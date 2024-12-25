import numpy as np
from scipy.linalg import pinv

class IKSolver:
    @staticmethod
    def newton_raphson_solver(robot, target_position, max_iter=100, tolerance=1e-3):
        """Newton-Raphson method for inverse kinematics"""
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
                
                # Calculate Jacobian
                J = robot.robot.jacob0(q)[:3, :]
                
                # Calculate pseudo-inverse of Jacobian
                J_pinv = pinv(J)
                
                # Calculate joint corrections using Newton-Raphson method
                f = current_pos - target_position
                dq = -np.dot(J_pinv, f)
                
                # Update joint angles
                q = q + dq
                
                # Apply joint limits
                for i in range(len(q)):
                    q[i] = np.clip(q[i], robot.qlim[i][0], robot.qlim[i][1])
            
            return q, max_iter
            
        except Exception as e:
            print(f"Newton-Raphson solution error: {str(e)}")
            return None, max_iter

    @staticmethod
    def ccd_solver(robot, target_position, max_iter=1000, tolerance=1e-4):
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