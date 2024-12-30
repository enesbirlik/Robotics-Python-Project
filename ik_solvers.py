import numpy as np
from scipy.linalg import pinv

class IKSolver:

    @staticmethod
    def validate_joint_limits(robot, q):
        """Joint değerlerinin limitleri kontrol et ve düzelt"""
        for i, (val, limits, jtype) in enumerate(zip(q, robot.qlim, robot.joint_types)):
            if jtype == 'R':
                # Açısal limitler için normalizasyon
                while val > limits[1]:
                    val -= 2*np.pi
                while val < limits[0]:
                    val += 2*np.pi
            q[i] = np.clip(val, limits[0], limits[1])
        return q

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
        """Forward And Backward Reaching Inverse Kinematics (FABRIK) yöntemi"""
        try:
            num_joints = len(robot.joint_types)
            q = np.zeros(num_joints)
            
            # İlk olarak tüm joint pozisyonlarını hesapla
            joint_positions = [np.zeros(3)]  # Base position
            current_transform = np.eye(4)
            transforms = [current_transform]
            
            # Her eklemin pozisyonunu hesapla
            for i in range(num_joints):
                pose = robot.forward_kinematics(q[:i+1])
                if pose is not None:
                    joint_positions.append(pose.t)
                    transforms.append(pose.A)  # Homogeneous transformation matrix
                else:
                    print(f"Forward kinematics failed for joint {i+1}")
                    return None, max_iter
            
            # Link uzunluklarını hesapla
            link_lengths = []
            for i in range(len(joint_positions)-1):
                length = np.linalg.norm(joint_positions[i+1] - joint_positions[i])
                if length < 1e-6:
                    length = 1e-6
                link_lengths.append(length)
            
            total_length = sum(link_lengths)
            target_distance = np.linalg.norm(target_position)
            
            # Hedefin ulaşılabilir olup olmadığını kontrol et
            if target_distance > total_length:
                scale = total_length / target_distance
                target_position = target_position * scale
            
            for iteration in range(max_iter):
                # Mevcut uç nokta pozisyonunu al
                current_end = joint_positions[-1]
                error = np.linalg.norm(current_end - target_position)
                
                if error < tolerance:
                    # Başarılı çözüm - eklem açılarını hesapla
                    joint_angles = np.zeros(num_joints)
                    
                    for i in range(num_joints):
                        if i < num_joints - 1:  # Son eklem hariç
                            v1 = joint_positions[i+1] - joint_positions[i]
                            v2 = joint_positions[i+2] - joint_positions[i+1]
                            
                            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                                if robot.joint_types[i] == 'R':
                                    # Revolute joint için açı hesapla
                                    v1_norm = v1 / np.linalg.norm(v1)
                                    v2_norm = v2 / np.linalg.norm(v2)
                                    
                                    # Z ekseni etrafındaki açıyı hesapla
                                    angle = np.arctan2(np.cross(v1_norm, v2_norm)[2],
                                                    np.dot(v1_norm, v2_norm))
                                    joint_angles[i] = np.clip(angle, robot.qlim[i][0], robot.qlim[i][1])
                                else:
                                    # Prismatic joint için uzaklık hesapla
                                    distance = np.linalg.norm(v2)
                                    joint_angles[i] = np.clip(distance, robot.qlim[i][0], robot.qlim[i][1])
                    
                    return joint_angles, iteration
                
                # BACKWARD REACHING
                joint_positions[-1] = target_position.copy()
                
                for i in range(len(joint_positions)-2, -1, -1):
                    direction = joint_positions[i] - joint_positions[i+1]
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm > 1e-6:
                        direction = direction / dir_norm
                        # Eklem tipine göre hareket sınırlaması
                        if i < num_joints and robot.joint_types[i] == 'P':
                            # Prismatic joint için uzaklık sınırlaması
                            move_dist = np.clip(dir_norm, robot.qlim[i][0], robot.qlim[i][1])
                        else:
                            move_dist = link_lengths[i]
                        joint_positions[i] = joint_positions[i+1] + direction * move_dist
                
                # FORWARD REACHING
                joint_positions[0] = np.zeros(3)
                
                for i in range(len(joint_positions)-1):
                    direction = joint_positions[i+1] - joint_positions[i]
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm > 1e-6:
                        direction = direction / dir_norm
                        # Eklem tipine göre hareket sınırlaması
                        if i < num_joints and robot.joint_types[i] == 'P':
                            # Prismatic joint için uzaklık sınırlaması
                            move_dist = np.clip(dir_norm, robot.qlim[i][0], robot.qlim[i][1])
                        else:
                            move_dist = link_lengths[i]
                        joint_positions[i+1] = joint_positions[i] + direction * move_dist
                    
                    # Revolute eklemler için açı sınırlaması
                    if i < num_joints and robot.joint_types[i] == 'R':
                        if i < len(joint_positions) - 2:
                            v1 = joint_positions[i+1] - joint_positions[i]
                            v2 = joint_positions[i+2] - joint_positions[i+1]
                            
                            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                                v1_norm = v1 / np.linalg.norm(v1)
                                v2_norm = v2 / np.linalg.norm(v2)
                                
                                angle = np.arctan2(np.cross(v1_norm, v2_norm)[2],
                                                np.dot(v1_norm, v2_norm))
                                
                                # Açıyı limitler içinde tut
                                angle = np.clip(angle, robot.qlim[i][0], robot.qlim[i][1])
                                
                                # Yeni pozisyonu hesapla
                                rotation_matrix = np.array([
                                    [np.cos(angle), -np.sin(angle), 0],
                                    [np.sin(angle), np.cos(angle), 0],
                                    [0, 0, 1]
                                ])
                                new_pos = np.dot(rotation_matrix, v1_norm) * link_lengths[i]
                                joint_positions[i+1] = joint_positions[i] + new_pos
            
            print(f"FABRIK failed to converge after {max_iter} iterations")
            return None, max_iter
            
        except Exception as e:
            print(f"FABRIK çözüm hatası: {str(e)}")
            import traceback
            traceback.print_exc()
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