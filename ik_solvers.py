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
    def fabrik_solver(robot, target_position, max_iter=100, tolerance=10):
        """FABRIK (Forward And Backward Reaching IK) Implementation"""
        try:
            # 1. Initialize
            num_joints = len(robot.joint_types)
            q = np.zeros(num_joints)
            
            print("\n=== FABRIK Debug Info ===")
            print(f"Target position: {target_position}")
            
            # Get initial joint positions
            joint_positions = [np.zeros(3)]  # Base position
            print("\nInitial Joint Positions:")
            for i in range(num_joints):
                pose = robot.forward_kinematics(q[:i+1])
                if pose is None:
                    print(f"Failed to get position for joint {i}")
                    return None, max_iter
                joint_positions.append(pose.t)
                print(f"Joint {i}: {joint_positions[-1]}")
            
            # Calculate link lengths
            link_lengths = []
            print("\nLink Lengths:")
            for i in range(len(joint_positions)-1):
                length = np.linalg.norm(joint_positions[i+1] - joint_positions[i])
                link_lengths.append(max(length, 10))
                print(f"Link {i}: {link_lengths[-1]:.3f}")
                
            total_length = sum(link_lengths)
            target_dist = np.linalg.norm(target_position)
            print(f"\nTotal chain length: {total_length:.3f}")
            print(f"Distance to target: {target_dist:.3f}")
            
            # Check reachability
            if target_dist > total_length:
                print("Target out of reach!")
                return None, max_iter
                
            base_pos = joint_positions[0].copy()
            
            # Main FABRIK Loop
            for iteration in range(max_iter):
                print(f"\nIteration {iteration+1}")
                
                # Forward reaching
                joint_positions[-1] = target_position.copy()
                print("\nForward reaching:")
                for i in range(len(joint_positions)-2, -1, -1):
                    direction = joint_positions[i] - joint_positions[i+1]
                    distance = link_lengths[i]
                    
                    if np.linalg.norm(direction) > 10:
                        direction = direction / np.linalg.norm(direction)
                        joint_positions[i] = joint_positions[i+1] + direction * distance
                    print(f"Joint {i} new pos: {joint_positions[i]}")
                
                # Backward reaching
                joint_positions[0] = base_pos.copy()
                print("\nBackward reaching:")
                for i in range(len(joint_positions)-1):
                    direction = joint_positions[i+1] - joint_positions[i]
                    distance = link_lengths[i]
                    
                    if np.linalg.norm(direction) > 10:
                        direction = direction / np.linalg.norm(direction)
                        joint_positions[i+1] = joint_positions[i] + direction * distance
                    print(f"Joint {i+1} new pos: {joint_positions[i+1]}")
                
                error = np.linalg.norm(joint_positions[-1] - target_position)
                print(f"\nCurrent error: {error:.6f}")
                
                if error < tolerance:
                    print("\nConverged!")
                    q_new = np.zeros(num_joints)
                    for i in range(num_joints):
                        if robot.joint_types[i] == 'R':
                            v1 = joint_positions[i+1] - joint_positions[i]
                            v2 = joint_positions[i+2] - joint_positions[i+1] if i < num_joints-1 else v1
                            angle = np.arctan2(np.cross(v1, v2)[2], np.dot(v1, v2))
                            q_new[i] = np.clip(angle, robot.qlim[i][0], robot.qlim[i][1])
                        else:
                            distance = np.linalg.norm(joint_positions[i+1] - joint_positions[i])
                            q_new[i] = np.clip(distance, robot.qlim[i][0], robot.qlim[i][1])
                    print(f"Final joint angles: {q_new}")
                    return q_new, iteration
            
            print("\nMax iterations reached without aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            return None, max_iter
            
        except Exception as e:
            print(f"FABRIK solver error: {str(e)}")
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