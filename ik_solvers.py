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
        """FABRIK (Forward And Backward Reaching Inverse Kinematics)"""
        try:
            num_joints = len(robot.joint_types)
            q = np.zeros(num_joints)
            
            # 1. Initialize
            joint_positions = [np.zeros(3)]  # Base position
            for i in range(num_joints):
                pose = robot.forward_kinematics(q[:i+1])
                if pose is None:
                    return None, max_iter
                joint_positions.append(pose.t)
            
            link_lengths = [np.linalg.norm(joint_positions[i+1] - joint_positions[i]) for i in range(num_joints)]
            total_length = sum(link_lengths)
            target_distance = np.linalg.norm(target_position - joint_positions[0])
            
            if target_distance > total_length:
                print("Target out of reach")
                return None, max_iter
            
            # 2. Main FABRIK Loop
            for iteration in range(max_iter):
                # Forward reaching
                joint_positions[-1] = target_position.copy()
                for i in range(num_joints-1, 0, -1):
                    direction = joint_positions[i] - joint_positions[i+1]
                    direction = direction / np.linalg.norm(direction) * link_lengths[i-1]
                    joint_positions[i] = joint_positions[i+1] + direction
                
                # Backward reaching
                joint_positions[0] = np.zeros(3)  # Base position
                for i in range(num_joints):
                    direction = joint_positions[i+1] - joint_positions[i]
                    direction = direction / np.linalg.norm(direction) * link_lengths[i]
                    joint_positions[i+1] = joint_positions[i] + direction
                
                # Check convergence
                error = np.linalg.norm(joint_positions[-1] - target_position)
                if error < tolerance:
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
                    return q_new, iteration
            
            print("Max iterations reached without convergence")
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
        
    @staticmethod
    def ccd_cozer_solver(robot, target_position, max_iter=100, tolerance=1e-3):
        """CCD (Cyclic Coordinate Descend Solver) Yöntemi"""
        try:
            q = np.zeros(len(robot.joint_types))  # Initial joint configuration
            joint_positions = robot.get_joint_positions(q)
            ee_position = joint_positions[-1]  # Use the last joint position as the end-effector position

            for iteration in range(max_iter):
                if np.linalg.norm(ee_position - target_position) < tolerance:
                    break

                for i in reversed(range(len(robot.joints))):
                    joint_position = joint_positions[i]

                    # Direction vectors
                    direction_to_effector = ee_position - joint_position
                    direction_to_goal = target_position - joint_position

                    # Normalize directions
                    direction_to_effector /= np.linalg.norm(direction_to_effector)
                    direction_to_goal /= np.linalg.norm(direction_to_goal)

                    # Calculate rotation axis and angle
                    angle = np.arccos(np.clip(
                        np.dot(direction_to_effector, direction_to_goal), -1.0, 1.0))
                    axis = np.cross(direction_to_effector, direction_to_goal)

                    # Skip if the axis is near zero (no rotation needed)
                    if np.linalg.norm(axis) < 1e-6:
                        continue
                    axis /= np.linalg.norm(axis)

                    amk = robot.robot.fkine_all(q)

                    # Update joint angle for revolute joints
                    if robot.joint_types[i] == 'R':
                        q[i] += angle * np.sign(np.dot(axis, amk[i].a))
                        if robot.joints[i].qlim is not None:
                            q[i] = np.clip(q[i], robot.joints[i].qlim[0], robot.joints[i].qlim[1])

                    elif robot.joint_types[i] == 'P':
                        vector_to_target = target_position - joint_position
                        translation_axis = np.array([0, 0, 1])  # Assuming z-axis
                        displacement = np.dot(vector_to_target, translation_axis)
                        displacement = np.clip(displacement, -np.linalg.norm(vector_to_target), np.linalg.norm(vector_to_target))

                        new_position = q[i] + displacement
                        q[i] = np.clip(new_position, robot.joints[i].qlim[0], robot.joints[i].qlim[1])

                    # Update joint positions and end-effector position
                    joint_positions = robot.get_joint_positions(q)
                    ee_position = joint_positions[-1]
                    print(q)
                return q, iteration
            
            print("Max iterations reached without convergence")
            return None, max_iter
        
        except Exception as e:
            print(f"CCD Cozer solver error: {str(e)}")
            return None, max_iter