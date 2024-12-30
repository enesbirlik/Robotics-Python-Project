import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
import matplotlib.pyplot as plt

class MyRobot:
    def __init__(self, dh_params, joint_types, joint_limits=None):
        """
        dh_params: [[d, a, alpha], ...] formatında liste
        joint_types: ['P', 'R', ...] formatında eklem tipleri listesi
                    'P': Prismatic (kayıcı) eklem
                    'R': Revolute (döner) eklem
        joint_limits: [[min, max], ...] formatında liste (opsiyonel)
        """
        
        if len(dh_params) != len(joint_types):
            raise ValueError("DH parametreleri ve eklem tiplerinin sayısı eşit olmalı!")
            
        # Robot modelini oluştur
        links = []
        for i in range(len(joint_types)):
            if joint_types[i].upper() == 'R':
                links.append(RevoluteDH(d=dh_params[i][0], 
                                      a=dh_params[i][1], 
                                      alpha=dh_params[i][2]))
            elif joint_types[i].upper() == 'P':
                links.append(PrismaticDH(theta=0,
                                       a=dh_params[i][1], 
                                       alpha=dh_params[i][2]))
            else:
                raise ValueError(f"Geçersiz eklem tipi: {joint_types[i]}")
        
        self.robot = DHRobot(links)
        self.num_joints = len(links)
        self.current_joints = np.zeros(self.num_joints)
        self.joint_types = joint_types

        # Varsayılan eklem limitleri
        if joint_limits is None:
            self.joint_limits = []
            for joint_type in joint_types:
                if joint_type.upper() == 'R':
                    self.joint_limits.append([-2*np.pi, 2*np.pi])  # Revolute için ±360°
                else:
                    self.joint_limits.append([-1000, 1000])  # Prismatic için ±1000mm
        else:
            if len(joint_limits) != self.num_joints:
                raise ValueError("Eklem limitleri sayısı eklem sayısına eşit olmalı!")
            self.joint_limits = joint_limits

    def forward_kinematics(self, joint_values):
        """Forward kinematik hesaplama"""
        return self.robot.fkine(joint_values)

    def calculate_jacobian(self, joint_values):
        """Jacobian matrisini hesapla"""
        return self.robot.jacob0(joint_values)

    def inverse_kinematics_dls(self, target_pose, max_iter=1000, epsilon=1e-6, lambda_val=0.1):
        """
        Damped Least Squares yöntemi ile inverse kinematik çözümü
        """
        current_joints = self.current_joints.copy()
        
        for i in range(max_iter):
            current_pose = self.forward_kinematics(current_joints)
            current_position = current_pose.t
            
            position_error = target_pose - current_position
            error = position_error[:3]
            
            if np.linalg.norm(error) < epsilon:
                print(f"Çözüm {i} iterasyonda bulundu")
                break
                
            J = self.calculate_jacobian(current_joints)
            J = J[:3]
            
            JT = J.transpose()
            temp = np.linalg.inv(J @ JT + lambda_val * np.eye(3))
            delta_theta = JT @ temp @ error
            
            current_joints = current_joints + delta_theta
            
            # Her eklem için kendi limitlerini uygula
            for j in range(self.num_joints):
                current_joints[j] = np.clip(current_joints[j], 
                                          self.joint_limits[j][0], 
                                          self.joint_limits[j][1])
            
        self.current_joints = current_joints
        return current_joints

    def plot_robot(self):
        """Robot modelini görselleştir"""
        plt.ion()
        fig = self.robot.plot(self.current_joints)
        return fig

def main():
    # SCARA Robot için (RRPR)
    dh_params = [
        [100,  250,  0],        # R1
        [0,    250,  0],        # R2
        [0,    0,    0],        # P
        [50,   0,    0]         # R4
    ]
    joint_types = ['R', 'R', 'P', 'R']
    
    # Her eklem için özel limitler (min, max)
    joint_limits = [
        [-np.pi/2, np.pi/2],    # R1: ±90°
        [-np.pi, np.pi],        # R2: ±180°
        [-500, 500],               # P3: 0-150mm
        [-np.pi/2, np.pi/2]     # R4: ±90°
    ]
    
    # Robot nesnesini oluştur
    robot = MyRobot(dh_params, joint_types, joint_limits)
    
    # Hedef pozisyon
    target_position = np.array([300, 300, 10])
    
    try:
        print("Hedef pozisyon:", target_position)
        solution = robot.inverse_kinematics_dls(target_position)
        print("\nBulunan eklem değerleri:")
        for i, (val, type) in enumerate(zip(solution, joint_types)):
            if type.upper() == 'P':
                print(f"Eklem {i+1} (Prismatic): {val:.2f} mm")
                print(f"  Limitler: [{joint_limits[i][0]:.2f}, {joint_limits[i][1]:.2f}] mm")
            else:
                print(f"Eklem {i+1} (Revolute): {val:.2f} rad ({np.degrees(val):.2f} derece)")
                print(f"  Limitler: [{np.degrees(joint_limits[i][0]):.2f}, {np.degrees(joint_limits[i][1]):.2f}] derece")
        
        final_pose = robot.forward_kinematics(solution)
        print("\nUlaşılan pozisyon:", final_pose.t)
        print("Hedef pozisyona olan hata:", np.linalg.norm(target_position - final_pose.t[:3]))
        
        fig = robot.plot_robot()
        plt.ioff()
        plt.show()
        
    except Exception as e:
        print("Hata oluştu:", str(e))

if __name__ == "__main__":
    main()