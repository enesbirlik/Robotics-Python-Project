import roboticstoolbox as rtb
import numpy as np
from spatialmath import SE3
import matplotlib.pyplot as plt

class IKSolver:
    def __init__(self):
        # Özel robotumuzu oluştur
        self.robot = self.create_custom_robot()
        
    def create_custom_robot(self):
        # Kendi DH parametrelerinizle robot oluşturun
        robot = rtb.DHRobot([
            rtb.RevoluteDH(d=0.3, a=0.0, alpha=np.pi/2),
            rtb.RevoluteDH(d=0.0, a=0.4, alpha=0),
            rtb.RevoluteDH(d=0.0, a=0.3, alpha=np.pi/2),
            # Diğer eklemler...
        ], name='CustomRobot')
        return robot
    
    def solve_ccd(self, target_pose, max_iter=100, tol=1e-3):
        """
        CCD (Cyclic Coordinate Descent) metodu implementasyonu
        
        Parametreler:
        target_pose: SE3 - Hedef pozisyon ve oryantasyon
        max_iter: int - Maksimum iterasyon sayısı
        tol: float - Hata toleransı
        """
        # Başlangıç açıları
        q = self.robot.q
        target_pos = target_pose.t  # Hedef pozisyon
        
        for iteration in range(max_iter):
            # Her eklem için geriye doğru iterasyon
            for joint in range(self.robot.n - 1, -1, -1):
                # Mevcut uç pozisyonunu hesapla
                current_pose = self.robot.fkine(q)
                current_pos = current_pose.t
                
                # Eklem pozisyonunu hesapla
                joint_pos = self.robot.fkine(q, end=joint).t
                
                # Vektörleri hesapla
                current_vector = current_pos - joint_pos
                target_vector = target_pos - joint_pos
                
                # Vektörleri normalize et
                current_vector = current_vector / np.linalg.norm(current_vector)
                target_vector = target_vector / np.linalg.norm(target_vector)
                
                # Rotasyon açısını hesapla
                dot_product = np.clip(np.dot(current_vector, target_vector), -1.0, 1.0)
                angle = np.arccos(dot_product)
                
                # Rotasyon ekseni
                axis = np.cross(current_vector, target_vector)
                if np.linalg.norm(axis) > 0:
                    axis = axis / np.linalg.norm(axis)
                    
                    # Açıyı güncelle
                    q[joint] += angle
                    
            # Hedef pozisyona yakınsama kontrolü
            if np.linalg.norm(current_pos - target_pos) < tol:
                break
                
        return q
    
    def solve_fabrik(self, target_pose, max_iter=100, tol=1e-3):
        """
        FABRIK metodu implementasyonu
        
        Parametreler:
        target_pose: SE3 - Hedef pozisyon ve oryantasyon
        max_iter: int - Maksimum iterasyon sayısı
        tol: float - Hata toleransı
        """
        # Başlangıç pozisyonlarını al
        joint_positions = []
        q = self.robot.q
        
        # Tüm eklem pozisyonlarını hesapla
        for i in range(self.robot.n + 1):
            pos = self.robot.fkine(q, end=i).t
            joint_positions.append(pos)
        
        joint_positions = np.array(joint_positions)
        target_pos = target_pose.t
        base_pos = joint_positions[0]
        
        # Link uzunluklarını hesapla
        link_lengths = []
        for i in range(len(joint_positions)-1):
            length = np.linalg.norm(joint_positions[i+1] - joint_positions[i])
            link_lengths.append(length)
        
        for iteration in range(max_iter):
            # İleri hareket
            joint_positions[-1] = target_pos
            
            for i in range(len(joint_positions)-2, -1, -1):
                r = joint_positions[i+1] - joint_positions[i]
                r = r / np.linalg.norm(r)
                joint_positions[i] = joint_positions[i+1] - link_lengths[i] * r
            
            # Geri hareket
            joint_positions[0] = base_pos
            
            for i in range(1, len(joint_positions)):
                r = joint_positions[i] - joint_positions[i-1]
                r = r / np.linalg.norm(r)
                joint_positions[i] = joint_positions[i-1] + link_lengths[i-1] * r
            
            # Hedef kontrolü
            if np.linalg.norm(joint_positions[-1] - target_pos) < tol:
                break
        
        # Eklem açılarını hesapla
        q_new = self.positions_to_angles(joint_positions)
        return q_new

    def positions_to_angles(self, joint_positions):
        """
        Eklem pozisyonlarından eklem açılarını hesapla
        """
        q = np.zeros(self.robot.n)
        
        for i in range(self.robot.n):
            # İki vektör arasındaki açıyı hesapla
            v1 = joint_positions[i+1] - joint_positions[i]
            v2 = joint_positions[i+2] - joint_positions[i+1]
            
            # Açıyı hesapla
            dot_product = np.dot(v1, v2)
            angle = np.arccos(dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            
            q[i] = angle
        
        return q
        
    def solve_jacobian(self, target_pose, max_iter=100, alpha=0.1, tol=1e-3):
        """
        Jacobian bazlı metod implementasyonu
        
        Parametreler:
        target_pose: SE3 - Hedef pozisyon ve oryantasyon
        max_iter: int - Maksimum iterasyon sayısı
        alpha: float - Öğrenme oranı
        tol: float - Hata toleransı
        """
        q = self.robot.q
        target_pos = target_pose.t
        
        for iteration in range(max_iter):
            # Mevcut pozisyon ve Jacobian'ı hesapla
            current_pose = self.robot.fkine(q)
            J = self.robot.jacob0(q)
            
            # Pozisyon hatası
            error = target_pos - current_pose.t
            
            # Jacobian'ın pseudo-inverse'ini hesapla
            J_pinv = np.linalg.pinv(J[:3])  # Sadece pozisyon için
            
            # Eklem açılarını güncelle
            dq = alpha * J_pinv @ error
            q = q + dq
            
            # Yakınsama kontrolü
            if np.linalg.norm(error) < tol:
                break
        
        return q
    
    def solve_dls(self, target_pose, max_iter=100, lambda_param=0.1, tol=1e-3):
        """
        Damped Least Squares metodu implementasyonu
        
        Parametreler:
        target_pose: SE3 - Hedef pozisyon ve oryantasyon
        max_iter: int - Maksimum iterasyon sayısı
        lambda_param: float - Sönümleme parametresi
        tol: float - Hata toleransı
        """
        q = self.robot.q
        target_pos = target_pose.t
        
        for iteration in range(max_iter):
            # Mevcut pozisyon ve Jacobian'ı hesapla
            current_pose = self.robot.fkine(q)
            J = self.robot.jacob0(q)
            
            # Pozisyon hatası
            error = target_pos - current_pose.t
            
            # DLS çözümü
            J_dls = J[:3].T @ np.linalg.inv(J[:3] @ J[:3].T + lambda_param * np.eye(3))
            
            # Eklem açılarını güncelle
            dq = J_dls @ error
            q = q + dq
            
            # Yakınsama kontrolü
            if np.linalg.norm(error) < tol:
                break
        
        return q
    
    def compute_trajectory(self, q_start, q_end, method='linear'):
        # İki nokta arası yörünge planlaması
        pass

def test_methods():
    solver = IKSolver()
    
    # Test hedefi
    target = SE3(0.5, 0.5, 0.5)
    
    # Farklı metodlarla çözüm
    q_ccd = solver.solve_ccd(target)
    q_fabrik = solver.solve_fabrik(target)
    q_jacobian = solver.solve_jacobian(target)
    q_dls = solver.solve_dls(target)
    
    # Sonuçları görselleştir
    plt.figure(figsize=(20, 5))
    
    plt.subplot(141)
    solver.robot.plot(q_ccd)
    plt.title('CCD Çözümü')
    
    plt.subplot(142)
    solver.robot.plot(q_fabrik)
    plt.title('FABRIK Çözümü')
    
    plt.subplot(143)
    solver.robot.plot(q_jacobian)
    plt.title('Jacobian Çözümü')
    
    plt.subplot(144)
    solver.robot.plot(q_dls)
    plt.title('DLS Çözümü')
    
    plt.show()

if __name__ == "__main__":
    test_methods()