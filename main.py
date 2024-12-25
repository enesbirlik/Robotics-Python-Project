from src.robot import CustomRobot
from src.solvers.ccd_solver import CCDSolver
from src.solvers.fabrik_solver import FABRIKSolver
from src.utils.visualization import RobotVisualizer
from spatialmath import SE3

def main():
    # Robot oluştur
    robot = CustomRobot()
    
    # Hedef pozisyon
    target = SE3(0.5, 0.5, 0.5)
    
    # Çözücüleri oluştur
    ccd_solver = CCDSolver(robot)
    fabrik_solver = FABRIKSolver(robot)
    
    # Çözümleri hesapla
    q_ccd = ccd_solver.solve(target)
    q_fabrik = fabrik_solver.solve(target)
    
    # Sonuçları görselleştir
    RobotVisualizer.compare_solutions(robot, {
        'CCD': q_ccd,
        'FABRIK': q_fabrik
    })

if __name__ == "__main__":
    main()