import numpy as np

def get_ur5_parameters():
    # [a, alpha, d, offset] UNIVERSAL ROBOTS UR5 PARAMETERS
    dh_params = [
        [0,     np.pi/2,     89.2,    0],
        [-425,  0,           0,       0],
        [-392,  0,           0,       0],
        [0,     np.pi/2,     109.3,   0],
        [0,    -np.pi/2,     94.75,   0],
        [0,     0,           82.5,    0]
    ]        

    joint_types = "RRRRRR"
    qlim = [[-2*np.pi, 2*np.pi] for _ in range(6)]
    
    return dh_params, joint_types, qlim

def get_custom_robot_parameters():
    # [a, alpha, d, offset] MY ROBOT PARAMETERS sf139
    dh_params = [
        [0,     0,          50,    0],
        [20,    0,          0,     0],
        [20,    np.pi,      40,    0],
        [0,     0,          10,    0],
        [0,     -np.pi/2,   0,     0],
        [0,     np.pi/2,    0,     0],
    ]        

    joint_types = "RRPRRR"

    qlim = [
        [-np.pi, np.pi],
        [-np.pi, np.pi],
        [-40,      40],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
        [-np.pi, np.pi]
    ]
    
    return dh_params, joint_types, qlim

def get_puma560_parameters():
    # Puma 560 robot parametreleri (dereceyi radyana çevir)
    dh_params = [
        [0.0, np.pi/2, 0.0, 0.0],
        [431.8, 0.0, 0.0, 0.0],
        [20.3, np.pi/2, 150.0, 0.0],
        [0.0, -np.pi/2, 431.8, 0.0],
        [0.0, np.pi/2, 0.0, 0.0],
        [0.0, 0.0, 56.5, 0.0]
    ]
    joint_types = "RRRRRR"
    qlim = [
        [np.radians(-160.0), np.radians(160.0)],
        [np.radians(-225.0), np.radians(45.0)],
        [np.radians(-45.0), np.radians(225.0)],
        [np.radians(-110.0), np.radians(170.0)],
        [np.radians(-100.0), np.radians(100.0)],
        [np.radians(-266.0), np.radians(266.0)]
    ]
    return dh_params, joint_types, qlim

def get_kr6_parameters():
    # KUKA KR6 robot parametreleri (dereceyi radyana çevir)
    dh_params = [
        [0.0, np.pi/2, 400.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [455.0, np.pi/2, 0.0, 0.0],
        [35.0, -np.pi/2, 420.0, 0.0],
        [0.0, np.pi/2, 0.0, 0.0],
        [0.0, 0.0, 80.0, 0.0]
    ]
    joint_types = "RRRRRR"
    qlim = [
        [np.radians(-170.0), np.radians(170.0)],
        [np.radians(-190.0), np.radians(45.0)],
        [np.radians(-120.0), np.radians(156.0)],
        [np.radians(-185.0), np.radians(185.0)],
        [np.radians(-120.0), np.radians(120.0)],
        [np.radians(-350.0), np.radians(350.0)]
    ]
    return dh_params, joint_types, qlim

def get_test_positions():
    return [
        [0, 60, 100],    # First position
        [40, 0, 100]     # Goal position   
    ]

def get_solver_parameters():
    return {
        'max_iter': 2000,
        'tolerance': 1e-2,
        'lambda_val': 0.2,  # For DLS
        'alpha': 0.5        # For Jacobian
    }
