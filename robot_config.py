import numpy as np

def get_ur5_parameters():
    """UR5 Robot Parameters"""
    dh_params = [
        [89.2,   0,      np.pi/2,  0],    # Base
        [0,      -425,   0,        0],    # Shoulder
        [0,      -392,   0,        0],    # Elbow
        [109.3,  0,      np.pi/2,  0],    # Wrist 1
        [94.75,  0,     -np.pi/2,  0],    # Wrist 2
        [82.5,   0,      0,        0]     # Wrist 3
    ]        
    joint_types = "RRRRRR"
    qlim = [[-2*np.pi, 2*np.pi] for _ in range(6)]
    
    return dh_params, joint_types, qlim

def get_custom_robot_parameters():
    """Custom Robot Parameters"""
    dh_params = [
        [0,    0,            30,     0],    # Joint 1 (R)
        [20,   np.pi/2,            0,      0],    # Joint 2 (R)
        [20,   np.pi,        40,     0],    # Joint 3 (P)
        # [0,    0,            40,     0],    # Joint 4 (R)
        # [0,    -np.pi/2,     0,      0],    # Joint 5 (R)
        # [0,    np.pi/2,      0,      0]     # Joint 6 (R)
    ]        
    joint_types = "RRP"
    qlim = [
        [-np.pi, np.pi],     # Joint 1 (R)
        [-np.pi, np.pi],     # Joint 2 (R)
        [-100,    100],        # Joint 3 (P) mm
        # [-np.pi, np.pi],     # Joint 4 (R)
        # [-np.pi, np.pi],     # Joint 5 (R)
        # [-np.pi, np.pi]      # Joint 6 (R)
    ]
    
    return dh_params, joint_types, qlim

def get_scara_parameters():
    """SCARA Robot Parameters"""
    dh_params = [
        [100,   250,    0,      0],    # Joint 1 (R)
        [0,     250,    np.pi,  0],    # Joint 2 (R)
        [0,     0,      0,      0],    # Joint 3 (P)
        #[0,     0,      0,      0]     # Joint 4 (R)
    ]
    joint_types = "RRP"
    qlim = [
        [-np.pi/2, np.pi/2],    # Joint 1 (R)
        [-np.pi, np.pi],        # Joint 2 (R)
        [-150, 150],            # Joint 3 (P) mm
        #[-np.pi/2, np.pi/2]     # Joint 4 (R)
    ]
    
    return dh_params, joint_types, qlim

def get_kr6_parameters():
    """KUKA KR6 Robot Parameters"""
    dh_params = [
        [400.0,  0.0,    np.pi/2,  0.0],    # Base
        [0.0,    0.0,    0.0,      0.0],    # Shoulder
        [0.0,    455.0,  np.pi/2,  0.0],    # Upper arm
        [420.0,  35.0,   -np.pi/2, 0.0],    # Forearm
        [0.0,    0.0,    np.pi/2,  0.0],    # Wrist pitch
        [80.0,   0.0,    0.0,      0.0]     # Wrist roll
    ]
    joint_types = "RRRRRR"
    qlim = [
        [np.radians(-170), np.radians(170)],    # Base
        [np.radians(-190), np.radians(45)],     # Shoulder
        [np.radians(-120), np.radians(156)],    # Elbow
        [np.radians(-185), np.radians(185)],    # Wrist 1
        [np.radians(-120), np.radians(120)],    # Wrist 2
        [np.radians(-350), np.radians(350)]     # Wrist 3
    ]
    
    return dh_params, joint_types, qlim

def get_test_positions():
    """Test positions for inverse kinematics"""
    return [
        [100, 20, 140],     # Start position (mm)
        # [40, 0, 100]      # Goal position (mm)
    ]

def get_solver_parameters():
    """Solver parameters for inverse kinematics"""
    return {
        'max_iter': 1000,      # Maximum iterations
        'tolerance': 1e-2,     # Error tolerance
        'lambda_val': 0.2,     # Damping factor for DLS
        'alpha': 0.5           # Step size for Jacobian
    }