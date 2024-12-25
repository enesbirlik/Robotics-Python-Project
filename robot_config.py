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

def get_my_robot_parameters():
    # [a, alpha, d, offset] MY ROBOT PARAMETERS
    dh_params = [
        [0,     np.pi/2,     100,    0],
        [0,     0,           0,      0],
        [0,     0,           0,      0],
        [0,     np.pi/2,     100,    0],
        [0,    -np.pi/2,     100,    0],
        [0,     0,           100,    0]
    ]        

    joint_types = "RRRRRR"
    qlim = [[-2*np.pi, 2*np.pi] for _ in range(6)]
    
    return dh_params, joint_types, qlim

def get_test_positions():
    return [
        [200, 400, 200],    # First position
        [300, 300, 300]       # Goal position   
    ]

def get_solver_parameters():
    return {
        'max_iter': 1000,
        'tolerance': 1e-3,
        'lambda_val': 0.2,  # For DLS
        'alpha': 0.5        # For Jacobian
    }