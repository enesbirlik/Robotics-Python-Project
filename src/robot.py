import roboticstoolbox as rtb
import numpy as np
from spatialmath import SE3

class CustomRobot:
    def __init__(self, dh_params=None):
        self.robot = self._create_robot(dh_params)
    
    def _create_robot(self, dh_params):
        if dh_params is None:
            # Varsayılan DH parametreleri
            return rtb.DHRobot([
                rtb.RevoluteDH(d=0.3, a=0.0, alpha=np.pi/2),
                rtb.RevoluteDH(d=0.0, a=0.4, alpha=0),
                rtb.RevoluteDH(d=0.0, a=0.3, alpha=np.pi/2),
                rtb.RevoluteDH(d=0.4, a=0.0, alpha=-np.pi/2),
                rtb.RevoluteDH(d=0.0, a=0.0, alpha=np.pi/2),
                rtb.RevoluteDH(d=0.2, a=0.0, alpha=0)
            ], name='CustomRobot')
        return rtb.DHRobot(dh_params)