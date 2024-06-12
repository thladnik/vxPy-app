import vxpy.core.protocol as vxprotocol
from vxpy.core.protocol import StaticProtocol
import vxpy.core.event as vxevent
from vxpy.extras.zf_eyeposition_tracking import ZFEyeTracking
from visuals.cmn_yunyi_0612repeat.main import FORE_NON_CMN
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground
import vxpy.core.visual as vxvisual
import numpy as np
import time


class CMN_FORE_BACK0612(vxprotocol.StaticProtocol):

    def create(self):
        # CMN
        phase = vxprotocol.Phase(duration=180)
        phase.set_visual(FORE_NON_CMN)
        self.add_phase(phase)
