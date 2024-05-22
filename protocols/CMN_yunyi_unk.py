import vxpy.core.protocol as vxprotocol
from visuals.yunyi_unc0521.cyy_intergrate_unc_0521 import ForegroundMovingForward
# from visuals.yunyi_pre.foreground_stationary import
# from visuals.yunyi_pre.foreground_moving_backward import ForegroundMovingBackward
# from visuals.yunyi_pre.background_stationary_backward import BackgroundStationaryBackward
# from visuals.yunyi_pre.background_stationary_foreward import BackgroundStationaryForward
# from visuals.yunyi_pre.stationary import Stationary
from visuals.yunyi_pre2.foreground_moving_forward import ForegroundMovingForward2
from visuals.yunyi_pre2.foreground_moving_backward import ForegroundMovingBackward2
from visuals.yunyi_pre2.background_stationary_backward import BackgroundStationaryBackward2


# from visuals.yunyi_pre2.background_stationary_foreward import BackgroundStationaryForward


class REVISECMNFOREBACK(vxprotocol.StaticProtocol):

    def create(self):

        # CMN
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(ForegroundMovingForward)
        self.add_phase(phase)

