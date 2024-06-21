import vxpy.core.protocol as vxprotocol
from visuals.yunyi_pre.intergrate import (Stationary, ForegroundStationary, ForegroundMovingForward,
                                          ForegroundMovingBackward, BackgroundStationaryBackward,
                                          BackgroundStationaryForward)
# from visuals.yunyi_pre.foreground_stationary import
# from visuals.yunyi_pre.foreground_moving_backward import ForegroundMovingBackward
# from visuals.yunyi_pre.background_stationary_backward import BackgroundStationaryBackward
# from visuals.yunyi_pre.background_stationary_foreward import BackgroundStationaryForward
# from visuals.yunyi_pre.stationary import Stationary
from visuals.yunyi_pre2.foreground_moving_forward import ForegroundMovingForward2
from visuals.yunyi_pre2.foreground_moving_backward import ForegroundMovingBackward2
from visuals.yunyi_pre2.background_stationary_backward import BackgroundStationaryBackward2


# from visuals.yunyi_pre2.background_stationary_foreward import BackgroundStationaryForward


class CMNFOREBACK(vxprotocol.StaticProtocol):

    def create(self):
        # Black
        phase = vxprotocol.Phase(duration=1 * 5)
        phase.set_visual(Stationary)
        self.add_phase(phase)

        # Black
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(ForegroundStationary)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(ForegroundMovingForward)
        self.add_phase(phase)

        # Black
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(ForegroundMovingBackward)
        self.add_phase(phase)

        # Black
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(BackgroundStationaryBackward)
        self.add_phase(phase)

        # Black
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(BackgroundStationaryForward)
        self.add_phase(phase)

        # Black
        # phase = vxprotocol.Phase(duration=3 * 5)
        # phase.set_visual(BackgroundStationaryBackward2)
        # self.add_phase(phase)

        # Black
        # phase = vxprotocol.Phase(duration=3 * 5)
        # phase.set_visual(ForegroundMovingBackward2)
        # self.add_phase(phase)

        # CMN
        # phase = vxprotocol.Phase(duration=3 * 5)
        # phase.set_visual(ForegroundMovingForward2)
        # self.add_phase(phase)

        # Black
        # phase = vxprotocol.Phase(duration=3 * 5)
        # phase.set_visual(ForegroundStationary)
        # self.add_phase(phase)

        # Black
        # phase = vxprotocol.Phase(duration=3 * 5)
        # phase.set_visual(Stationary)
        # self.add_phase(phase)
