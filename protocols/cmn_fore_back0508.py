import vxpy.core.protocol as vxprotocol
from visuals.cmn_yunyi_0508_basetim.main import CMN_FORE_BACK, Stationary, ForegroundStationary, \
    ForegroundMovingForward, ForegroundMovingBackward, BackgroundStationaryBackward, BackgroundStationaryForward, \
    BackgroundStationaryBackwardReverse, ForegroundMovingBackwardReverse, ForegroundMovingForwardReverse, \
    ForegroundStationaryReverse, StationaryReverse


class CMN_FORE_BACK0508(vxprotocol.StaticProtocol):

    def create(self):
        # CMN
        phase = vxprotocol.Phase(duration=1 * 5)
        phase.set_visual(Stationary)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(ForegroundStationary)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(ForegroundMovingForward)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(ForegroundMovingBackward)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(BackgroundStationaryBackward)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=6 * 5)
        phase.set_visual(BackgroundStationaryForward)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(BackgroundStationaryBackwardReverse)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(ForegroundMovingBackwardReverse)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(ForegroundMovingForwardReverse)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(ForegroundStationaryReverse)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=1 * 5)
        phase.set_visual(StationaryReverse)
        self.add_phase(phase)


