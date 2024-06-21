import vxpy.core.protocol as vxprotocol
from visuals.cmn_yunyi_0514_protocalbaserevise.main import CMN, ForegroundStationary, ForegroundMovingForward, ForegroundMovingBackward, BackgroundStationaryBackward, BackgroundStationaryForward, \
    BackgroundStationaryBackwardReverse, ForegroundMovingBackwardReverse, ForegroundMovingForwardReverse, ForegroundStationaryReverse, CMNReverse, Pause2, Pause3, Pause4, Pause5, Pause6, Pause7, Pause8, Pause9


class CMN_FORE_BACK0510(vxprotocol.StaticProtocol):
    def create(self):
        # CMN
        phase = vxprotocol.Phase(duration=1 * 5)
        phase.set_visual(CMN)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=1 * 5)
        phase.set_visual(ForegroundStationary)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(ForegroundMovingForward)
        self.add_phase(phase)

        # Pause
        phase = vxprotocol.Phase(duration=1 * 5)
        phase.set_visual(Pause2)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(ForegroundMovingBackward)
        self.add_phase(phase)

        # Pause
        phase = vxprotocol.Phase(duration=1 * 5)
        phase.set_visual(Pause3)
        self.add_phase(phase)

        # Pause
        phase = vxprotocol.Phase(duration=1 * 5)
        phase.set_visual(Pause4)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(BackgroundStationaryBackward)
        self.add_phase(phase)

        # Pause
        phase = vxprotocol.Phase(duration=1 * 5)
        phase.set_visual(Pause5)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=6 * 5)
        phase.set_visual(BackgroundStationaryForward)
        self.add_phase(phase)

        # Pause
        phase = vxprotocol.Phase(duration=1 * 5)
        phase.set_visual(Pause6)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(BackgroundStationaryBackwardReverse)
        self.add_phase(phase)

        # Pause
        phase = vxprotocol.Phase(duration=1 * 5)
        phase.set_visual(Pause7)
        self.add_phase(phase)

        # Pause
        phase = vxprotocol.Phase(duration=1 * 5)
        phase.set_visual(Pause8)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(ForegroundMovingBackwardReverse)
        self.add_phase(phase)

        # Pause
        phase = vxprotocol.Phase(duration=1 * 5)
        phase.set_visual(Pause9)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=3 * 5)
        phase.set_visual(ForegroundMovingForwardReverse)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=1 * 5)
        phase.set_visual(ForegroundStationaryReverse)
        self.add_phase(phase)

        # CMN
        phase = vxprotocol.Phase(duration=1 * 5)
        phase.set_visual(CMNReverse)
        self.add_phase(phase)


