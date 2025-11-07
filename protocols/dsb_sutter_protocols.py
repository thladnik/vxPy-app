import vxpy.core.protocol as vxprotocol
from controls import sutter_micromanipulator_controls
from visuals.spherical_grating import SphericalBlackWhiteGrating
import vxpy.core.event as vxevent

class TestSutterDSB(vxprotocol.StaticProtocol):

    def create(self):


        # illumination phase
        phase = vxprotocol.Phase(duration=5)
        phase.set_visual(SphericalBlackWhiteGrating,
                     {SphericalBlackWhiteGrating.waveform: 'rectangular',
                      SphericalBlackWhiteGrating.motion_axis: 'vertical',
                      SphericalBlackWhiteGrating.motion_type: 'rotation',
                      SphericalBlackWhiteGrating.angular_period: 5,
                      SphericalBlackWhiteGrating.angular_velocity: 0})
        phase.set_control(sutter_micromanipulator_controls.ControlSutterMP,
                          {'move_to_x': -500, 'move_to_y': 0, 'move_to_z': 0})
        self.add_phase(phase)


