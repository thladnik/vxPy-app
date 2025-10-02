import vxpy.core.control as vxcontrol
from plugins.led_pwm_plugin import LedTriggerRoutine


class LedPWMControl(vxcontrol.BaseControl):

    light_intensity: float = None

    def __init__(self, *args, **kwargs):
        vxcontrol.BaseControl.__init__(self, *args, **kwargs)

    def initialize(self, **kwargs):
        LedTriggerRoutine.instance().light_intensity = 0
        LedTriggerRoutine.instance().enable_light = True

    def main(self, dt, **pins):
        if self.light_intensity is None:
            return

        # pins['led_pwm_out'].write(self.light_intensity)
        LedTriggerRoutine.instance().light_intensity = self.light_intensity


    def _end(self):
        pass
