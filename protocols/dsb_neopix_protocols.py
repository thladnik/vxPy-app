import vxpy.core.protocol as vxprotocol
from controls import neopix_leds_controls
import vxpy.core.event as vxevent


class TestProtocol(vxprotocol.StaticProtocol):

    led_total_num = 21
    red_ch = 100
    green_ch = 0
    blue_ch = 0

    def create(self):

        for led_num in range(1, self.led_total_num+1):

            # illumination phase
            phase = vxprotocol.Phase(duration=3)
            phase.set_control(neopix_leds_controls.ControlTest,
                                  {'led_num': led_num, 'red_ch': self.red_ch, 'gree_ch': self.green_ch, 'blue_ch': self.blue_ch})
            self.add_phase(phase)


class DLR_protocol_tvw(vxprotocol.StaticProtocol):

    led_total_num = 21
    red_ch = 105    # 0 - 255 daylight: sRGB = (255,241,234)
    green_ch = 91
    blue_ch = 84

    def create(self):

        for led_num in range(1, self.led_total_num+1):

            #if led_num % 2 == 0:
                #continue

            # pause phase
            phase = vxprotocol.Phase(duration=3)
            phase.set_control(neopix_leds_controls.ControlTest,
                              {'led_num': led_num, 'red_ch': 0, 'green_ch': 0, 'blue_ch': 0})
            self.add_phase(phase)

            # illumination phase
            phase = vxprotocol.Phase(duration=9)
            phase.set_control(neopix_leds_controls.ControlTest,
                                  {'led_num': led_num, 'red_ch': self.red_ch, 'green_ch': self.green_ch, 'blue_ch': self.blue_ch})
            self.add_phase(phase)

