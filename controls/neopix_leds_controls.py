import numpy as np

import vxpy.core.attribute as vxattribute
import vxpy.core.control as vxcontrol
import vxpy.core.devices.serial as vxserial
import vxpy.core.container as vxcontainer
from vxpy.devices.neopix_serial import NeopixSerial
import vxpy.core.container as vxcontainer


class ControlTest(vxcontrol.BaseControl):

    device: NeopixSerial
    led_num: int = None
    red_ch: int = 0
    green_ch: int = 0
    blue_ch: int = 0

    def __init__(self, *args, **kwargs):
        vxcontrol.BaseControl.__init__(self, *args, **kwargs)

        self.device = vxserial.get_serial_device_by_id('Dev_neopix')

    def initialize(self, **kwargs):

        #self.device.clear_pixels()
        self.device.set_led(self.led_num, [self.red_ch, self.green_ch, self.blue_ch])

        # Add datasets to be saved to file
        vxcontainer.add_phase_attributes({'velocity': self.led_num, 'red_ch': self.red_ch, 'green_ch': self.green_ch, 'blue_ch': self.blue_ch})

        #vxcontainer.create_phase_dataset('led_left', (1,), np.float32)
        #vxcontainer.create_phase_dataset('led_right', (1,), np.float32)

    def main(self, dt, **pins):

        pass

    def _end(self):
        self.device.clear_pixels()
        print('-----END PHASE')
