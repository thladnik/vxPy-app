import numpy as np

import vxpy.core.attribute as vxattribute
import vxpy.core.control as vxcontrol
import vxpy.core.devices.serial as vxserial
import vxpy.core.container as vxcontainer


class Control01(vxcontrol.BaseControl):

    def __init__(self, *args, **kwargs):
        vxcontrol.BaseControl.__init__(self, *args, **kwargs)

        self.device = vxserial.get_serial_device_by_id('Dev_kebab')

    def initialize(self, **kwargs):
        pass

    def main(self, dt, **pins):
        vxserial.write_pin('led_ctrl1', np.random.rand() < 0.05)
        vxserial.write_pin('led_ctrl2', np.random.rand() < 0.05)

        vxcontainer.add_to_dataset('some_position', np.random.randint(100))
