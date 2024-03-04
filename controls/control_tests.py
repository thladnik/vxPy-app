import numpy as np

import vxpy.core.attribute as vxattribute
import vxpy.core.control as vxcontrol
import vxpy.core.devices.serial as vxserial
import vxpy.core.container as vxcontainer


class TestControl01(vxcontrol.BaseControl):

    def initialize(self, **kwargs):
        pass

    def main(self, dt, **pins):
        vxserial.write_pin('led_ctrl1', np.random.rand() < 0.05)
        vxserial.write_pin('led_ctrl2', np.random.rand() < 0.05)

        vxcontainer.add_to_dataset('some_position', np.random.randint(100))
