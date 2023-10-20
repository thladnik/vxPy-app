import numpy as np

import vxpy.core.attribute as vxattribute
import vxpy.core.control as vxcontrol

class TestControl01(vxcontrol.BaseControl):

    def initialize(self, **kwargs):
        pass

    def main(self, dt):
        vxattribute.write_attribute('do_led_ctrl1', np.random.rand() < 0.01)
        vxattribute.write_attribute('do_led_ctrl2', np.random.rand() < 0.01)
