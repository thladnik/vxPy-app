import time
from typing import Dict

import numpy as np
from PySide6 import QtWidgets

import vxpy.core.routine as vxroutine
from vxpy.core.devices import serial as vxserial
import vxpy.core.ui as vxui
from vxpy.utils import widgets


class LEdTriggerControl(vxroutine.IoRoutine):

    light_intensity = 0.5
    enable_light = False

    def __init__(self, *args, **kwargs):
        vxroutine.IoRoutine.__init__(self, *args, **kwargs)

    def initialize(self):
        pass

    def main(self, **pins: Dict[str, vxserial.DaqPin]):

        if self.enable_light:
            value = self.light_intensity
        else:
            value = 0.

        pins['led_pwm_out'].write(value)


class LEdTriggerControlUI(vxui.IoAddonWidget):

    def __init__(self, *args, **kwargs):
        vxui.IoAddonWidget.__init__(self, *args, **kwargs)

        self.central_widget.setLayout(QtWidgets.QVBoxLayout())

        self.enable_output = widgets.Checkbox(self, 'Output active',
                                              default=LEdTriggerControl.instance().enable_light)
        self.enable_output.connect_callback(self.set_enable_output)
        self.central_widget.layout().addWidget(self.enable_output)
        self.light_intensity = widgets.DoubleSliderWidget(self, 'Intensity',
                                                          default=LEdTriggerControl.instance().light_intensity,
                                                          limits=(0., 1.01), step_size=0.01)
        self.light_intensity.connect_callback(self.set_light_intensity)
        self.central_widget.layout().addWidget(self.light_intensity)

    def set_enable_output(self, value):
        LEdTriggerControl.instance().enable_light = value

    def set_light_intensity(self, value):
        LEdTriggerControl.instance().light_intensity = value
