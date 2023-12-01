import numpy as np

import vxpy.core.attribute as vxattribute
import vxpy.core.control as vxcontrol
import vxpy.core.devices.serial as vxserial
import vxpy.core.container as vxcontainer
from devices.telemetrix_stepper import TelemetrixStepperKebab
import vxpy.core.container as vxcontainer


class Control01(vxcontrol.BaseControl):

    device: TelemetrixStepperKebab
    velocity: int = None    # [°/s]
    direction: int = None   # [CW:1, CCW: -1]
    led_left_state = 0.     # [%]
    led_right_state = 0.    # [%]
    rounds: int = None
    steps_to_full_rotation = 3200

    def __init__(self, *args, **kwargs):
        vxcontrol.BaseControl.__init__(self, *args, **kwargs)

        self.device = vxserial.get_serial_device_by_id('Dev_kebab')

    @staticmethod
    def _led_state_output(value: float) -> int:
            return int((value * 255))

    @staticmethod
    def _stepper_velocity_output(value: float) -> int:
        steps_to_full_rotation: int = 3200

        if value != 0:
            return int(value/(360/steps_to_full_rotation))
        else:
            return 0

    def _callback_phase_completion(self, data):
        self.device.board.stepper_set_current_position(self.device.motor, 0)
        self.device.board.analog_write(self.device.led_left, 0)
        self.device.board.analog_write(self.device.led_right, 0)

    def initialize(self, **kwargs):

        self.device.board.analog_write(self.device.led_left, self._led_state_output(self.led_left_state))
        self.device.board.analog_write(self.device.led_right, self._led_state_output(self.led_right_state))

        self.device.board.stepper_set_max_speed(self.device.motor, 1000)
        self.device.board.stepper_move_to(self.device.motor, int(self.direction * self.steps_to_full_rotation * self.rounds))
        self.device.board.stepper_set_speed(self.device.motor, int(self.direction * self._stepper_velocity_output(self.velocity)))
        self.device.board.stepper_run_speed_to_position(self.device.motor, completion_callback=self._callback_phase_completion)

        # Add datasets to be saved to file
        vxcontainer.add_phase_attributes({'velocity': self.velocity, 'direction': self.direction,
                                              'rounds': self.rounds, 'steps_to_full_rotation': self.steps_to_full_rotation})
        vxcontainer.create_phase_dataset('led_left', (1,), np.float32)
        vxcontainer.create_phase_dataset('led_right', (1,), np.float32)

    def main(self, dt, **pins):

        vxcontainer.add_to_phase_dataset('led_left', self.led_left_state)
        vxcontainer.add_to_phase_dataset('led_right', self.led_right_state)

    def _end(self):
        print('-----END')
        self.device.board.analog_write(self.device.led_left, 0)
        self.device.board.analog_write(self.device.led_right, 0)
        self.device.board.stepper_stop(self.device.motor)
        self.device.board.stepper_set_current_position(self.device.motor, 0)
