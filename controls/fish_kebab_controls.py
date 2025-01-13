import numpy as np
import time

import vxpy.core.attribute as vxattribute
import vxpy.core.control as vxcontrol
import vxpy.core.devices.serial as vxserial
import vxpy.core.container as vxcontainer
from devices.telemetrix_stepper import TelemetrixStepperKebab
import vxpy.core.container as vxcontainer

from vxpy.definitions import *
import vxpy.core.ipc as vxipc


class Control01(vxcontrol.BaseControl):
    device: TelemetrixStepperKebab
    velocity: int = None  # [°/s]
    direction: int = None  # [CW:1, CCW: -1]
    led_left_state = 0.  # [%]
    led_right_state = 0.  # [%]
    rounds: int = None
    steps_to_full_rotation: int = 3200

    # LED werte für start, mitte und Ende für Rechts und links übergeben

    def __init__(self, *args, **kwargs):
        vxcontrol.BaseControl.__init__(self, *args, **kwargs)

        self.device = vxserial.get_serial_device_by_id('Dev_kebab')

    def _led_state_output(self, value: float) -> int:
        return int((value * 255))

    def _stepper_velocity_output(self, value: float) -> int:

        if value != 0:
            return int(value / (360 / self.steps_to_full_rotation))
        else:
            return 0

    def _callback_phase_completion(self, data):
        self.device.board.stepper_set_current_position(self.device.motor, 0)

    def initialize(self, **kwargs):

        self.start_time = 0

        self.device.board.analog_write(self.device.led_left, self._led_state_output(self.led_left_state))
        self.device.board.analog_write(self.device.led_right, self._led_state_output(self.led_right_state))

        self.device.board.stepper_set_max_speed(self.device.motor, 1000)
        self.device.board.stepper_move_to(self.device.motor,
                                          int(self.direction * self.steps_to_full_rotation * self.rounds))
        self.device.board.stepper_set_speed(self.device.motor,
                                            int(self.direction * self._stepper_velocity_output(self.velocity)))
        self.device.board.stepper_run_speed_to_position(self.device.motor,
                                                        completion_callback=self._callback_phase_completion)

        # Add datasets to be saved to file
        vxcontainer.add_phase_attributes({'velocity': self.velocity, 'direction': self.direction,
                                          'rounds': self.rounds, 'steps_to_full_rotation': self.steps_to_full_rotation})
        vxcontainer.create_phase_dataset('led_left', (1,), np.float32)
        vxcontainer.create_phase_dataset('led_right', (1,), np.float32)

    def main(self, dt, **pins):

        vxcontainer.add_to_phase_dataset('led_left', self.led_left_state)
        vxcontainer.add_to_phase_dataset('led_right', self.led_right_state)

    def _end(self):
        print('-----END PHASE')
        self.device.board.analog_write(self.device.led_left, 0)
        self.device.board.analog_write(self.device.led_right, 0)
        self.device.board.stepper_stop(self.device.motor)
        self.device.board.stepper_set_current_position(self.device.motor, 0)


class Control02(vxcontrol.BaseControl):
    device: TelemetrixStepperKebab
    velocity: int = None  # [°/s]
    direction: int = None  # [CW:1, CCW: -1]
    current_time: float = None

    led_left_start_state: float = 0.
    led_left_end_state: float = 0.
    led_left_state: float = 0.  # [%]
    led_right_start_state: float = 0.
    led_right_end_state: float = 0.
    led_right_state: float = 0.  # [%]
    led_start_state_duration: int = 0.
    led_end_state_duration: int = 0.
    led_middle_state_duration: int = 0.
    left_led = None
    right_led = None
    led_left_current_state: float = 0.
    led_right_current_state: float = 0.
    time = None
    rounds: int = None
    steps_to_full_rotation: int = 3200

    def __init__(self, *args, **kwargs):
        vxcontrol.BaseControl.__init__(self, *args, **kwargs)

        self.device = vxserial.get_serial_device_by_id('Dev_kebab')

    def _led_state_output(self, value: float) -> int:
        return int((value * 255))

    def _stepper_velocity_output(self, value: float) -> int:

        if value != 0:
            return int(value / (360 / self.steps_to_full_rotation))
        else:
            return 0

    def _callback_phase_completion(self, data):
        self.device.board.stepper_set_current_position(self.device.motor, 0)

    def initialize(self, **kwargs):

        self.current_time = 0.
        self.led_left_current_state = self.led_left_start_state
        self.led_right_current_state = self.led_right_start_state

        self.device.board.analog_write(self.device.led_left, self._led_state_output(self.led_left_start_state))
        self.device.board.analog_write(self.device.led_right, self._led_state_output(self.led_right_start_state))

        self.device.board.stepper_set_max_speed(self.device.motor, 1000)
        self.device.board.stepper_move_to(self.device.motor,
                                          int(self.direction * self.steps_to_full_rotation * self.rounds))
        self.device.board.stepper_set_speed(self.device.motor,
                                            int(self.direction * self._stepper_velocity_output(self.velocity)))
        self.device.board.stepper_run_speed_to_position(self.device.motor,
                                                        completion_callback=self._callback_phase_completion)

        # Add datasets to be saved to file
        vxcontainer.add_phase_attributes({'velocity': self.velocity, 'direction': self.direction,
                                          'rounds': self.rounds, 'steps_to_full_rotation': self.steps_to_full_rotation})
        vxcontainer.create_phase_dataset('led_left', (1,), np.float32)
        vxcontainer.create_phase_dataset('led_right', (1,), np.float32)

    def main(self, dt, **pins):

        self.current_time += dt

        if self.current_time < self.led_start_state_duration:
            self.led_left_current_state += (
                        ((self.led_left_state - self.led_left_start_state) / self.led_start_state_duration) * dt)
            self.led_right_current_state += (
                        ((self.led_right_state - self.led_right_start_state) / self.led_start_state_duration) * dt)

        elif self.current_time > self.led_start_state_duration + self.led_middle_state_duration:
            self.led_left_current_state += (
                        ((self.led_left_end_state - self.led_left_state) / self.led_end_state_duration) * dt)
            self.led_right_current_state += (
                        ((self.led_right_end_state - self.led_right_state) / self.led_end_state_duration) * dt)

        else:
            self.led_left_current_state = self.led_left_state
            self.led_right_current_state = self.led_right_state

        self.device.board.analog_write(self.device.led_left, self._led_state_output(self.led_left_current_state))
        self.device.board.analog_write(self.device.led_right, self._led_state_output(self.led_right_current_state))

        vxcontainer.add_to_phase_dataset('led_left', self.led_left_current_state)
        vxcontainer.add_to_phase_dataset('led_right', self.led_right_current_state)

    def _end(self):
        print('-----END PHASE')
        self.device.board.analog_write(self.device.led_left, self._led_state_output(self.led_left_end_state))
        self.device.board.analog_write(self.device.led_right, self._led_state_output(self.led_right_end_state))
        self.device.board.stepper_stop(self.device.motor)
        self.device.board.stepper_set_current_position(self.device.motor, 0)


class ControlSinusoidal(vxcontrol.BaseControl):
    device: TelemetrixStepperKebab
    frequency: float = None  # [Hz]
    amplitude: int = None  # [°]
    rounds: int = None
    round_counter: int = None
    delta_t: float = None  # [s]
    led_left_state = 0.  # [%]
    led_right_state = 0.  # [%]
    target_pos_steps: int = None  # steps
    steps_to_full_rotation: int = 3200

    running: bool = None
    current_pos: int = None
    old_pos: int = None
    new_pos_received: bool = None

    def __init__(self, *args, **kwargs):
        vxcontrol.BaseControl.__init__(self, *args, **kwargs)

        self.device = vxserial.get_serial_device_by_id('Dev_kebab')

    def _led_state_output(self, value: float) -> int:
        return int((value * 255))

    def _stepper_velocity_output(self, value: float) -> int:

        if value != 0:
            return int(value / (360 / self.steps_to_full_rotation))
        else:
            return 0

    def _callback_move_completion(self, data):
        if data[1]:
            self.running = True
        else:
            self.running = False

    def _callback_current_pos(self, data):
        self.current_pos = data[2]

    def initialize(self, **kwargs):

        self.delta_t = 0.
        self.running = False
        self.round_counter = 0
        self.current_pos = 1
        self.old_pos = 1
        self.new_pos_received = False

        self.device.board.analog_write(self.device.led_left, self._led_state_output(self.led_left_state))
        self.device.board.analog_write(self.device.led_right, self._led_state_output(self.led_right_state))

        self.device.board.stepper_set_max_speed(self.device.motor, 1000)
        self.device.board.stepper_set_acceleration(self.device.motor, 1000)
        self.device.board.stepper_set_current_position(self.device.motor, 0)

        # Add datasets to be saved to file
        vxcontainer.add_phase_attributes(
            {'rounds': self.rounds, 'frequency [Hz]': self.frequency, 'amplitude [°]': self.amplitude,
             'steps_to_full_rotation': self.steps_to_full_rotation})
        vxcontainer.create_phase_dataset('led_left', (1,), np.float32)
        vxcontainer.create_phase_dataset('led_right', (1,), np.float32)
        vxcontainer.create_phase_dataset('motor_pos [°]', (1,), np.float32)


    def main(self, dt, **pins):

        if self.round_counter >= self.rounds * 2:
            return

        self.delta_t += dt

        # get current motor position in degree
        if self.delta_t > 0.1:
            self.device.board.stepper_get_current_position(self.device.motor, current_position_callback=self._callback_current_pos)

        # check if zero-position is reached
        if self.current_pos * self.old_pos < 0 or self.current_pos == 0:
            self.round_counter += 1
            print('times zero-position is reached: ', self.round_counter)

            # check if break condition (x rounds) is reached
            if self.round_counter >= self.rounds * 2:
                self.device.board.stepper_stop(self.device.motor)
                self.device.board.stepper_move_to(self.device.motor, 0)
                self.device.board.stepper_set_speed(self.device.motor, 800)
                self.device.board.stepper_run_speed_to_position(self.device.motor,
                                                                completion_callback=self._callback_move_completion)
                time.sleep(0.5)
                vxipc.CONTROL[CTRL_PRCL_PHASE_END_TIME] = vxipc.get_time()
                return
        self.old_pos = self.current_pos

        # set next position
        if not self.running:
            self.target_pos_steps = int(self.amplitude * np.sin(2 * np.pi * self.frequency * self.delta_t) * (
                        self.steps_to_full_rotation / 360))
            self.device.board.stepper_move_to(self.device.motor, self.target_pos_steps)
            self.device.board.stepper_run(self.device.motor, completion_callback=self._callback_move_completion)

        # store data in HDF5 file
        vxcontainer.add_to_phase_dataset('led_left', self.led_left_state)
        vxcontainer.add_to_phase_dataset('led_right', self.led_right_state)
        vxcontainer.add_to_phase_dataset('motor_pos [°]', (self.current_pos / self.steps_to_full_rotation) * 360)

    def _end(self):
        print('-----END PHASE')
        self.device.board.analog_write(self.device.led_left, 0)
        self.device.board.analog_write(self.device.led_right, 0)
        self.device.board.stepper_stop(self.device.motor)
        self.device.board.stepper_set_current_position(self.device.motor, 0)


class ControlDiscretePositions(vxcontrol.BaseControl):
    device: TelemetrixStepperKebab
    velocity: int = None  # [°/s]
    direction: int = None  # [CW:1, CCW: -1]
    led_left_state = 0.  # [%]
    led_right_state = 0.  # [%]
    target_angle: int = None
    current_angle: int = None
    steps_to_full_rotation: int = 3200

    # LED werte für start, mitte und Ende für Rechts und links übergeben

    def __init__(self, *args, **kwargs):
        vxcontrol.BaseControl.__init__(self, *args, **kwargs)

        self.device = vxserial.get_serial_device_by_id('Dev_kebab')

    def _led_state_output(self, value: float) -> int:
        return int((value * 255))

    def _stepper_velocity_output(self, value: float) -> int:

        if value != 0:
            return int(value / (360 / self.steps_to_full_rotation))
        else:
            return 0

    def _callback_phase_completion(self, data):
        self.device.board.stepper_set_current_position(self.device.motor, 0)

    def initialize(self, **kwargs):

        self.start_time = 0

        self.device.board.analog_write(self.device.led_left, self._led_state_output(self.led_left_state))
        self.device.board.analog_write(self.device.led_right, self._led_state_output(self.led_right_state))

        self.device.board.stepper_set_max_speed(self.device.motor, 1000)
        self.device.board.stepper_move_to(self.device.motor, int(self.direction * (
                    self.steps_to_full_rotation / 360) * self.current_angle))
        self.device.board.stepper_set_speed(self.device.motor,
                                            int(self.direction * self._stepper_velocity_output(self.velocity)))
        self.device.board.stepper_run_speed_to_position(self.device.motor,
                                                        completion_callback=self._callback_phase_completion)

        # Add datasets to be saved to file
        vxcontainer.add_phase_attributes(
            {'velocity': self.velocity, 'direction': self.direction, 'target_angle': self.target_angle,
             'steps_to_full_rotation': self.steps_to_full_rotation})
        vxcontainer.create_phase_dataset('led_left', (1,), np.float32)
        vxcontainer.create_phase_dataset('led_right', (1,), np.float32)

    def main(self, dt, **pins):

        vxcontainer.add_to_phase_dataset('led_left', self.led_left_state)
        vxcontainer.add_to_phase_dataset('led_right', self.led_right_state)

    def _end(self):
        print('-----END PHASE')
        self.device.board.analog_write(self.device.led_left, 0)
        self.device.board.analog_write(self.device.led_right, 0)
        self.device.board.stepper_stop(self.device.motor)
        self.device.board.stepper_set_current_position(self.device.motor, 0)
