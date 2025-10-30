import vxpy.core.protocol as vxprotocol
from controls import fish_kebab_controls
import vxpy.core.event as vxevent

# total runtime = 1376s = ca. 23min


class VORTestTriggered(vxprotocol.TriggeredProtocol):

    rounds = 8
    direction = 1
    leds = 0.

    def __init__(self, *args, **kwargs):
        vxprotocol.TriggeredProtocol.__init__(self, *args, **kwargs)

        trigger = vxevent.OnTrigger('stepper_full_rotation_trigger')
        self.set_phase_trigger(trigger)

        for lights in range(2):

            for cycle in range(2):

                chill_phase = vxprotocol.Phase(duration=30)
                chill_phase.set_control(fish_kebab_controls.Control01, {'velocity': 0, 'direction': self.direction, 'rounds': 0,
                                                                        'led_left_state': self.leds, 'led_right_state': self.leds})
                self.add_phase(chill_phase)

                # 90°/s
                phase = vxprotocol.Phase(duration=self._min_phase_duration(90, self.rounds))
                phase.set_control(fish_kebab_controls.Control01, {'velocity': 90, 'direction': self.direction, 'rounds': self.rounds,
                                                                  'led_left_state': self.leds, 'led_right_state': self.leds})
                self.add_phase(phase)

                # chill phase
                chill_phase = vxprotocol.Phase(duration=30)
                chill_phase.set_control(fish_kebab_controls.Control01, {'velocity': 0, 'direction': self.direction, 'rounds': 0,
                                                                        'led_left_state': self.leds, 'led_right_state': self.leds})
                self.add_phase(chill_phase)

                self.direction *= -1

                # 45°/s
                phase = vxprotocol.Phase(duration=self._min_phase_duration(45, self.rounds))
                phase.set_control(fish_kebab_controls.Control01, {'velocity': 45, 'direction': self.direction, 'rounds': self.rounds,
                                                                  'led_left_state': self.leds, 'led_right_state': self.leds})
                self.add_phase(phase)

                # chill phase
                chill_phase = vxprotocol.Phase(duration=30)
                chill_phase.set_control(fish_kebab_controls.Control01, {'velocity': 0, 'direction': self.direction, 'rounds': 0,
                                                                        'led_left_state': self.leds, 'led_right_state': self.leds})
                self.add_phase(chill_phase)

                self.direction *= -1

                # 22.5°/s
                phase = vxprotocol.Phase(duration=self._min_phase_duration(22.5, self.rounds))
                phase.set_control(fish_kebab_controls.Control01, {'velocity': 22.5, 'direction': self.direction, 'rounds': self.rounds,
                                                                  'led_left_state': self.leds, 'led_right_state': self.leds})
                self.add_phase(phase)

                # chill phase
                chill_phase = vxprotocol.Phase(duration=30)
                chill_phase.set_control(fish_kebab_controls.Control01, {'velocity': 0, 'direction': self.direction, 'rounds': 0,
                                                                        'led_left_state': self.leds, 'led_right_state': self.leds})
                self.add_phase(chill_phase)

                self.direction *= -1

                # 11.25°/s
                phase = vxprotocol.Phase(duration=self._min_phase_duration(11.25, self.rounds))
                phase.set_control(fish_kebab_controls.Control01,
                                  {'velocity': 11.25, 'direction': self.direction, 'rounds': self.rounds,
                                   'led_left_state': self.leds, 'led_right_state': self.leds})
                self.add_phase(phase)

                # chill phase
                chill_phase = vxprotocol.Phase(duration=30)
                chill_phase.set_control(fish_kebab_controls.Control01,
                                        {'velocity': 0, 'direction': self.direction, 'rounds': 0,
                                         'led_left_state': self.leds, 'led_right_state': self.leds})
                self.add_phase(chill_phase)

            self.leds = 0.1

    def _min_phase_duration(self, velocity, rounds):
        return int(rounds * (360 / velocity))


class DLR_protocol_dsb(vxprotocol.StaticProtocol):

    rounds = 0
    direction = 1
    velocity = 0

    led_ratios = {
        'dark': [0.0, 0.0],
        '0°': [0.15, 0.15],
        '+90': [0.0, 0.3],
        '-67.5°': [0.225, 0.075],
        '+45°': [0.1, 0.2],
        '-22.5°': [0.175, 0.125],
        '-90°': [0.3, 0.0],
        '+67.5°': [0.075, 0.225],
        '-45°': [0.2, 0.1],
        '+22.5°': [0.125, 0.175]
    }

    def create(self):

        for i in range(2):

            for led_ratio in self.led_ratios.keys():

                # illumination phase
                phase = vxprotocol.Phase(duration=60)
                phase.set_control(fish_kebab_controls.Control01,
                                  {'velocity': self.velocity, 'direction': self.direction,
                                   'rounds': self.rounds,
                                   'led_left_state': self.led_ratios[led_ratio][0], 'led_right_state': self.led_ratios[led_ratio][1]})
                self.add_phase(phase)

                # chill phase
                phase = vxprotocol.Phase(duration=30)
                phase.set_control(fish_kebab_controls.Control01,
                                  {'velocity': self.velocity, 'direction': self.direction,
                                   'rounds': self.rounds,
                                   'led_left_state': 0.005, 'led_right_state': 0.005})
                self.add_phase(phase)


class VOR_protocol_tvw(vxprotocol.TriggeredProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.TriggeredProtocol.__init__(self, *args, **kwargs)

        trigger = vxevent.OnTrigger('stepper_full_rotation_trigger')
        self.set_phase_trigger(trigger)

    rounds = 8
    directions = [1, -1]
    velocities = [90, 45, 22.5]

    def _min_phase_duration(self, velocity, rounds):
        return int(rounds * (360 / velocity))

    def create(self):

        for velocity in self.velocities:
            for direction in self.directions:

                # chill phase
                chill_phase = vxprotocol.Phase(duration=30)
                chill_phase.set_control(fish_kebab_controls.Control01,
                                        {'velocity': 0, 'direction': direction, 'rounds': 0,
                                         'led_left_state': 0, 'led_right_state': 0})
                self.add_phase(chill_phase)

                # stim phase
                phase = vxprotocol.Phase(duration=self._min_phase_duration(velocity, self.rounds))
                phase.set_control(fish_kebab_controls.Control01,
                                  {'velocity': velocity, 'direction': direction, 'rounds': self.rounds,
                                   'led_left_state': 0, 'led_right_state': 0})
                self.add_phase(phase)


class DLR_dynamic_protocol_dsb(vxprotocol.StaticProtocol):

    rounds = 0
    direction = 1
    velocity = 0
    led_left_start_state = led_left_end_state = 0.075
    led_right_start_state = led_right_end_state = 0.075
    led_left_chill_state = led_right_chill_state = 0.075

    led_ratios = {
        #'dark': [0.0, 0.0],
        '0°': [0.075, 0.075],
        '+90': [0.0, 0.15],
        '-67.5°': [0.1125, 0.0375],
        '+45°': [0.05, 0.1],
        '-22.5°': [0.0875, 0.0625],
        '-90°': [0.15, 0.0],
        '+67.5°': [0.0375, 0.1125],
        '-45°': [0.1, 0.05],
        '+22.5°': [0.0625, 0.0875]
    }

    def create(self):

        for i in range(2):

            for led_ratio in self.led_ratios.keys():

                # illumination phase
                phase = vxprotocol.Phase(duration=60)
                phase.set_control(fish_kebab_controls.Control02,
                                  {'velocity': self.velocity, 'direction': self.direction,
                                   'rounds': self.rounds,
                                   'led_left_start_state': self.led_left_start_state,
                                   'led_left_state': self.led_ratios[led_ratio][0],
                                   'led_left_end_state': self.led_left_end_state,

                                   'led_right_start_state': self.led_right_start_state,
                                   'led_right_state': self.led_ratios[led_ratio][1],
                                   'led_right_end_state': self.led_right_end_state,

                                   'led_start_state_duration': 10,
                                   'led_middle_state_duration': 40,
                                   'led_end_state_duration': 10
                                   })
                self.add_phase(phase)

                # chill phase
                phase = vxprotocol.Phase(duration=30)
                phase.set_control(fish_kebab_controls.Control02,
                                  {'velocity': self.velocity, 'direction': self.direction,
                                   'rounds': self.rounds,
                                   'led_left_start_state': self.led_left_chill_state,
                                   'led_left_state': self.led_left_chill_state,
                                   'led_left_end_state': self.led_left_chill_state,

                                   'led_right_start_state': self.led_right_chill_state,
                                   'led_right_state': self.led_right_chill_state,
                                   'led_right_end_state': self.led_right_chill_state,

                                   'led_start_state_duration': 10,
                                   'led_middle_state_duration': 10,
                                   'led_end_state_duration': 10
                                   })
                self.add_phase(phase)

        # end phase
        phase = vxprotocol.Phase(duration=10)
        phase.set_control(fish_kebab_controls.Control01, {'velocity': 0, 'direction': 1, 'rounds': 0,
                                                          'led_left_state': 0., 'led_right_state': 0.})
        self.add_phase(phase)


class SetupDemonstration(vxprotocol.TriggeredProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.TriggeredProtocol.__init__(self, *args, **kwargs)

        trigger = vxevent.OnTrigger('stepper_full_rotation_trigger')
        self.set_phase_trigger(trigger)

        for step in range(3):

            # chill phase
            chill_phase = vxprotocol.Phase(duration=5)
            chill_phase.set_control(fish_kebab_controls.Control01,
                                    {'velocity': 0, 'direction': 1, 'rounds': 0,
                                     'led_left_state': 0.0, 'led_right_state': 0.0})
            self.add_phase(chill_phase)

            # 45°/s
            phase = vxprotocol.Phase(duration=self._min_phase_duration(45, 1))
            phase.set_control(fish_kebab_controls.Control01,
                                    {'velocity': 45, 'direction': 1, 'rounds': 1,
                                        'led_left_state': 1.0, 'led_right_state': 1.0})
            self.add_phase(phase)

            # chill phase
            chill_phase = vxprotocol.Phase(duration=3)
            chill_phase.set_control(fish_kebab_controls.Control01,
                                    {'velocity': 0, 'direction': 1, 'rounds': 0,
                                        'led_left_state': 0.0, 'led_right_state': 0.0})
            self.add_phase(chill_phase)

            # 90°/s
            phase = vxprotocol.Phase(duration=self._min_phase_duration(90, 2))
            phase.set_control(fish_kebab_controls.Control01,
                                    {'velocity': 90, 'direction': -1, 'rounds': 3,
                                        'led_left_state': 0.7, 'led_right_state': 0.1})
            self.add_phase(phase)

            # 22.5°/s
            phase = vxprotocol.Phase(duration=self._min_phase_duration(22.5, 1))
            phase.set_control(fish_kebab_controls.Control01,
                              {'velocity': 22.5, 'direction': 1, 'rounds': 1,
                               'led_left_state': 0.0, 'led_right_state': 0.5})
            self.add_phase(phase)

            # chill phase
            chill_phase = vxprotocol.Phase(duration=3)
            chill_phase.set_control(fish_kebab_controls.Control01,
                                    {'velocity': 0, 'direction': 1, 'rounds': 0,
                                     'led_left_state': 0.2, 'led_right_state': 0.2})
            self.add_phase(chill_phase)

            # chill phase
            chill_phase = vxprotocol.Phase(duration=2)
            chill_phase.set_control(fish_kebab_controls.Control01,
                                    {'velocity': 0, 'direction': 1, 'rounds': 0,
                                     'led_left_state': 0.0, 'led_right_state': 0.0})
            self.add_phase(chill_phase)

    def _min_phase_duration(self, velocity, rounds):
        return int(rounds * (360 / velocity))



class Protocol01Static(vxprotocol.StaticProtocol):

    def create(self):

        phase = vxprotocol.Phase(duration=30)
        phase.set_control(fish_kebab_controls.Control01, {'velocity': 45, 'direction': 1, 'rounds': 3,
                                                          'led_left_state': 1., 'led_right_state': 0.})
        self.add_phase(phase)

        phase = vxprotocol.Phase(duration=30)
        phase.set_control(fish_kebab_controls.Control01, {'velocity': 45, 'direction': -1, 'rounds': 3,
                                                          'led_left_state': 1., 'led_right_state': 0.})
        self.add_phase(phase)


class SinusoidalVOR(vxprotocol.StaticProtocol):

    frequencies = [0.0078125,0.015625,0.03125,0.0625, 0.125, 0.25,2]    #0.0078125,0.015625,0.03125,0.0625, 0.125, 0.25 (initially by set up by David)
    amplitude = 60
    rounds = 8

    def create(self):

        # chill phase
        phase = vxprotocol.Phase(duration=10)
        phase.set_control(fish_kebab_controls.ControlSinusoidal, {'rounds': 0, 'frequency': 0, 'amplitude': 0,
                                                                  'led_left_state': 0., 'led_right_state': 0.})
        self.add_phase(phase)

        for frequency in self.frequencies:

            print(frequency)

            phase = vxprotocol.Phase(duration=(1/frequency)*self.rounds + 5)
            phase.set_control(fish_kebab_controls.ControlSinusoidal, {'rounds': self.rounds, 'frequency': frequency, 'amplitude': self.amplitude,
                                                              'led_left_state': 0., 'led_right_state': 0.})
            self.add_phase(phase)

            # chill phase
            phase = vxprotocol.Phase(duration=10)
            phase.set_control(fish_kebab_controls.ControlSinusoidal,{'rounds': 0, 'frequency':0, 'amplitude': 0,
                                                                     'led_left_state': 0., 'led_right_state': 0.})
            self.add_phase(phase)


class DiscreteStepperVOR(vxprotocol.StaticProtocol):

    start_angle = current_angle = 0
    stop_angle = 180     # [°]
    step_size = 10       # [°]
    direction = 1
    phase_duration = 6     # [s]
    anz_executions = 4

    def create(self):

        # chill phase
        phase = vxprotocol.Phase(duration=30)
        phase.set_control(fish_kebab_controls.ControlDiscretePositions, {'velocity': 0, 'direction': 1,
                                                                         'target_angle': 0,
                                                                         'current_angle': 0,
                                                                         'led_left_state': 0., 'led_right_state': 0.})
        self.add_phase(phase)


        # for positive and negative angles
        for i in range(self.anz_executions * 2):

            # moving to stop angle
            while self.current_angle < self.stop_angle:

                self.current_angle += self.step_size

                phase = vxprotocol.Phase(duration=self.phase_duration)
                phase.set_control(fish_kebab_controls.ControlDiscretePositions, {'velocity': 5, 'direction': self.direction, 'target_angle': self.current_angle * self.direction,
                                                                                 'current_angle': self.step_size, 'led_left_state': 0., 'led_right_state': 0.})
                self.add_phase(phase)

            self.direction *= -1

            # moving backwards to start angle
            while self.current_angle > self.start_angle:

                self.current_angle -= self.step_size

                phase = vxprotocol.Phase(duration=self.phase_duration)
                phase.set_control(fish_kebab_controls.ControlDiscretePositions, {'velocity': 5, 'direction': self.direction, 'target_angle': self.current_angle * self.direction,
                                                                                 'current_angle': self.step_size, 'led_left_state': 0., 'led_right_state': 0.})
                self.add_phase(phase)

        # chill phase
        phase = vxprotocol.Phase(duration=30)
        phase.set_control(fish_kebab_controls.ControlDiscretePositions, {'velocity': 0, 'direction': 1,
                                                                         'target_angle': 0,
                                                                         'current_angle': 0,
                                                                         'led_left_state': 0., 'led_right_state': 0.})
        self.add_phase(phase)
