import itertools
import random
import numpy as np
import vxpy.core.protocol as vxprotocol
from controls import sutter_micromanipulator_controls
from visuals.ct_preyCaptureStereopsis import TwoEllipsesRotatingAroundAxis
from visuals.spherical_grating import SphericalBlackWhiteGrating
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground

def params_prey(pos_az, pos_el, pos_offset, mov_ax1, mov_ax2, mov_speed, el_ax1, el_ax2, el_mirror):
    return {TwoEllipsesRotatingAroundAxis.polarity: 2,
            TwoEllipsesRotatingAroundAxis.pos_azimut_angle: pos_az,
            TwoEllipsesRotatingAroundAxis.pos_elev_angle: pos_el,
            TwoEllipsesRotatingAroundAxis.movement_major_axis: mov_ax1,
            TwoEllipsesRotatingAroundAxis.movement_minor_axis: mov_ax2,
            TwoEllipsesRotatingAroundAxis.movement_angular_velocity: mov_speed,
            TwoEllipsesRotatingAroundAxis.el_diameter_horiz: el_ax1,
            TwoEllipsesRotatingAroundAxis.el_diameter_vert: el_ax2,
            TwoEllipsesRotatingAroundAxis.el_rotating: 1,
            TwoEllipsesRotatingAroundAxis.el_mirror: el_mirror,
            TwoEllipsesRotatingAroundAxis.el_color_r: 1.0,
            TwoEllipsesRotatingAroundAxis.el_color_g: 1.0,
            TwoEllipsesRotatingAroundAxis.el_color_b: 1.0,
            TwoEllipsesRotatingAroundAxis.pos_azimut_offset: pos_offset + 90,
            }

def pick_occluder_pos(keyword, occl_dict):
    # keyword = string or bool
    occluder_pos = 0
    if not keyword:
        occluder_pos = occl_dict['occlusion_all_open']
    elif keyword == 'left':
        occluder_pos = occl_dict['occlusion_left_open']
    elif keyword == 'right':
        occluder_pos = occl_dict['occlusion_right_open']
    elif keyword == 'middle':
        occluder_pos = occl_dict['occlusion_zero_pos']
    return occluder_pos


class ct_prey_capture_stereopsis_blue(vxprotocol.StaticProtocol):
    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        ################################# List of parameters: #################################
        red_conditions = False

        # fixed parameters
        prey_size = (11,5) # width x hight in deg
        prey_pos_sz = (21,25) # az, el in °
        prey_pos_out = (14,25)  # az, el in °  # need to adjust stimulus size?
        prey_pos_offset = 0 # °  # enter number if eyes not symetrically
        movement_path = (10,5) # width x hight in deg. eliptical path.
        movement_speed = 180 # deg/sec

        occlusion_zero_pos = 0   # stereopsis
        occlusion_left_open = -250   # left eye can see
        occlusion_right_open = 250  # right eye can see
        occlusion_all_open = 2000  # NEEDS TO BE IN Z (or y axis on manipulator) small means up

        phase_duration = 10 # s
        #move_duration_ori = 5 #s  soll individuell berechnet werden
        min_move_duration = 5
        sutter_speed = 200  # um/s -> take from sutter_micromanipulator_controls!
        num_repetitions = 3 #3

        rotation_spacial_freq = 1 / 0.06
        rotation_speed = 7.5 # angular velocity

        # changing parameters:
        parameters_changing = {'dot_binocular': [True, False],  # 1 or 2 dots presented
                               'pos_strike_zone': [True], # in or out of SZ
                               'occluder_up': [True, False, 'left', 'right'],
                               'left_eye_presentation': [True, False]
                               }

        ################################# Do stuff: ##########################################
        ######################################################################################

        occl_dict = {'occlusion_zero_pos': occlusion_zero_pos, 'occlusion_left_open': occlusion_left_open,
                     'occlusion_right_open': occlusion_right_open, 'occlusion_all_open': occlusion_all_open}

        ############### Make shuffled list of protocol ###############
        keys, values = zip(*parameters_changing.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # remove nonsense stimulus conditions
        for i, phase in enumerate(permutations_dicts):
            if phase['dot_binocular']:
                if phase['left_eye_presentation']:
                    permutations_dicts[i] = []
            #if not phase['dot_binocular'] and phase['occluder_up'] == 'left' or phase['occluder_up'] == 'right':
            #    permutations_dicts[i] = []
            # RED CONDITION
            if red_conditions:
               #if not phase['pos_strike_zone']:
               #    permutations_dicts[i] = []
               if not phase['dot_binocular']:
                   permutations_dicts[i] = []
               if phase['dot_binocular']:
                   if phase['occluder_up'] == 'left' or phase['occluder_up'] == 'right':
                       permutations_dicts[i] = []



        permutations_dicts = list(filter(None, permutations_dicts))

        # make copies of list -> number of repeats
        permutations_dicts *= num_repetitions

        # Random shuffle
        random.seed(0)
        random.shuffle(permutations_dicts)

        ################################# Execute protocol #################################
        #TODO: add short no movement phase to record zero position of occluder?
        for phase in permutations_dicts:
            this_phase = {}

            if phase['dot_binocular']:  # 2 dots
                this_phase['el_mirror'] = 1
                if phase['pos_strike_zone']:  # Position SZ
                    this_phase['prey_pos'] = prey_pos_sz
                if not phase['pos_strike_zone']:
                    this_phase['prey_pos'] = prey_pos_out

            if not phase['dot_binocular']:  # 1 dot
                this_phase['el_mirror'] = 0
                if phase['left_eye_presentation']:  # Mono: LINKES auge
                    if phase['pos_strike_zone']:  # SZ links
                        this_phase['prey_pos'] = (-prey_pos_sz[0], prey_pos_sz[1])
                    if not phase['pos_strike_zone']:
                        this_phase['prey_pos'] = (-prey_pos_sz[0], prey_pos_sz[1])

                if not phase['left_eye_presentation']:  # Mono: RECHTES auge
                    if phase['pos_strike_zone']:  # SZ rechts
                        this_phase['prey_pos'] = prey_pos_sz
                    if not phase['pos_strike_zone']:
                        this_phase['prey_pos'] = prey_pos_out
            this_phase['occluder_pos'] = pick_occluder_pos(phase['occluder_up'], occl_dict)  # Occlusion positions

            move_duration = abs(this_phase['occluder_pos']) / sutter_speed + 1
            if move_duration < min_move_duration:
                move_duration = min_move_duration

            # move micromanipulator
            p = vxprotocol.Phase(move_duration)
            p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([0, 0, 0])})
            if this_phase['occluder_pos'] == occlusion_all_open:
                p.set_control(sutter_micromanipulator_controls.ControlSutterMP,
                              {'move_to_x': 0, 'move_to_y': this_phase['occluder_pos'], 'move_to_z': 0})
            else:
                p.set_control(sutter_micromanipulator_controls.ControlSutterMP,
                              {'move_to_x': this_phase['occluder_pos'], 'move_to_y': 0, 'move_to_z': 0})
            self.add_phase(p)

            # show visual
            p = vxprotocol.Phase(phase_duration)
            p.set_visual(TwoEllipsesRotatingAroundAxis,
                             params_prey(this_phase['prey_pos'][0], this_phase['prey_pos'][1],
                                         prey_pos_offset,
                                         movement_path[0], movement_path[1],
                                         movement_speed,
                                         prey_size[0], prey_size[1],
                                         this_phase['el_mirror']))
            self.add_phase(p)

            # move micromanipulator back
            p = vxprotocol.Phase(move_duration)
            p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([0, 0, 0])})
            if this_phase['occluder_pos'] == occlusion_all_open:
                p.set_control(sutter_micromanipulator_controls.ControlSutterMP,
                              {'move_to_x': 0, 'move_to_y': this_phase['occluder_pos'] *-1, 'move_to_z': 0})
            else:
                p.set_control(sutter_micromanipulator_controls.ControlSutterMP,
                              {'move_to_x': this_phase['occluder_pos'] *-1, 'move_to_y': 0, 'move_to_z': 0})
            self.add_phase(p)



        # STREIFENMUSTER
        # Am Ende Streifenmuster Zeigen (nicht ganz sauber...)

        # move occlusion away
        p = vxprotocol.Phase(abs(occlusion_all_open) / sutter_speed + 1)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([0, 0, 0])})
        p.set_control(sutter_micromanipulator_controls.ControlSutterMP,
                      {'move_to_x': 0, 'move_to_y': occlusion_all_open, 'move_to_z': 0})
        self.add_phase(p)

        for i in range(0,2):  # repeat grating 2x
            # static grating
            p = vxprotocol.Phase(phase_duration)
            p.set_visual(SphericalBlackWhiteGrating,
                         {SphericalBlackWhiteGrating.waveform: 'rectangular',
                          SphericalBlackWhiteGrating.motion_axis: 'vertical',
                          SphericalBlackWhiteGrating.motion_type: 'rotation',
                          SphericalBlackWhiteGrating.angular_period: rotation_spacial_freq,
                          SphericalBlackWhiteGrating.angular_velocity: 0}
                         )
            self.add_phase(p)

            # rotation left
            p = vxprotocol.Phase(phase_duration)
            p.set_visual(SphericalBlackWhiteGrating,
                         {SphericalBlackWhiteGrating.waveform: 'rectangular',
                          SphericalBlackWhiteGrating.motion_axis: 'vertical',
                          SphericalBlackWhiteGrating.motion_type: 'rotation',
                          SphericalBlackWhiteGrating.angular_period: rotation_spacial_freq,
                          SphericalBlackWhiteGrating.angular_velocity: rotation_speed}
                         )
            self.add_phase(p)

            # static grating
            p = vxprotocol.Phase(phase_duration)
            p.set_visual(SphericalBlackWhiteGrating,
                         {SphericalBlackWhiteGrating.waveform: 'rectangular',
                          SphericalBlackWhiteGrating.motion_axis: 'vertical',
                          SphericalBlackWhiteGrating.motion_type: 'rotation',
                          SphericalBlackWhiteGrating.angular_period: rotation_spacial_freq,
                          SphericalBlackWhiteGrating.angular_velocity: 0}
                         )
            self.add_phase(p)

            # rotation right
            p = vxprotocol.Phase(phase_duration)
            p.set_visual(SphericalBlackWhiteGrating,
                         {SphericalBlackWhiteGrating.waveform: 'rectangular',
                          SphericalBlackWhiteGrating.motion_axis: 'vertical',
                          SphericalBlackWhiteGrating.motion_type: 'rotation',
                          SphericalBlackWhiteGrating.angular_period: rotation_spacial_freq,
                          SphericalBlackWhiteGrating.angular_velocity: -rotation_speed}
                         )
            self.add_phase(p)

        # move micromanipulator back and make dark again
        p = vxprotocol.Phase(abs(occlusion_all_open) / sutter_speed + 1)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([0, 0, 0])})
        p.set_control(sutter_micromanipulator_controls.ControlSutterMP,
                      {'move_to_x': 0, 'move_to_y': occlusion_all_open * -1, 'move_to_z': 0})
        self.add_phase(p)

######################################################################################################################
######################################################################################################################
######################################################################################################################
# Red protocol

class ct_prey_capture_stereopsis_red(vxprotocol.StaticProtocol):
    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        ################################# List of parameters: #################################
        red_conditions = True

        # fixed parameters
        prey_size = (11,5) # width x hight in deg
        prey_pos_sz = (21,25) # az, el in °
        prey_pos_out = (14,25)  # az, el in °  # need to adjust stimulus size?
        prey_pos_offset = 0 # °  # enter number if eyes not symetrically
        movement_path = (10,5) # width x hight in deg. eliptical path.
        movement_speed = 180 # deg/sec

        occlusion_zero_pos = 0   # stereopsis
        occlusion_left_open = -250   # left eye can see
        occlusion_right_open = 250  # right eye can see
        occlusion_all_open = 2000  # NEEDS TO BE IN Z (or y axis on manipulator) small means up

        phase_duration = 10 # s
        #move_duration_ori = 5 #s  soll individuell berechnet werden
        min_move_duration = 5
        sutter_speed = 200  # um/s -> take from sutter_micromanipulator_controls!
        num_repetitions = 3 #3

        rotation_spacial_freq = 1 / 0.06
        rotation_speed = 7.5 # angular velocity

        # changing parameters:
        parameters_changing = {'dot_binocular': [True, False],  # 1 or 2 dots presented
                               'pos_strike_zone': [True], # in or out of SZ
                               'occluder_up': [True, False, 'left', 'right'],
                               'left_eye_presentation': [True, False]
                               }

        ################################# Do stuff: ##########################################
        ######################################################################################

        occl_dict = {'occlusion_zero_pos': occlusion_zero_pos, 'occlusion_left_open': occlusion_left_open,
                     'occlusion_right_open': occlusion_right_open, 'occlusion_all_open': occlusion_all_open}

        ############### Make shuffled list of protocol ###############
        keys, values = zip(*parameters_changing.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # remove nonsense stimulus conditions
        for i, phase in enumerate(permutations_dicts):
            if phase['dot_binocular']:
                if phase['left_eye_presentation']:
                    permutations_dicts[i] = []
            #if not phase['dot_binocular'] and phase['occluder_up'] == 'left' or phase['occluder_up'] == 'right':
            #    permutations_dicts[i] = []
            # RED CONDITION
            if red_conditions:
               #if not phase['pos_strike_zone']:
               #    permutations_dicts[i] = []
               if not phase['dot_binocular']:
                   permutations_dicts[i] = []
               if phase['dot_binocular']:
                   if phase['occluder_up'] == 'left' or phase['occluder_up'] == 'right':
                       permutations_dicts[i] = []



        permutations_dicts = list(filter(None, permutations_dicts))

        # make copies of list -> number of repeats
        permutations_dicts *= num_repetitions

        # Random shuffle
        random.seed(0)
        random.shuffle(permutations_dicts)

        ################################# Execute protocol #################################
        for phase in permutations_dicts:
            this_phase = {}

            if phase['dot_binocular']:  # 2 dots
                this_phase['el_mirror'] = 1
                if phase['pos_strike_zone']:  # Position SZ
                    this_phase['prey_pos'] = prey_pos_sz
                if not phase['pos_strike_zone']:
                    this_phase['prey_pos'] = prey_pos_out

            if not phase['dot_binocular']:  # 1 dot
                this_phase['el_mirror'] = 0
                if phase['left_eye_presentation']:  # Mono: LINKES auge
                    if phase['pos_strike_zone']:  # SZ links
                        this_phase['prey_pos'] = (-prey_pos_sz[0], prey_pos_sz[1])
                    if not phase['pos_strike_zone']:
                        this_phase['prey_pos'] = (-prey_pos_sz[0], prey_pos_sz[1])

                if not phase['left_eye_presentation']:  # Mono: RECHTES auge
                    if phase['pos_strike_zone']:  # SZ rechts
                        this_phase['prey_pos'] = prey_pos_sz
                    if not phase['pos_strike_zone']:
                        this_phase['prey_pos'] = prey_pos_out
            this_phase['occluder_pos'] = pick_occluder_pos(phase['occluder_up'], occl_dict)  # Occlusion positions

            move_duration = abs(this_phase['occluder_pos']) / sutter_speed + 1
            if move_duration < min_move_duration:
                move_duration = min_move_duration

            # move micromanipulator
            p = vxprotocol.Phase(move_duration)
            p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([0, 0, 0])})
            if this_phase['occluder_pos'] == occlusion_all_open:
                p.set_control(sutter_micromanipulator_controls.ControlSutterMP,
                              {'move_to_x': 0, 'move_to_y': this_phase['occluder_pos'], 'move_to_z': 0})
            else:
                p.set_control(sutter_micromanipulator_controls.ControlSutterMP,
                              {'move_to_x': this_phase['occluder_pos'], 'move_to_y': 0, 'move_to_z': 0})
            self.add_phase(p)

            # show visual
            p = vxprotocol.Phase(phase_duration)
            p.set_visual(TwoEllipsesRotatingAroundAxis,
                             params_prey(this_phase['prey_pos'][0], this_phase['prey_pos'][1],
                                         prey_pos_offset,
                                         movement_path[0], movement_path[1],
                                         movement_speed,
                                         prey_size[0], prey_size[1],
                                         this_phase['el_mirror']))
            self.add_phase(p)

            # move micromanipulator back
            p = vxprotocol.Phase(move_duration)
            p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([0, 0, 0])})
            if this_phase['occluder_pos'] == occlusion_all_open:
                p.set_control(sutter_micromanipulator_controls.ControlSutterMP,
                              {'move_to_x': 0, 'move_to_y': this_phase['occluder_pos'] *-1, 'move_to_z': 0})
            else:
                p.set_control(sutter_micromanipulator_controls.ControlSutterMP,
                              {'move_to_x': this_phase['occluder_pos'] *-1, 'move_to_y': 0, 'move_to_z': 0})
            self.add_phase(p)



        # STREIFENMUSTER
        # Am Ende Streifenmuster Zeigen (nicht ganz sauber...)

        # move occlusion away
        p = vxprotocol.Phase(abs(occlusion_all_open) / sutter_speed + 1)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([0, 0, 0])})
        p.set_control(sutter_micromanipulator_controls.ControlSutterMP,
                      {'move_to_x': 0, 'move_to_y': occlusion_all_open, 'move_to_z': 0})
        self.add_phase(p)

        for i in range(0,2):  # repeat grating 2x
            # static grating
            p = vxprotocol.Phase(phase_duration)
            p.set_visual(SphericalBlackWhiteGrating,
                         {SphericalBlackWhiteGrating.waveform: 'rectangular',
                          SphericalBlackWhiteGrating.motion_axis: 'vertical',
                          SphericalBlackWhiteGrating.motion_type: 'rotation',
                          SphericalBlackWhiteGrating.angular_period: rotation_spacial_freq,
                          SphericalBlackWhiteGrating.angular_velocity: 0}
                         )
            self.add_phase(p)

            # rotation left
            p = vxprotocol.Phase(phase_duration)
            p.set_visual(SphericalBlackWhiteGrating,
                         {SphericalBlackWhiteGrating.waveform: 'rectangular',
                          SphericalBlackWhiteGrating.motion_axis: 'vertical',
                          SphericalBlackWhiteGrating.motion_type: 'rotation',
                          SphericalBlackWhiteGrating.angular_period: rotation_spacial_freq,
                          SphericalBlackWhiteGrating.angular_velocity: rotation_speed}
                         )
            self.add_phase(p)

            # static grating
            p = vxprotocol.Phase(phase_duration)
            p.set_visual(SphericalBlackWhiteGrating,
                         {SphericalBlackWhiteGrating.waveform: 'rectangular',
                          SphericalBlackWhiteGrating.motion_axis: 'vertical',
                          SphericalBlackWhiteGrating.motion_type: 'rotation',
                          SphericalBlackWhiteGrating.angular_period: rotation_spacial_freq,
                          SphericalBlackWhiteGrating.angular_velocity: 0}
                         )
            self.add_phase(p)

            # rotation right
            p = vxprotocol.Phase(phase_duration)
            p.set_visual(SphericalBlackWhiteGrating,
                         {SphericalBlackWhiteGrating.waveform: 'rectangular',
                          SphericalBlackWhiteGrating.motion_axis: 'vertical',
                          SphericalBlackWhiteGrating.motion_type: 'rotation',
                          SphericalBlackWhiteGrating.angular_period: rotation_spacial_freq,
                          SphericalBlackWhiteGrating.angular_velocity: -rotation_speed}
                         )
            self.add_phase(p)

        # move micromanipulator back and make dark again
        p = vxprotocol.Phase(abs(occlusion_all_open) / sutter_speed + 1)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([0, 0, 0])})
        p.set_control(sutter_micromanipulator_controls.ControlSutterMP,
                      {'move_to_x': 0, 'move_to_y': occlusion_all_open * -1, 'move_to_z': 0})
        self.add_phase(p)