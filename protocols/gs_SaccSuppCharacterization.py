import numpy as np

import vxpy.core.protocol as vxprotocol
from visuals.gs_characterization_stims.sft_grating import SphericalSFTGrating
from visuals.gs_characterization_stims.DS_grating import SphericalDSGrating
from visuals.gs_characterization_stims.uniform_flash import UniformFlashStep
from visuals.gs_characterization_stims.chirp import LogChirp
from visuals.single_moving_dot import SingleDotRotatingBackAndForth
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground

# define Parameter functions
def paramsSFGrat(waveform, motion_type, motion_axis, ang_vel, ang_period, offset):
    return {
        SphericalSFTGrating.waveform: waveform,
        SphericalSFTGrating.motion_type: motion_type,
        SphericalSFTGrating.motion_axis: motion_axis,
        SphericalSFTGrating.angular_velocity: ang_vel,
        SphericalSFTGrating.angular_period: ang_period,
        SphericalSFTGrating.offset: offset
    }

def paramsDSGrat(waveform, motion_type, motion_axis, ang_vel, ang_period):
    return {
        SphericalDSGrating.waveform: waveform,
        SphericalDSGrating.motion_type: motion_type,
        SphericalDSGrating.motion_axis: motion_axis,
        SphericalDSGrating.angular_velocity: ang_vel,
        SphericalDSGrating.angular_period: ang_period
    }

def paramsFlash(base_lum, flash_start, flash_dur, flash_amp):
    return {
        UniformFlashStep.baseline_lum: base_lum,
        UniformFlashStep.flash_start_time: flash_start,
        UniformFlashStep.flash_duration: flash_dur,
        UniformFlashStep.flash_amp: flash_amp
    }

def paramsChirp(base_lum, sine_amp, f_start, f_final, chirp_dur):
    return {
        LogChirp.baseline_lum: base_lum,
        LogChirp.sine_amp: sine_amp,
        LogChirp.starting_freq: f_start,
        LogChirp.final_freq: f_final,
        LogChirp.chirp_duration: chirp_dur
    }

def paramsDot(motion_axis, dot_pol, dot_start_ang, dot_ang_vel, dot_dia, dot_offset, t_switch):
    return {
        SingleDotRotatingBackAndForth.motion_axis: motion_axis,
        SingleDotRotatingBackAndForth.dot_polarity: dot_pol,
        SingleDotRotatingBackAndForth.dot_start_angle: dot_start_ang,
        SingleDotRotatingBackAndForth.dot_angular_velocity: dot_ang_vel,
        SingleDotRotatingBackAndForth.dot_angular_diameter: dot_dia,
        SingleDotRotatingBackAndForth.dot_offset_angle: dot_offset,
        SingleDotRotatingBackAndForth.t_switch: t_switch
    }

class CharacterizationProtocol(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        self.global_visual_props['azim_angle'] = 0.

        # set fixed parameters PART 1: SF Tuning Motion
        sfm_waveform = 'rectangular'
        sfm_motion_type = 'rotation'
        sft_motion_axis = 'vertical'
        sfm_temp_freq = 4    # Hz (cyc/sec)
        sfm_mov_phase_dur = 12 # sec
        sfm_static_phase_dur = 6 # sec

        # set fixed parameters PART 2: direction selectivity
        ds_waveform = 'rectangular'
        ds_ang_vel = 90 # °/sec
        ds_ang_period = 22.5  # °/cyc --> TF = 2 Hz (cyc/s)
        ds_mov_phase_dur = 12    # sec
        ds_static_phase_dur = 6 # sec

        # set fixed parameters PART 3a: On/Off Steps
        onoff_phase_dur = 12 # sec

        # set fixed parameters PART 3b: On/Off Flashes
        flash_base_lum = 0.5
        flash_start = 3 # in sec
        flash_dur = 0.5 # in sec
        flash_amp = 0.5 # change sign for bright or dark flashes
        flash_phase_dur = 8  # sec

        # set fixed parameters PART 4: SF Tuning Static
        sfs_waveform = 'rectangular'
        sfs_motion_type = 'rotation'
        sfs_motion_axis = 'vertical'
        sfs_pattern_phase_duration = 6   # sec
        sfs_grey_phase_duration = 6 # sec

        # set fixed parameters PART 5: Chirp down
        chirp_base_lum = 0.5
        chirp_sine_amp = 0.5
        chirp_f_start = 10  # Hz
        chirp_f_final = 0.5   # Hz
        chirp_dur = 20  # sec
        chirp_pause_dur = 6 # sec

        # set fixed parameters PART 6: Dot
        dot_motion_axis = 'vertical'
        dot_polarity = 'dark-on-light'
        dot_ang_vel = 120   # °/sec
        dot_grey_phase_duration = 6 # sec

        # Variable Conditions:
        SFTm_conditions = [(-1, 90), (1, 90), (-1, 45), (1, 45), (-1, 22.5), (1, 22.5), (-1, 11.25), (1, 11.25), (-1, 5.625), (1, 5.625)]   # direction and angular period in °
        DS_conditions = [(-1,'translation','forward'),(1,'translation','forward'),(-1,'translation','vertical'),
                        (1,'translation','vertical'),(-1,'rotation','forward'),(1,'rotation','forward')]  # direction, motion type and motion axis
        onoff_color = [0.5, 1, 0.5, -1]
        flash_direction = [-1,1]
        SFTs_conditions = [(90,0),(90,0.25),(90,0.5),(90,0.75),(45,0),(45,0.25),(45,0.5),(45,0.75),(22.5,0),(22.5,0.25),
                           (22.5,0.5),(22.5,0.75),(11.25,0),(11.25,0.25),(11.25,0.5),(11.25,0.75),(5.625,0),(5.625,0.25),
                           (5.625,0.5),(5.625,0.75)]    # angular period in °/cyc, phase shift in cyc
        dot_conditions = [(-90,45,5),(-90,45,30),(0,15,5),(0,15,30),(90,-15,5),(90,-15,30),(-180,15,5),(-180,15,30)]    # dot starting position in azimuth °, dot starting position in elevation °, dot diameter in °


        # Add pre-phase (5 sec uniform grey)
        p = vxprotocol.Phase(5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.5, .5, .5])})
        self.add_phase(p)


        # PART 1: Spatial Frequency Tuning, Motion
        repeats = 2
        for i in range(repeats):
            for direction, ang_per in np.random.permutation(SFTm_conditions):
                # Static Phase
                p = vxprotocol.Phase(sfm_static_phase_dur)
                p.set_visual(SphericalSFTGrating, paramsSFGrat(sfm_waveform, sfm_motion_type, sft_motion_axis,0, ang_per,0))
                self.add_phase(p)

                # Moving Phase
                p = vxprotocol.Phase(sfm_mov_phase_dur) # produces +0.5 pi phase shift, by running 0.0625 sec longer (at 4Hz TF, 1/4 cycle = 0.0625s)
                p.set_visual(SphericalSFTGrating, paramsSFGrat(sfm_waveform, sfm_motion_type, sft_motion_axis, direction * ang_per * sfm_temp_freq,
                                                               ang_per,0))
                self.add_phase(p)



        # 5 sec grey between characterization sections
        p = vxprotocol.Phase(5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.5, .5, .5])})
        self.add_phase(p)


        # PART 2: Direction Selectivity
        repeats = 2
        for i in range(repeats):
            for direction, motion_type, motion_axis in np.random.permutation(DS_conditions):
                direction = int(direction)
                # Static Phase
                p = vxprotocol.Phase(ds_static_phase_dur)
                p.set_visual(SphericalDSGrating, paramsDSGrat(ds_waveform, motion_type, motion_axis,
                                                                  0, ds_ang_period))
                self.add_phase(p)

                # Moving Phase
                p = vxprotocol.Phase(ds_mov_phase_dur)
                p.set_visual(SphericalDSGrating, paramsDSGrat(ds_waveform, motion_type, motion_axis,
                                                                  direction * ds_ang_vel, ds_ang_period))
                self.add_phase(p)



        # 5 sec grey between characterization sections
        p = vxprotocol.Phase(5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.5, .5, .5])})
        self.add_phase(p)


        # PART 3a: On/Off Steps
        repeats = 3
        for i in range(repeats):
            for color in onoff_color:
                p = vxprotocol.Phase(onoff_phase_dur)
                p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([color, color, color])})
                self.add_phase(p)


        # 5 sec grey between characterization sections
        p = vxprotocol.Phase(5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.5, .5, .5])})
        self.add_phase(p)


        # PART 3b: On/Off Flashes
        repeats = 10
        variants = np.array(flash_direction * repeats).reshape((-1,))
        for flash_dir in np.random.permutation(variants):  #(np.full((repeats-1,len(flash_direction)),flash_direction).reshape(len(flash_direction)*(repeats-1))):
            p = vxprotocol.Phase(flash_phase_dur)
            p.set_visual(UniformFlashStep, paramsFlash(flash_base_lum, flash_start, flash_dur, flash_dir * flash_amp))
            self.add_phase(p)


        # 5 sec grey between characterization sections
        p = vxprotocol.Phase(5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.5, .5, .5])})
        self.add_phase(p)


        # PART 4: Spatial Frequency Tuning, Static.
        for ang_per, phase_shift in np.random.permutation(SFTs_conditions):
            # Pattern Phase
            offset = ang_per * phase_shift
            p = vxprotocol.Phase(sfs_pattern_phase_duration)
            p.set_visual(SphericalSFTGrating,paramsSFGrat(sfs_waveform,sfs_motion_type,sfs_motion_axis,0,ang_per,offset))
            self.add_phase(p)

            # Grey Phase
            p = vxprotocol.Phase(sfs_grey_phase_duration)
            p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.5, .5, .5])})
            self.add_phase(p)

        # 5 sec grey between characterization sections
        p = vxprotocol.Phase(5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.5, .5, .5])})
        self.add_phase(p)


        # PART 5: Chirp Down
        repeats = 3
        for i in range(repeats):
            # Pause Phase
            p = vxprotocol.Phase(chirp_pause_dur)
            p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.5, .5, .5])})
            self.add_phase(p)

            # Chirp Phase
            p = vxprotocol.Phase(chirp_dur)
            p.set_visual(LogChirp, paramsChirp(chirp_base_lum,chirp_sine_amp,chirp_f_start,chirp_f_final,chirp_dur))
            self.add_phase(p)

        # 5 sec grey between characterization sections
        p = vxprotocol.Phase(5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.5, .5, .5])})
        self.add_phase(p)

        # PART 6: Dot
        repeats = 3
        for i in range(repeats):
            for azim_start, elev_start, dot_dia in np.random.permutation(dot_conditions):
                # Grey Phase
                p = vxprotocol.Phase(dot_grey_phase_duration)
                p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([1, 1, 1])})
                self.add_phase(p)

                # Dot Phase
                if dot_dia < 6:  # small dot
                    for elev_offset in [0, 5, 10, 15, 20, 25, 30]:
                        dot_phase_dur = 3 * np.cos((elev_start-elev_offset) * np.pi / 180)
                        p = vxprotocol.Phase(dot_phase_dur)
                        p.set_visual(SingleDotRotatingBackAndForth, paramsDot(dot_motion_axis, dot_polarity, azim_start,
                                                            dot_ang_vel/np.cos((elev_start-elev_offset) * np.pi / 180),
                                                            dot_dia, elev_start-elev_offset, dot_phase_dur/2))
                        self.add_phase(p)
                elif dot_dia > 6:   # large dot
                    for elev_offset in [0, 15, 30]:
                        dot_phase_dur = 3 * np.cos((elev_start-elev_offset) * np.pi / 180)
                        p = vxprotocol.Phase(dot_phase_dur)
                        p.set_visual(SingleDotRotatingBackAndForth, paramsDot(dot_motion_axis, dot_polarity, azim_start,
                                                            dot_ang_vel/np.cos((elev_start-elev_offset) * np.pi / 180),
                                                            dot_dia, elev_start-elev_offset, dot_phase_dur/2))
                        self.add_phase(p)


            # 5 sec grey between characterization sections
            p = vxprotocol.Phase(5)
            p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.5, .5, .5])})
            self.add_phase(p)
