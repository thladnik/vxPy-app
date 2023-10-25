import numpy as np

import vxpy.core.protocol as vxprotocol
from visuals.gs_characterization_stims.sft_grating import SphericalSFTGrating
from visuals.gs_characterization_stims.DS_grating import SphericalDSGrating
from visuals.gs_characterization_stims.uniform_flash import UniformFlashStep
from visuals.single_moving_dot import SingleDotRotatingSpiral
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground

# define Parameter functions
def paramsSFGrat(waveform, motion_type, ang_vel, ang_period):
    return {
        SphericalSFTGrating.waveform: waveform,
        SphericalSFTGrating.motion_type: motion_type,
        SphericalSFTGrating.angular_velocity: ang_vel,
        SphericalSFTGrating.angular_period: ang_period
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


def paramsDot(motion_axis, dot_pol, dot_start_ang, dot_ang_vel, dot_dia, elev_vel, elev_start):
    return {
        SingleDotRotatingSpiral.motion_axis: motion_axis,
        SingleDotRotatingSpiral.dot_polarity: dot_pol,
        SingleDotRotatingSpiral.dot_start_angle: dot_start_ang,
        SingleDotRotatingSpiral.dot_angular_velocity: dot_ang_vel,
        SingleDotRotatingSpiral.dot_angular_diameter: dot_dia,
        SingleDotRotatingSpiral.elevation_vel: elev_vel,
        SingleDotRotatingSpiral.elevation_start: elev_start
    }

class CharacterizationProtocol(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        self.global_visual_props['azim_angle'] = 0.

        # set fixed parameters PART 1: SF Tuning
        sf_waveform = 'rectangular'
        sf_motion_type = 'rotation'
        sf_temp_freq = 4    # Hz (cyc/sec)
        sf_phase_dur = 6 # sec

        # set fixed parameters PART 2: direction selectivity
        ds_waveform = 'rectangular'
        ds_motion_axis = 'vertical'
        ds_ang_vel = 60 # °/sec
        ds_ang_period = 30  # °/cyc --> TF = 2 Hz (cyc/s)
        ds_phase_dur = 6    # sec

        # set fixed parameters PART 3: On/Off
        onoff_phase_dur = 6 # sec

        # set fixed parameters PART 4: Flashes
        base_lum = 0.5
        flash_start = 3 # in sec
        flash_dur = 0.5 # in sec
        flash_amp = 0.5 # change sign for bright or dark flashes
        flash_phase_dur = 6  # sec

        # set fixed parameters PART 5: Dot
        dot_motion_axis = 'vertical'
        dot_polarity = 'dark-on-light'
        dot_start_angle = 0 # °
        dot_ang_vel = 120   # °/sec (on elevation axis) --> 3s/360° = 3s/cyc
        dot_dia_small = 5   # °
        dot_dia_big = 30  # °
        dot_small_elev_vel = 10/3   # °/s
        dot_big_elev_vel = 10   # °/s
        small_dot_phase_dur = 27  # sec; 90° elevation range @ 10° elevation gain/cycle = 9 cyc * 3s/cyc
        big_dot_phase_dur = 12  # sec, 120° elevation range @ 30° elevation gain/cycle = 4 cyc * 3s/cyc

        # Variable Conditions:
        SFTangular_periods = [90, 45, 22.5, 11.25, 5.625]   # PART 1: SF Tuning
        DSmotion_type = ['translation','rotation']  # PART 2: Direction Selectivity
        onoff_color = [0.5, 1, 0.5, -1] # PART 3: On/Off
        flash_direction = [-1,1]  # PART 4: Flashes
        dot_small_directions = [(1,-45), (-1,45)]  # ° (for upward moving (+1), start @ -45, for downward moving (-1) start @ +45)
        dot_big_directions = [(1,-90), (-1,30)]  # ° (for upward moving (+1), start @ -90, for downward moving (-1) start @ 30)

        # Add pre-phase (5 sec uniform grey)
        p = vxprotocol.Phase(5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.5, .5, .5])})
        self.add_phase(p)


        # PART 1: Spatial Frequency Tuning
        repeats = 2
        for i in range(repeats):
            for ang_per in SFTangular_periods:
                self.global_visual_props['azim_angle'] = 0.
                # Static Phase, 0 pi phase shifted
                p = vxprotocol.Phase(sf_phase_dur)
                p.set_visual(SphericalSFTGrating, paramsSFGrat(sf_waveform, sf_motion_type, 0, ang_per))
                self.add_phase(p)

                # Moving Phase CW_1
                p = vxprotocol.Phase(sf_phase_dur + 0.0625) # produces +0.5 pi phase shift, by running 0.0625 sec longer (at 4Hz TF, 1/4 cycle = 0.0625s)
                p.set_visual(SphericalSFTGrating, paramsSFGrat(sf_waveform, sf_motion_type, ang_per * sf_temp_freq,
                                                               ang_per))
                self.add_phase(p)

                # Static Phase, 0.5 pi phase shifted
                p = vxprotocol.Phase(sf_phase_dur)
                p.set_visual(SphericalSFTGrating, paramsSFGrat(sf_waveform, sf_motion_type, 0, ang_per))
                self.add_phase(p)

                # Moving Phase CCW_1
                p = vxprotocol.Phase(sf_phase_dur + 0.125) # produces -1 pi phase shift, by running 0.125 sec longer (at 4Hz TF, 1/2 cycle = 0.125s)
                p.set_visual(SphericalSFTGrating, paramsSFGrat(sf_waveform, sf_motion_type, -ang_per * sf_temp_freq,
                                                               ang_per))
                self.add_phase(p)

                # Static Phase, -0.5 pi == 1.5 pi phase shifted
                p = vxprotocol.Phase(sf_phase_dur)
                p.set_visual(SphericalSFTGrating, paramsSFGrat(sf_waveform, sf_motion_type, 0, ang_per))
                self.add_phase(p)

                # Moving Phase CW_2
                p = vxprotocol.Phase(sf_phase_dur - 0.0625)  # produces -0.5 pi phase shift, by running 0.0625 sec shorter (at 4Hz TF, 1/4 cycle = 0.0625s)
                p.set_visual(SphericalSFTGrating, paramsSFGrat(sf_waveform, sf_motion_type, ang_per * sf_temp_freq,
                                                               ang_per))
                self.add_phase(p)

                # Static Phase, -1 pi == 1 pi phase shifted
                p = vxprotocol.Phase(sf_phase_dur)
                p.set_visual(SphericalSFTGrating, paramsSFGrat(sf_waveform, sf_motion_type, 0, ang_per))
                self.add_phase(p)

                # Moving Phase CCW_2
                p = vxprotocol.Phase(sf_phase_dur + 0.125)  # produces -1 pi phase shift, by running 0.125 sec longer (at 4Hz TF, 1/2 cycle = 0.125s)
                p.set_visual(SphericalSFTGrating, paramsSFGrat(sf_waveform, sf_motion_type, -ang_per * sf_temp_freq,
                                                               ang_per))
                self.add_phase(p)


        # 5 sec grey between characterization sections
        p = vxprotocol.Phase(5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.5, .5, .5])})
        self.add_phase(p)


        # PART 2: Direction Selectivity
        repeats = 3
        for i in range(repeats):
            for motion_type in DSmotion_type:
                for direction in [-1,1]:
                    # Static Phase
                    p = vxprotocol.Phase(ds_phase_dur)
                    p.set_visual(SphericalDSGrating, paramsDSGrat(ds_waveform, motion_type, ds_motion_axis,
                                                                  0, ds_ang_period))
                    self.add_phase(p)

                    # Moving Phase
                    p = vxprotocol.Phase(ds_phase_dur)
                    p.set_visual(SphericalDSGrating, paramsDSGrat(ds_waveform, motion_type, ds_motion_axis,
                                                                  direction * ds_ang_vel, ds_ang_period))
                    self.add_phase(p)


        # 5 sec grey between characterization sections
        p = vxprotocol.Phase(5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.5, .5, .5])})
        self.add_phase(p)


        # PART 3: On/Off
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


        # PART 4: Flashes
        repeats = 10
        for i in range(repeats):
            for flash_dir in flash_direction:
                p = vxprotocol.Phase(flash_phase_dur)
                p.set_visual(UniformFlashStep, paramsFlash(base_lum, flash_start, flash_dur, flash_dir * flash_amp))
                self.add_phase(p)


        # 5 sec grey between characterization sections
        p = vxprotocol.Phase(5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.5, .5, .5])})
        self.add_phase(p)


        # PART 5: Dot
        repeats = 4
        for i in range(repeats):
            for elev_dir, elev_start in dot_small_directions:
                p = vxprotocol.Phase(small_dot_phase_dur)
                p.set_visual(SingleDotRotatingSpiral, paramsDot(dot_motion_axis, dot_polarity, dot_start_angle,
                                                                elev_dir * dot_ang_vel, dot_dia_small,
                                                                elev_dir * dot_small_elev_vel, elev_start))
                self.add_phase(p)

            for elev_dir, elev_start in dot_big_directions:
                p = vxprotocol.Phase(big_dot_phase_dur)
                p.set_visual(SingleDotRotatingSpiral, paramsDot(dot_motion_axis, dot_polarity, dot_start_angle,
                                                                elev_dir * dot_ang_vel, dot_dia_big,
                                                                elev_dir * dot_big_elev_vel, elev_start))
                self.add_phase(p)

        # 5 sec grey between characterization sections
        p = vxprotocol.Phase(5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.5, .5, .5])})
        self.add_phase(p)
