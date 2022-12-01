import numpy as np

import vxpy.core.protocol as vxprotocol
from vxpy.visuals import pause

from visuals.single_moving_dot import SingleDotRotatingAroundAxis
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground
from visuals.spherical_grating import SphericalBlackWhiteGrating


class Protocol01(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        # Fix seed
        np.random.seed(1)

        # Black background at beginning
        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0.0,) * 3})
        self.add_phase(p)

        # Add global moving gratings
        grating_velocity = 60
        grating_period = 60
        moving_grating_duration = 4 * grating_period / grating_velocity
        motion_combinations = [('vertical', 'rotation'), ('vertical', 'translation'),
                               ('forward', 'rotation'), ('forward', 'translation')]
        for i in range(3):
            for motion_axis, motion_type in np.random.permutation(motion_combinations):
                # Randomize grating direction order
                grating_direction = 1 if np.random.randint(2) > 0 else -1

                p = vxprotocol.Phase(duration=2)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_axis: motion_axis,
                              SphericalBlackWhiteGrating.motion_type: motion_type,
                              SphericalBlackWhiteGrating.angular_velocity: 0,
                              SphericalBlackWhiteGrating.angular_period: grating_period})
                self.add_phase(p)

                p = vxprotocol.Phase(duration=moving_grating_duration)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_axis: motion_axis,
                              SphericalBlackWhiteGrating.motion_type: motion_type,
                              SphericalBlackWhiteGrating.angular_velocity: grating_direction * grating_velocity,
                              SphericalBlackWhiteGrating.angular_period: grating_period})
                self.add_phase(p)

                p = vxprotocol.Phase(duration=2)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_axis: motion_axis,
                              SphericalBlackWhiteGrating.motion_type: motion_type,
                              SphericalBlackWhiteGrating.angular_velocity: 0,
                              SphericalBlackWhiteGrating.angular_period: grating_period})
                self.add_phase(p)

                p = vxprotocol.Phase(duration=moving_grating_duration)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_axis: motion_axis,
                              SphericalBlackWhiteGrating.motion_type: motion_type,
                              SphericalBlackWhiteGrating.angular_velocity: -grating_direction * grating_velocity,
                              SphericalBlackWhiteGrating.angular_period: grating_period})
                self.add_phase(p)

                p = vxprotocol.Phase(duration=2)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_axis: motion_axis,
                              SphericalBlackWhiteGrating.motion_type: motion_type,
                              SphericalBlackWhiteGrating.angular_velocity: 0,
                              SphericalBlackWhiteGrating.angular_period: grating_period})
                self.add_phase(p)

        # Global luminance steps
        for i in range(3):
            for luminance in [0.00, 0.25, 0.50, 0.75, 1.0, 0.75, 0.50, 0.25]:
                # Black background at beginning
                p = vxprotocol.Phase(duration=4)
                p.set_visual(SphereUniformBackground,
                             {SphereUniformBackground.u_color: (luminance,) * 3})
                self.add_phase(p)

        # Present dots
        dot_sizes = [(1, -1), (3, -1), (5, -1), (10, -1), (20, -1), (40, -1),
                     (1, 1), (3, 1), (5, 1), (10, 1), (20, 1), (40, 1)]
        dot_velocity = 60
        for i in range(3):
            for dot_size, dot_direction in np.random.permutation(dot_sizes):
                duration = 360 / dot_velocity
                p = vxprotocol.Phase(duration=duration)
                p.set_visual(SingleDotRotatingAroundAxis,
                             {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                              SingleDotRotatingAroundAxis.dot_polarity: 'dark-on-light',
                              SingleDotRotatingAroundAxis.dot_start_angle: -90,
                              SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                              SingleDotRotatingAroundAxis.dot_angular_velocity: 0,
                              SingleDotRotatingAroundAxis.dot_offset_angle: -15,
                              })
                self.add_phase(p)

                p = vxprotocol.Phase(duration=duration)
                p.set_visual(SingleDotRotatingAroundAxis,
                             {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                              SingleDotRotatingAroundAxis.dot_polarity: 'dark-on-light',
                              SingleDotRotatingAroundAxis.dot_start_angle: -90,
                              SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                              SingleDotRotatingAroundAxis.dot_angular_velocity: dot_direction * dot_velocity,
                              SingleDotRotatingAroundAxis.dot_offset_angle: -15,
                              })
                self.add_phase(p)

        # Add black background at end
        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0.0,) * 3})
        self.add_phase(p)


class ProtocolJustDots(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0.0,) * 3})
        self.add_phase(p)

        dot_sizes = [1, 3, 5, 10, 20, 40]
        dot_elevations = [0, -15, -30]
        dot_velocity = 60
        dot_polarity = 'light-on-dark'
        for i in range(3):
            for dot_elev in dot_elevations:
                for dot_size in np.random.permutation(dot_sizes):
                    duration = 180 / dot_velocity

                    p = vxprotocol.Phase(duration=duration)
                    p.set_visual(SingleDotRotatingAroundAxis,
                                 {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                                  SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                                  SingleDotRotatingAroundAxis.dot_start_angle: 0,
                                  SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                                  SingleDotRotatingAroundAxis.dot_angular_velocity: 0,
                                  SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                                  })
                    self.add_phase(p)

                    p = vxprotocol.Phase(duration=duration)
                    p.set_visual(SingleDotRotatingAroundAxis,
                                 {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                                  SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                                  SingleDotRotatingAroundAxis.dot_start_angle: 0,
                                  SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                                  SingleDotRotatingAroundAxis.dot_angular_velocity: dot_velocity,
                                  SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                                  })
                    self.add_phase(p)

                    p = vxprotocol.Phase(duration=duration)
                    p.set_visual(SingleDotRotatingAroundAxis,
                                 {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                                  SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                                  SingleDotRotatingAroundAxis.dot_start_angle: 180,
                                  SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                                  SingleDotRotatingAroundAxis.dot_angular_velocity: 0,
                                  SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                                  })
                    self.add_phase(p)

                    p = vxprotocol.Phase(duration=duration)
                    p.set_visual(SingleDotRotatingAroundAxis,
                                 {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                                  SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                                  SingleDotRotatingAroundAxis.dot_start_angle: 180,
                                  SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                                  SingleDotRotatingAroundAxis.dot_angular_velocity: -dot_velocity,
                                  SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                                  })
                    self.add_phase(p)

        # Add black background at end
        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0.0,) * 3})
        self.add_phase(p)


class ProtocolJustDots01(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0.0,) * 3})
        self.add_phase(p)

        dot_sizes = [2, 3, 5, 10, 20, 30, 40]
        dot_elev = 0.0
        dot_velocity = 40
        dot_polarity = 'light-on-dark'
        for i in range(5):
            for dot_size in np.random.permutation(dot_sizes):
                duration = 180 / dot_velocity

                p = vxprotocol.Phase(duration=duration)
                p.set_visual(SingleDotRotatingAroundAxis,
                             {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                              SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                              SingleDotRotatingAroundAxis.dot_start_angle: 0,
                              SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                              SingleDotRotatingAroundAxis.dot_angular_velocity: 0,
                              SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                              })
                self.add_phase(p)

                p = vxprotocol.Phase(duration=duration)
                p.set_visual(SingleDotRotatingAroundAxis,
                             {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                              SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                              SingleDotRotatingAroundAxis.dot_start_angle: 0,
                              SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                              SingleDotRotatingAroundAxis.dot_angular_velocity: dot_velocity,
                              SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                              })
                self.add_phase(p)

                p = vxprotocol.Phase(duration=duration)
                p.set_visual(SingleDotRotatingAroundAxis,
                             {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                              SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                              SingleDotRotatingAroundAxis.dot_start_angle: 180,
                              SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                              SingleDotRotatingAroundAxis.dot_angular_velocity: 0,
                              SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                              })
                self.add_phase(p)

                p = vxprotocol.Phase(duration=duration)
                p.set_visual(SingleDotRotatingAroundAxis,
                             {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                              SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                              SingleDotRotatingAroundAxis.dot_start_angle: 180,
                              SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                              SingleDotRotatingAroundAxis.dot_angular_velocity: -dot_velocity,
                              SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                              })
                self.add_phase(p)

        # Add black background at end
        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0.0,) * 3})
        self.add_phase(p)


class ProtocolJustDots360deg(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0.0,) * 3})
        self.add_phase(p)

        dot_sizes = [2, 3, 5, 10, 20, 30, 40]
        dot_elev = 0.0
        dot_velocity = 40
        dot_polarity = 'light-on-dark'
        for i in range(5):
            for dot_size in np.random.permutation(dot_sizes):
                duration = 360 / dot_velocity

                p = vxprotocol.Phase(duration=4)
                p.set_visual(SingleDotRotatingAroundAxis,
                             {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                              SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                              SingleDotRotatingAroundAxis.dot_start_angle: -90,
                              SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                              SingleDotRotatingAroundAxis.dot_angular_velocity: 0,
                              SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                              })
                self.add_phase(p)

                p = vxprotocol.Phase(duration=duration)
                p.set_visual(SingleDotRotatingAroundAxis,
                             {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                              SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                              SingleDotRotatingAroundAxis.dot_start_angle: -90,
                              SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                              SingleDotRotatingAroundAxis.dot_angular_velocity: dot_velocity,
                              SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                              })
                self.add_phase(p)

                p = vxprotocol.Phase(duration=2)
                p.set_visual(SingleDotRotatingAroundAxis,
                             {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                              SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                              SingleDotRotatingAroundAxis.dot_start_angle: -90,
                              SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                              SingleDotRotatingAroundAxis.dot_angular_velocity: 0,
                              SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                              })
                self.add_phase(p)

                p = vxprotocol.Phase(duration=duration)
                p.set_visual(SingleDotRotatingAroundAxis,
                             {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                              SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                              SingleDotRotatingAroundAxis.dot_start_angle: -90,
                              SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                              SingleDotRotatingAroundAxis.dot_angular_velocity: -dot_velocity,
                              SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                              })
                self.add_phase(p)

        # Add black background at end
        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0.0,) * 3})
        self.add_phase(p)


class ProtocolLastResort(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0.0,) * 3})
        self.add_phase(p)

        dot_sizes = [2, 3, 5, 10, 20, 30, 40]
        dot_elev = 0.0
        dot_velocity = 40
        dot_polarity = 'light-on-dark'
        duration = 360 / dot_velocity
        for dot_size in dot_sizes:

            for direction in [-1, 1]:
                # Pause
                p = vxprotocol.Phase(duration=10)
                p.set_visual(SingleDotRotatingAroundAxis,
                             {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                              SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                              SingleDotRotatingAroundAxis.dot_start_angle: -90,
                              SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                              SingleDotRotatingAroundAxis.dot_angular_velocity: 0,
                              SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                              })
                self.add_phase(p)

                for i in range(3):
                    p = vxprotocol.Phase(duration=duration)
                    p.set_visual(SingleDotRotatingAroundAxis,
                                 {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                                  SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                                  SingleDotRotatingAroundAxis.dot_start_angle: -90,
                                  SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                                  SingleDotRotatingAroundAxis.dot_angular_velocity: direction * dot_velocity,
                                  SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                                  })
                    self.add_phase(p)

        # Add black background at end
        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0.0,) * 3})
        self.add_phase(p)


class ProtocolLastResortAlternatingDirections(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0.0,) * 3})
        self.add_phase(p)

        dot_sizes = [2, 3, 5, 10, 20, 30, 40]
        dot_elev = 0.0
        dot_velocity = 40
        dot_polarity = 'light-on-dark'
        duration = 360 / dot_velocity
        for dot_size in dot_sizes:

            # Pause
            p = vxprotocol.Phase(duration=10)
            p.set_visual(SingleDotRotatingAroundAxis,
                         {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                          SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                          SingleDotRotatingAroundAxis.dot_start_angle: -90,
                          SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                          SingleDotRotatingAroundAxis.dot_angular_velocity: 0,
                          SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                          })
            self.add_phase(p)

            for i in range(3):

                for direction in [-1, 1]:
                    p = vxprotocol.Phase(duration=duration)
                    p.set_visual(SingleDotRotatingAroundAxis,
                                 {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                                  SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                                  SingleDotRotatingAroundAxis.dot_start_angle: -90,
                                  SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                                  SingleDotRotatingAroundAxis.dot_angular_velocity: direction * dot_velocity,
                                  SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                                  })
                    self.add_phase(p)

        # Add black background at end
        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0.0,) * 3})
        self.add_phase(p)


class ProtocolLastResortAlternatingDirectionsRandomSizes(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        np.random.seed(1)

        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0.0,) * 3})
        self.add_phase(p)

        dot_sizes = [2, 3, 5, 10, 20, 30, 40]
        dot_elev = 0.0
        dot_velocity = 40
        dot_polarity = 'light-on-dark'
        duration = 360 / dot_velocity
        for dot_size in np.random.permutation(dot_sizes):

            # Pause
            p = vxprotocol.Phase(duration=10)
            p.set_visual(SingleDotRotatingAroundAxis,
                         {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                          SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                          SingleDotRotatingAroundAxis.dot_start_angle: -90,
                          SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                          SingleDotRotatingAroundAxis.dot_angular_velocity: 0,
                          SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                          })
            self.add_phase(p)

            for i in range(3):

                for direction in [-1, 1]:
                    p = vxprotocol.Phase(duration=duration)
                    p.set_visual(SingleDotRotatingAroundAxis,
                                 {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                                  SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                                  SingleDotRotatingAroundAxis.dot_start_angle: -90,
                                  SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                                  SingleDotRotatingAroundAxis.dot_angular_velocity: direction * dot_velocity,
                                  SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                                  })
                    self.add_phase(p)

        # Add black background at end
        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0.0,) * 3})
        self.add_phase(p)


class ProtocolLastResortAlternatingDirectionsFinal(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        np.random.seed(1)

        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0.0,) * 3})
        self.add_phase(p)

        dot_sizes = [2, 3, 5, 10, 20, 30]
        dot_elev = 0.0
        dot_velocity = 40
        dot_polarity = 'light-on-dark'
        duration = 360 / dot_velocity
        for dot_size in np.random.permutation(dot_sizes):

            for i in range(3):

                for direction in [-1, 1]:

                    # Pause
                    p = vxprotocol.Phase(duration=5)
                    p.set_visual(SingleDotRotatingAroundAxis,
                                 {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                                  SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                                  SingleDotRotatingAroundAxis.dot_start_angle: -90,
                                  SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                                  SingleDotRotatingAroundAxis.dot_angular_velocity: 0,
                                  SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                                  })
                    self.add_phase(p)

                    p = vxprotocol.Phase(duration=duration)
                    p.set_visual(SingleDotRotatingAroundAxis,
                                 {SingleDotRotatingAroundAxis.motion_axis: 'vertical',
                                  SingleDotRotatingAroundAxis.dot_polarity: dot_polarity,
                                  SingleDotRotatingAroundAxis.dot_start_angle: -90,
                                  SingleDotRotatingAroundAxis.dot_angular_diameter: dot_size,
                                  SingleDotRotatingAroundAxis.dot_angular_velocity: direction * dot_velocity,
                                  SingleDotRotatingAroundAxis.dot_offset_angle: dot_elev,
                                  })
                    self.add_phase(p)

        # Add black background at end
        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0.0,) * 3})
        self.add_phase(p)
