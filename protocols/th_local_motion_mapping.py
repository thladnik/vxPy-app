"""
vxpy_app ./protocols/spherical_gratings.py
Copyright (C) 2020 Tim Hladnik

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
import scipy.io

import vxpy.core.protocol as vxprotocol
from vxpy.utils.sphere import IcosahedronSphere
from vxpy.utils.geometry import cart2sph1


from visuals.local_translation_grating import LocalTranslationGrating_RoundArea
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground


def create_params(period, velocity, center_az, center_el, field_diameter):
    return {
        LocalTranslationGrating_RoundArea.grating_angular_period: period,
        LocalTranslationGrating_RoundArea.grating_angular_velocity: velocity,
        LocalTranslationGrating_RoundArea.stimulus_patch_center_azimuth: center_az,
        LocalTranslationGrating_RoundArea.stimulus_patch_center_elevation: center_el,
        LocalTranslationGrating_RoundArea.stimulus_patch_diameter: field_diameter
    }


class ProtocolRE(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        velocity = -30

        sphere = IcosahedronSphere(2)
        verts = sphere.get_vertices()
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        selected = np.logical_and(az > 0.0, el <= 25)
        selected_az = az[selected]
        selected_el = el[selected]

        combos = []
        for coords in zip(selected_az, selected_el):
            combos.append((15, *coords))
            combos.append((40, *coords))

        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(2):
            for size, az, el in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, velocity, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class ProtocolLE(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        velocity = 30

        sphere = IcosahedronSphere(2)
        verts = sphere.get_vertices()
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        selected = np.logical_and(az < 0.0, el <= 25)
        selected_az = az[selected]
        selected_el = el[selected]

        combos = []
        for coords in zip(selected_az, selected_el):
            combos.append((15, *coords))
            combos.append((40, *coords))

        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(2):
            for size, az, el in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, velocity, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class ProtocolRE20221223(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        velocity = -30

        sphere = IcosahedronSphere(2)
        verts = sphere.get_vertices()
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters = [5, 35]
        selected = np.logical_and(az > -22.5, el <= (45 - min(diameters) / 2))
        selected_az = az[selected]
        selected_el = el[selected]

        combos = []
        for coords in zip(selected_az, selected_el):
            # Skip 0/0, as that is a singularity
            if coords[0] == 0.0 and coords[1] == 0.0:
                continue

            for dia in diameters:
                combos.append((dia, *coords))

        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(2):
            for size, az, el in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, velocity, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class ProtocolLE20221223(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        base_velocity = 30

        sphere = IcosahedronSphere(2)
        verts = sphere.get_vertices()
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters = [5, 35]
        selected = np.logical_and(az < 22.5, el <= (45 - min(diameters) / 2))
        selected_az = az[selected]
        selected_el = el[selected]

        combos = []
        for a, e in zip(selected_az, selected_el):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters:
                combos.append((dia, a, e, vsign * base_velocity))

        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(2):
            for size, az, el, vel in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, vel, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class RepulsiveSphere140Demo(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        base_velocity = 30
        repeat_num = 3

        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_140.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters = [25]
        selected = np.logical_and(np.logical_and(az > -50, az < 90), el < 45.0)
        selected_az = az[selected]
        selected_el = el[selected]
        combos = []
        for a, e in zip(selected_az, selected_el):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters:
                combos.append((dia, a, e, vsign * base_velocity))



        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        for i in range(repeat_num):
            for size, az, el, vel in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, vel, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class RepulsiveSphere140RightEye20230216(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        base_velocity = 30
        repeat_num = 3

        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_140.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters = [25]
        selected = np.logical_and(np.logical_or(az > -30, az < -(180-30)), el < 45.0)
        selected_az = az[selected]
        selected_el = el[selected]

        combos = []
        for a, e in zip(selected_az, selected_el):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters:
                combos.append((dia, a, e, vsign * base_velocity))

        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(repeat_num):
            for size, az, el, vel in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, vel, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class RepulsiveSphere140LeftEye20230216(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        base_velocity = 30
        repeat_num = 3

        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_140.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters = [25]
        selected = np.logical_and(np.logical_or(az < 30, az > (180-30)), el < 45.0)
        selected_az = az[selected]
        selected_el = el[selected]

        combos = []
        for a, e in zip(selected_az, selected_el):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters:
                combos.append((dia, a, e, vsign * base_velocity))

        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(repeat_num):
            for size, az, el, vel in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, vel, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class RepulsiveSphere140and40RightEye20230223(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        base_velocity = 30
        repeat_num = 3

        # Create list for all combos
        combos = []

        # Densely sampled stimuli
        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_140.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters_dense = [25]
        selected_dense = np.logical_and(np.logical_or(az > -40, az < -(180 - 15)), el < 45.0)
        selected_az_dense = az[selected_dense]
        selected_el_dense = el[selected_dense]

        for a, e in zip(selected_az_dense, selected_el_dense):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters_dense:
                combos.append((dia, a, e, vsign * base_velocity))

        # Sparsely sampled stimuli
        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_40.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters_sparse = [45]
        selected_sparse = np.logical_and(np.logical_or(az > -40, az < -(180 - 15)), el < 45.0)
        selected_az_sparse = az[selected_sparse]
        selected_el_sparse = el[selected_sparse]

        for a, e in zip(selected_az_sparse, selected_el_sparse):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters_sparse:
                combos.append((dia, a, e, vsign * base_velocity))

        # Shuffle all
        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        # Create protocol phases
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(repeat_num):
            for size, az, el, vel in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, vel, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class RepulsiveSphere140and40LeftEye20230223(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        base_velocity = 30
        repeat_num = 3

        # Create list for all combos
        combos = []

        # Densely sampled stimuli
        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_140.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters_dense = [25]
        selected_dense = np.logical_and(np.logical_or(az < 40, az > (180 - 15)), el < 45.0)
        selected_az_dense = az[selected_dense]
        selected_el_dense = el[selected_dense]

        for a, e in zip(selected_az_dense, selected_el_dense):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters_dense:
                combos.append((dia, a, e, vsign * base_velocity))

        # Sparsely sampled stimuli
        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_40.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters_sparse = [45]
        selected_sparse = np.logical_and(np.logical_or(az < 40, az > (180 - 15)), el < 45.0)
        selected_az_sparse = az[selected_sparse]
        selected_el_sparse = el[selected_sparse]

        for a, e in zip(selected_az_sparse, selected_el_sparse):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters_sparse:
                combos.append((dia, a, e, vsign * base_velocity))

        # Shuffle all
        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        # Create protocol phases
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(repeat_num):
            for size, az, el, vel in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, vel, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class ProtocolRE_DEMO(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        velocity = -30

        sphere = IcosahedronSphere(2)
        verts = sphere.get_vertices()
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters = [5, 35]
        selected = np.logical_and(az > -22.5, el <= (45 - min(diameters) / 2))
        selected = np.logical_and(az < 90.0, selected)
        selected_az = az[selected]
        selected_el = el[selected]

        combos = []
        for coords in zip(selected_az, selected_el):
            # Skip 0/0, as that is a singularity
            if coords[0] == 0.0 and coords[1] == 0.0:
                continue

            for dia in diameters:
                combos.append((dia, *coords))

        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        p = vxprotocol.Phase(5)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(2):
            for size, az, el in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, velocity, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(5)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


if __name__ == '__main__':
    pass