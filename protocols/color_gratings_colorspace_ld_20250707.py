

import random
import itertools
from itertools import product

import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.linalg import solve
import trimesh
import tqdm
import matplotlib
from scipy.interpolate import interp1d

from itertools import product

import vxpy.core.protocol as vxprotocol

from visuals.spherical_grating import SphericalColorContrastGrating
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground



# Load the normalized clipped irradiance data
normalized_clipped_df = pd.read_csv("data/normalized_irradiance_red_and_clipped_475_420.csv")

# Define a function to convert a set of relative BGR intensities to VxPy inputs
def bgr_to_vxpy_input(bgr: list[float],set_green_to_zero = False) -> list[float]:
    """
    Convert relative LED intensities [Blue, Green, Red] into VxPy input levels (0 to 1)
    using normalized irradiance curves: Red is original, Blue and Green are clipped to Red's max.
    """
    wavelengths = ["420nm", "475nm", "590nm"]
    vxpy_input_levels = np.linspace(0.0, 1.0, len(normalized_clipped_df))
    vxpy_inputs = []

    for intensity, led in zip(bgr, wavelengths):
        led_curve = normalized_clipped_df[led]
        interp_func = interp1d(led_curve, vxpy_input_levels, fill_value="extrapolate", bounds_error=False)
        vxpy_input = float(interp_func(intensity))
        vxpy_inputs.append(vxpy_input)

    if set_green_to_zero == True and bgr[1] == 0:
        vxpy_inputs[1] = 0.0

    return vxpy_inputs


class Protocol01(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        angular_period_degrees = 16.7
        angular_velocity_degrees = 30

        c1 = [0.0, 0.1, 0.18, 0.32, 0.56, 1.0]
        c2 = [0.0, 0.1, 0.18, 0.32, 0.56, 1.0]

        combinations = list(product(c1, c2))

        d1 = [1 if i % 2 == 0 else -1 for i in range(len(combinations))]
        d2 = [1 if i % 2 != 0 else -1 for i in range(len(combinations))]

        # rg
        random.shuffle(combinations)
        combinations1 = list(product(combinations))
        params11 = [(c[0], 0., 0., c[1], 0., 0., d1[i]) for i, c in enumerate(combinations)]
        params12 = [(c[0], 0., 0., c[1], 0., 0., d2[i]) for i, c in enumerate(combinations)]
        params1 = params11 + params12

        # gb
        random.shuffle(combinations)
        params21 = [(0., 0., c[0], 0., 0., c[1], d1[i]) for i, c in enumerate(combinations)]
        params22 = [(0., 0., c[0], 0., 0., c[1], d1[i]) for i, c in enumerate(combinations)]
        params2 = params21 + params22

        # br
        random.shuffle(combinations)
        params31 = [(c[0], 0., 0., 0., 0., c[1], d1[i]) for i, c in enumerate(combinations)]
        params32 = [(c[0], 0., 0., 0., 0., c[1], d1[i])  for i, c in enumerate(combinations)]
        params3 = params31 + params32

        # final param combination list
        params = params1 + params2 + params3

        random.shuffle(params)

        # initial pause phase
        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        led_calculator = LED_calculator()
        led_calculator.init()
        led_calculator.set_led_wavelengths_to_use([420, 475, 590])

        # reps
        for i in range(2):

            # current param combination
            for current_params in params:

                current_c_red1, current_c_red2, current_c_green1, current_c_green2, current_c_blue1, current_c_blue2, current_d = list(current_params)

                #red green plane
                if current_c_red1 > 0.0 and current_c_green2 > 0.0:
                    current_contrasts_1 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_gr(cone_iso_contrasts=(current_c_blue1, current_c_green1, current_c_red1))
                    current_contrasts_2 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_gr(cone_iso_contrasts=(current_c_blue2, current_c_green2, current_c_red2))


                #red blue plane
                if current_c_red1 > 0.0 and current_c_blue2 > 0.0:
                    current_contrasts_1 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_br(cone_iso_contrasts=(current_c_blue1, current_c_green1, current_c_red1))
                    current_contrasts_2 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_br(cone_iso_contrasts=(current_c_blue2, current_c_green2, current_c_red2))

                #blue green plane
                if current_c_green1 > 0.0 and current_c_blue2 > 0.0:
                    current_contrasts_1 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_bg(cone_iso_contrasts=(current_c_blue1, current_c_green1, current_c_red1))
                    current_contrasts_2 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_bg(cone_iso_contrasts=(current_c_blue2, current_c_green2, current_c_red2))

                #only red contrast
                if current_c_red1 > 0.0 and current_c_green2 == 0.0 and current_c_blue2 == 0.0:
                    gr_plane = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_gr(cone_iso_contrasts=(current_c_blue1, current_c_green1, current_c_red1))
                    br_plane = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_br(cone_iso_contrasts=(current_c_blue1, current_c_green1, current_c_red1))
                    if gr_plane[2] > br_plane[2]:
                        current_contrasts_1 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_gr(
                            cone_iso_contrasts=(current_c_blue1, current_c_green1, current_c_red1))
                        current_contrasts_2 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_gr(
                            cone_iso_contrasts=(current_c_blue2, current_c_green2, current_c_red2))
                    else:
                        current_contrasts_1 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_br(
                            cone_iso_contrasts=(current_c_blue1, current_c_green1, current_c_red1))
                        current_contrasts_2 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_br(
                            cone_iso_contrasts=(current_c_blue2, current_c_green2, current_c_red2))

                        # only green contrast gb plane
                if current_c_green1 > 0.0 and current_c_blue2 == 0.0:
                    # print('g', current_params)
                    current_contrasts_1 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_bg(
                        cone_iso_contrasts=(current_c_blue1, current_c_green1, current_c_red1))
                    current_contrasts_2 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_bg(
                        cone_iso_contrasts=(current_c_blue2, current_c_green2, current_c_red2))
                    # only green contrast gr plane
                if current_c_green2 > 0.0 and current_c_red1 == 0:
                    current_contrasts_1 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_gr(
                        cone_iso_contrasts=(current_c_blue1, current_c_green1, current_c_red1))
                    current_contrasts_2 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_gr(
                        cone_iso_contrasts=(current_c_blue2, current_c_green2, current_c_red2))

                    # only blue contrast
                if current_c_blue2 > 0.0 and current_c_green1 == 0.0 and current_c_red1 == 0.0:
                    bg_plane = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_bg(
                        cone_iso_contrasts=(current_c_blue1, current_c_green1, current_c_red1))
                    br_plane = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_br(
                        cone_iso_contrasts=(current_c_blue1, current_c_green1, current_c_red1))
                    if bg_plane[2] > br_plane[2]:
                        # print('b', current_params)
                        current_contrasts_1 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_bg(
                            cone_iso_contrasts=(current_c_blue1, current_c_green1, current_c_red1))
                        current_contrasts_2 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_bg(
                            cone_iso_contrasts=(current_c_blue2, current_c_green2, current_c_red2))
                    else:
                        current_contrasts_1 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_br(
                            cone_iso_contrasts=(current_c_blue1, current_c_green1, current_c_red1))
                        current_contrasts_2 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_br(
                            cone_iso_contrasts=(current_c_blue2, current_c_green2, current_c_red2))

                        # only zero contrast
                if current_c_red1 == 0.0 and current_c_green1 == 0 and current_c_green2 == 0 and current_c_blue2 == 0:
                    current_contrasts_1 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_gr(
                        cone_iso_contrasts=(current_c_red1, current_c_green1, current_c_blue1))
                    current_contrasts_2 = led_calculator.calc_and_get_led_intensities_for_iso_cone_contrasts_gr(
                        cone_iso_contrasts=(current_c_red2, current_c_green2, current_c_blue2))


                for i, contrast in enumerate(current_contrasts_1):
                    if contrast < 0.0:
                        current_contrasts_1[i] = 0.000000001
                for i, contrast in enumerate(current_contrasts_2):
                    if contrast < 0.0:
                        current_contrasts_2[i] = 0.000000001

                #clip contrasts to fit LED intensities
                clipped_contrasts_1 = bgr_to_vxpy_input(current_contrasts_1)
                clipped_contrasts_2 = bgr_to_vxpy_input(current_contrasts_2)

                #scale contrasts to literature values of cone sensitivities
                scaled_contrasts_1 = [clipped_contrasts_1[0]*0.41, clipped_contrasts_1[1]*0.18, clipped_contrasts_1[2]*1]
                scaled_contrasts_2 = [clipped_contrasts_2[0]*0.41, clipped_contrasts_2[1]*0.18, clipped_contrasts_2[2]*1]


                # static
                phase = vxprotocol.Phase(duration=6)
                phase.set_visual(SphericalColorContrastGrating,
                                     {SphericalColorContrastGrating.angular_period: angular_period_degrees,
                                      SphericalColorContrastGrating.angular_velocity: 0.,
                                      SphericalColorContrastGrating.waveform: 'rectangular',
                                      SphericalColorContrastGrating.motion_type: 'rotation',
                                      SphericalColorContrastGrating.motion_axis: 'vertical',
                                      SphericalColorContrastGrating.red01: scaled_contrasts_1[2],
                                      SphericalColorContrastGrating.green01: scaled_contrasts_1[1],
                                      SphericalColorContrastGrating.blue01: scaled_contrasts_1[0],
                                      SphericalColorContrastGrating.red02: scaled_contrasts_2[2],
                                      SphericalColorContrastGrating.green02: scaled_contrasts_2[1],
                                      SphericalColorContrastGrating.blue02: scaled_contrasts_2[0],
                                      })
                self.add_phase(phase)

                # dynamic
                phase = vxprotocol.Phase(duration=4)
                phase.set_visual(SphericalColorContrastGrating,
                                     {SphericalColorContrastGrating.angular_period: angular_period_degrees,
                                      SphericalColorContrastGrating.angular_velocity: current_d * angular_velocity_degrees,
                                      SphericalColorContrastGrating.waveform: 'rectangular',
                                      SphericalColorContrastGrating.motion_type: 'rotation',
                                      SphericalColorContrastGrating.motion_axis: 'vertical',
                                      SphericalColorContrastGrating.red01: scaled_contrasts_1[2],
                                      SphericalColorContrastGrating.green01: scaled_contrasts_1[1],
                                      SphericalColorContrastGrating.blue01: scaled_contrasts_1[0],
                                      SphericalColorContrastGrating.red02: scaled_contrasts_2[2],
                                      SphericalColorContrastGrating.green02: scaled_contrasts_2[1],
                                      SphericalColorContrastGrating.blue02: scaled_contrasts_2[0],
                                      })
                self.add_phase(phase)

        # final pause phase
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class LED_calculator():

    _wavelengths_to_use = [420, 475, 590]
    _cone_iso_contrasts = [0.0, 0.0, 0.0]
    _led_intensities_for_iso_cone_contrasts = None
    _wavelengths = np.arange(200, 800)
    _normlize_wavelength = plt.Normalize(360, 650)
    _wavelength_cmap = plt.get_cmap('nipy_spectral')
    _cones_to_use = ['s', 'm', 'l']
    _cone_activations = None
    _activation_spectra = None
    _cone_plot_colors = ['blue', 'green', 'red']
    _chrolis_spectra = None
    _led_spectra = None
    _cone_spectra = None
    _cube_corner_coords = None
    _pc = None

    wavelength_cmap = plt.get_cmap('nipy_spectral')
    normlize_wavelength = plt.Normalize(360, 650)

    wavelengths = np.arange(200, 800)

    def set_led_wavelengths_to_use(self, wavelengths_to_use: list):
        self._wavelengths_to_use = wavelengths_to_use

        # choose LEDs to work with
        self._led_spectra = np.array([self._chrolis_spectra[i] for i in wavelengths_to_use])

        self._calc_relative_cone_activations()
        self._calc_color_space()
        self._calc_color_plane_bg()
        self._calc_color_plane_br()
        self._calc_color_plane_gr()

    def set_cone_iso_contrasts(self, cone_iso_contrasts: list):
        self._cone_iso_contrasts = cone_iso_contrasts

    def calc_led_intensities_for_iso_cone_contrasts(self, cone_iso_contrasts: list):
        self._cone_iso_contrasts = cone_iso_contrasts
        self._calc_led_intensities_for_iso_cone_contrasts()

    def calc_led_intensities_for_iso_cone_contrasts_bg(self, cone_iso_contrasts: list):
        self._cone_iso_contrasts = cone_iso_contrasts
        self._calc_led_intensities_for_iso_cone_contrasts_bg()

    def calc_led_intensities_for_iso_cone_contrasts_br(self, cone_iso_contrasts: list):
        self._cone_iso_contrasts = cone_iso_contrasts
        self._calc_led_intensities_for_iso_cone_contrasts_br()

    def calc_led_intensities_for_iso_cone_contrasts_gr(self, cone_iso_contrasts: list):
        self._cone_iso_contrasts = cone_iso_contrasts
        self._calc_led_intensities_for_iso_cone_contrasts_gr()

    def calc_and_get_led_intensities_for_iso_cone_contrasts(self, cone_iso_contrasts: list):
        self._cone_iso_contrasts = cone_iso_contrasts
        self._calc_led_intensities_for_iso_cone_contrasts()
        return self._led_intensities_for_iso_cone_contrasts

    def calc_and_get_led_intensities_for_iso_cone_contrasts_bg(self, cone_iso_contrasts: list):
        self._cone_iso_contrasts = cone_iso_contrasts
        self._calc_led_intensities_for_iso_cone_contrasts_bg()
        return self._led_intensities_for_iso_cone_contrasts

    def calc_and_get_led_intensities_for_iso_cone_contrasts_gr(self, cone_iso_contrasts: list):
        self._cone_iso_contrasts = cone_iso_contrasts
        self._calc_led_intensities_for_iso_cone_contrasts_gr()
        return self._led_intensities_for_iso_cone_contrasts

    def calc_and_get_led_intensities_for_iso_cone_contrasts_br(self, cone_iso_contrasts: list):
        self._cone_iso_contrasts = cone_iso_contrasts
        self._calc_led_intensities_for_iso_cone_contrasts_br()
        return self._led_intensities_for_iso_cone_contrasts

    def get_led_intensities_for_iso_cone_contrasts(self):
        return self._led_intensities_for_iso_cone_contrasts

    def init(self):

        # load chrolis stimulation spectra
        wavelengths = np.arange(200, 800)
        chrolis_stimulation_spectra = pd.read_csv('./data/chrolis_spectra_raw.csv')
        for k in chrolis_stimulation_spectra.columns:
            chrolis_stimulation_spectra.loc[np.isnan(chrolis_stimulation_spectra[k]), k] = 0.
        self._chrolis_spectra = {
            wl: scipy.interpolate.interp1d(chrolis_stimulation_spectra[f'{wl}_wl'], chrolis_stimulation_spectra[f'{wl}'],
                                           fill_value=0., bounds_error=False)(wavelengths)
            for wl in [365, 420, 475, 565, 590, 625]
        }

        cone_absorption_spectra = pd.read_csv('data/cone_absorption_spectra.csv')
        zf_cone_spectra = {
            c: scipy.interpolate.interp1d(cone_absorption_spectra['wavelength'], cone_absorption_spectra[c],
                                          fill_value=0., bounds_error=False)(wavelengths)
            for c in ['uv', 's', 'm', 'l']
        }

        self._cone_spectra = np.array([zf_cone_spectra[s] for s in self._cones_to_use])

    def _calc_relative_cone_activations(self):

        # calc relative activations
        self._activation_spectra = np.zeros((len(self._cone_spectra), len(self._wavelengths_to_use), len(self._wavelengths)))
        for i, cone_spectrum in enumerate(self._cone_spectra):
            for j, led_spectrum in enumerate(self._led_spectra):
                activation_spectrum = cone_spectrum * led_spectrum

                self._activation_spectra[i, j] = activation_spectrum

        #self._cone_activations = self._activation_spectra.sum(axis=-1).T

        # norm it to area under curve == 100%
        self._cone_activations = self._activation_spectra.sum(axis=-1).T / self._cone_spectra.sum(axis=-1)

    def _calc_color_space(self):

        # calc color space
        verts = np.zeros((3, len(self._wavelengths_to_use) + 1))
        verts[:, 1:] = self._cone_activations.T

        hull = ConvexHull(verts.T)

        # create voxel matrix
        mesh = trimesh.Trimesh(vertices=verts.T, faces=hull.simplices)
        voxel_pitch = verts.max() / 30
        voxel_grid = mesh.voxelized(pitch=voxel_pitch, max_iter=100)
        voxel_matrix = voxel_grid.fill().matrix
        voxel_centers = voxel_grid.points

        # get indices of voxels inside mesh
        filled = np.argwhere(voxel_matrix)

        # calculate the largest cube that fits inside volume and whose sides are parallel with the S/M/L coordinate system
        largest_vol = 0
        largest_cube_idcs = np.zeros((8, 3), dtype=np.int32)
        for start_voxel_idcs in tqdm.tqdm(filled):
            for i in range(voxel_matrix.shape[0]):

                # TODO: try with z + 0 for plane instead of cube
                cube_idcs = np.repeat(start_voxel_idcs[None, :], 8, axis=0)
                cube_idcs[1, 0] += i
                cube_idcs[2, 1] += i
                cube_idcs[3, 2] += i
                cube_idcs[4, [0, 1]] += i
                cube_idcs[5, [1, 2]] += i
                cube_idcs[6, [0, 2]] += i
                cube_idcs[7, [0, 1, 2]] += i

                # check if voxel index sequence is in filled voxels
                match = True
                for idcs in cube_idcs:
                    match &= np.any(np.all(filled == idcs, axis=1))

                if not match:
                    break

                # if volume is larger than previous largest one, overwrite
                vol = i ** 3
                if vol > largest_vol:
                    largest_vol = vol
                    largest_cube_idcs = cube_idcs

        self._cube_corner_coords = largest_cube_idcs * voxel_pitch

        faces = np.array([
            [0, 1, 4, 2],  # Bottom face
            [3, 6, 7, 5],  # Top face
            [0, 1, 6, 3],  # Front face
            [2, 4, 7, 5],  # Back face
            [1, 4, 7, 6],  # Right face
            [0, 2, 5, 3],  # Left face
        ])

        face_vertices = [self._cube_corner_coords[face] for face in faces]
        self._pc = Poly3DCollection(face_vertices, facecolors=None, edgecolors='black', linestyles='--', linewidth=0.5, alpha=0.05)

    def _calc_color_plane_bg(self):

        # calc color space
        verts = np.zeros((3, len(self._wavelengths_to_use) + 1))
        verts[:, 1:] = self._cone_activations.T

        hull = ConvexHull(verts.T)

        # create voxel matrix
        mesh = trimesh.Trimesh(vertices=verts.T, faces=hull.simplices)
        voxel_pitch = verts.max() / 30
        voxel_grid = mesh.voxelized(pitch=voxel_pitch, max_iter=100)
        voxel_matrix = voxel_grid.fill().matrix
        voxel_centers = voxel_grid.points

        # get indices of voxels inside mesh
        filled = np.argwhere(voxel_matrix)

        # calculate the largest cube that fits inside volume and whose sides are parallel with the S/M/L coordinate system
        largest_vol = 0
        largest_cube_idcs = np.zeros((8, 3), dtype=np.int32)
        for start_voxel_idcs in (filled):
            for i in range(voxel_matrix.shape[0]):

                # TODO: try with z + 0 for plane instead of cube
                cube_idcs = np.repeat(start_voxel_idcs[None, :], 8, axis=0)
                #cube_idcs[1, 0] += i
                #cube_idcs[2, 1] += i
                #cube_idcs[3, 2] += i
                #cube_idcs[4, [0, 1]] += i
                #cube_idcs[5, [1, 2]] += i
                #cube_idcs[6, [0, 2]] += i
                #cube_idcs[7, [0, 1, 2]] += i

                cube_idcs[1, 0] += i
                cube_idcs[2, 1] += i
                cube_idcs[4, [0, 1]] += i
                cube_idcs[5, 1] += i
                cube_idcs[6, 0] += i
                cube_idcs[7, [0, 1]] += i

                # check if voxel index sequence is in filled voxels
                match = True
                for idcs in cube_idcs:
                    match &= np.any(np.all(filled == idcs, axis=1))

                if not match:
                    break

                # if volume is larger than previous largest one, overwrite
                vol = i ** 3
                if vol > largest_vol:
                    largest_vol = vol
                    largest_cube_idcs = cube_idcs

        self._cube_corner_coords_bg = largest_cube_idcs * voxel_pitch

        faces = np.array([
            [0, 1, 4, 2],  # Bottom face
            [3, 6, 7, 5],  # Top face
            [0, 1, 6, 3],  # Front face
            [2, 4, 7, 5],  # Back face
            [1, 4, 7, 6],  # Right face
            [0, 2, 5, 3],  # Left face
        ])

        face_vertices = [self._cube_corner_coords_bg[face] for face in faces]
        self._pc = Poly3DCollection(face_vertices, facecolors=None, edgecolors='black', linestyles='--', linewidth=0.5, alpha=0.05)

    def _calc_color_plane_gr(self):

        # calc color space
        verts = np.zeros((3, len(self._wavelengths_to_use) + 1))
        verts[:, 1:] = self._cone_activations.T

        hull = ConvexHull(verts.T)

        # create voxel matrix
        mesh = trimesh.Trimesh(vertices=verts.T, faces=hull.simplices)
        voxel_pitch = verts.max() / 30
        voxel_grid = mesh.voxelized(pitch=voxel_pitch, max_iter=100)
        voxel_matrix = voxel_grid.fill().matrix
        voxel_centers = voxel_grid.points

        # get indices of voxels inside mesh
        filled = np.argwhere(voxel_matrix)

        # calculate the largest cube that fits inside volume and whose sides are parallel with the S/M/L coordinate system
        largest_vol = 0
        largest_cube_idcs = np.zeros((8, 3), dtype=np.int32)
        for start_voxel_idcs in (filled):
            for i in range(voxel_matrix.shape[0]):

                # TODO: try with z + 0 for plane instead of cube
                cube_idcs = np.repeat(start_voxel_idcs[None, :], 8, axis=0)
                #cube_idcs[1, 0] += i
                #cube_idcs[2, 1] += i
                #cube_idcs[3, 2] += i
                #cube_idcs[4, [0, 1]] += i
                #cube_idcs[5, [1, 2]] += i
                #cube_idcs[6, [0, 2]] += i
                #cube_idcs[7, [0, 1, 2]] += i

                cube_idcs[2, 1] += i
                cube_idcs[3, 2] += i
                cube_idcs[4, 1] += i
                cube_idcs[5, [1, 2]] += i
                cube_idcs[6, 2] += i
                cube_idcs[7, [1, 2]] += i

                # check if voxel index sequence is in filled voxels
                match = True
                for idcs in cube_idcs:
                    match &= np.any(np.all(filled == idcs, axis=1))

                if not match:
                    break

                # if volume is larger than previous largest one, overwrite
                vol = i ** 3
                if vol > largest_vol:
                    largest_vol = vol
                    largest_cube_idcs = cube_idcs

        self._cube_corner_coords_gr = largest_cube_idcs * voxel_pitch

        faces = np.array([
            [0, 1, 4, 2],  # Bottom face
            [3, 6, 7, 5],  # Top face
            [0, 1, 6, 3],  # Front face
            [2, 4, 7, 5],  # Back face
            [1, 4, 7, 6],  # Right face
            [0, 2, 5, 3],  # Left face
        ])

        face_vertices = [self._cube_corner_coords_gr[face] for face in faces]
        self._pc = Poly3DCollection(face_vertices, facecolors=None, edgecolors='black', linestyles='--', linewidth=0.5, alpha=0.05)

    def _calc_color_plane_br(self):

        # calc color space
        verts = np.zeros((3, len(self._wavelengths_to_use) + 1))
        verts[:, 1:] = self._cone_activations.T

        hull = ConvexHull(verts.T)

        # create voxel matrix
        mesh = trimesh.Trimesh(vertices=verts.T, faces=hull.simplices)
        voxel_pitch = verts.max() / 30
        voxel_grid = mesh.voxelized(pitch=voxel_pitch, max_iter=100)
        voxel_matrix = voxel_grid.fill().matrix
        voxel_centers = voxel_grid.points

        # get indices of voxels inside mesh
        filled = np.argwhere(voxel_matrix)

        # calculate the largest cube that fits inside volume and whose sides are parallel with the S/M/L coordinate system
        largest_vol = 0
        largest_cube_idcs = np.zeros((8, 3), dtype=np.int32)
        for start_voxel_idcs in (filled):
            for i in range(voxel_matrix.shape[0]):

                # TODO: try with z + 0 for plane instead of cube
                cube_idcs = np.repeat(start_voxel_idcs[None, :], 8, axis=0)
                #cube_idcs[1, 0] += i
                #cube_idcs[2, 1] += i
                #cube_idcs[3, 2] += i
                #cube_idcs[4, [0, 1]] += i
                #cube_idcs[5, [1, 2]] += i
                #cube_idcs[6, [0, 2]] += i
                #cube_idcs[7, [0, 1, 2]] += i

                cube_idcs[1, 0] += i
                cube_idcs[3, 2] += i
                cube_idcs[4, 0] += i
                cube_idcs[5, 2] += i
                cube_idcs[6, [0, 2]] += i
                cube_idcs[7, [0, 2]] += i

                # check if voxel index sequence is in filled voxels
                match = True
                for idcs in cube_idcs:
                    match &= np.any(np.all(filled == idcs, axis=1))

                if not match:
                    break

                # if volume is larger than previous largest one, overwrite
                vol = i ** 3
                if vol > largest_vol:
                    largest_vol = vol
                    largest_cube_idcs = cube_idcs

        self._cube_corner_coords_br = largest_cube_idcs * voxel_pitch

        faces = np.array([
            [0, 1, 4, 2],  # Bottom face
            [3, 6, 7, 5],  # Top face
            [0, 1, 6, 3],  # Front face
            [2, 4, 7, 5],  # Back face
            [1, 4, 7, 6],  # Right face
            [0, 2, 5, 3],  # Left face
        ])

        face_vertices = [self._cube_corner_coords_br[face] for face in faces]
        self._pc = Poly3DCollection(face_vertices, facecolors=None, edgecolors='black', linestyles='--', linewidth=0.5, alpha=0.05)

    def _calc_led_intensities_for_iso_cone_contrasts(self):

        # Illustrate cone isolating stimulation axes in cubix subspace
        cone_isolating_origin = self._cube_corner_coords[0]

        # Mark target contrasts in each cone isolating axis with a dot
        cone_isolating_directions = self._cube_corner_coords[1:4] - cone_isolating_origin
        cone_iso_vectors = cone_isolating_origin + self._cone_iso_contrasts * cone_isolating_directions

        # !!
        # This coordinate transformation only works for 3 LEDs,
        # bc the original S/M/L coordinate system also only has 3 dimensions

        # Transform cube coordinates to non-euclidean coordinates
        stimulation_space_norms = self._cone_activations / np.linalg.norm(self._cone_activations, axis=1)

        wl_space_cube_corners = np.array([np.linalg.solve(stimulation_space_norms, v) for v in self._cube_corner_coords])
        wl_space_cube_origin = wl_space_cube_corners[0]

        # Calculate vectors in coordinate system defined by wavelength vectors in S/M/L coordinate system
        wavelength_vectors = np.array([np.linalg.solve(stimulation_space_norms, v) for v in cone_iso_vectors])

        wl_space_led_intensities = wavelength_vectors - wl_space_cube_origin
        wl_space_led_intensities = wl_space_led_intensities.sum(axis=1) + wl_space_cube_origin

        # Divide by lengths of original cone activation vectors for each stimulation color
        self._led_intensities_for_iso_cone_contrasts = wl_space_led_intensities / np.linalg.norm(self._cone_activations, axis=0)

    def _calc_led_intensities_for_iso_cone_contrasts_bg(self):

        # Illustrate cone isolating stimulation axes in cubix subspace
        cone_isolating_origin = self._cube_corner_coords_bg[0]

        # Mark target contrasts in each cone isolating axis with a dot
        cone_isolating_directions = self._cube_corner_coords_bg[1:4] - cone_isolating_origin
        cone_iso_vectors = cone_isolating_origin + self._cone_iso_contrasts * cone_isolating_directions

        # !!
        # This coordinate transformation only works for 3 LEDs,
        # bc the original S/M/L coordinate system also only has 3 dimensions

        # Transform cube coordinates to non-euclidean coordinates
        stimulation_space_norms = self._cone_activations / np.linalg.norm(self._cone_activations, axis=1)

        wl_space_cube_corners = np.array([np.linalg.solve(stimulation_space_norms, v) for v in self._cube_corner_coords_bg])
        wl_space_cube_origin = wl_space_cube_corners[0]

        # Calculate vectors in coordinate system defined by wavelength vectors in S/M/L coordinate system
        wavelength_vectors = np.array([np.linalg.solve(stimulation_space_norms, v) for v in cone_iso_vectors])

        wl_space_led_intensities = wavelength_vectors - wl_space_cube_origin
        wl_space_led_intensities = wl_space_led_intensities.sum(axis=1) + wl_space_cube_origin

        # Divide by lengths of original cone activation vectors for each stimulation color
        self._led_intensities_for_iso_cone_contrasts = wl_space_led_intensities / np.linalg.norm(self._cone_activations, axis=0)

    def _calc_led_intensities_for_iso_cone_contrasts_br(self):

        # Illustrate cone isolating stimulation axes in cubix subspace
        cone_isolating_origin = self._cube_corner_coords_br[0]

        # Mark target contrasts in each cone isolating axis with a dot
        cone_isolating_directions = self._cube_corner_coords_br[1:4] - cone_isolating_origin
        cone_iso_vectors = cone_isolating_origin + self._cone_iso_contrasts * cone_isolating_directions

        # !!
        # This coordinate transformation only works for 3 LEDs,
        # bc the original S/M/L coordinate system also only has 3 dimensions

        # Transform cube coordinates to non-euclidean coordinates
        stimulation_space_norms = self._cone_activations / np.linalg.norm(self._cone_activations, axis=1)

        wl_space_cube_corners = np.array([np.linalg.solve(stimulation_space_norms, v) for v in self._cube_corner_coords_br])
        wl_space_cube_origin = wl_space_cube_corners[0]

        # Calculate vectors in coordinate system defined by wavelength vectors in S/M/L coordinate system
        wavelength_vectors = np.array([np.linalg.solve(stimulation_space_norms, v) for v in cone_iso_vectors])

        wl_space_led_intensities = wavelength_vectors - wl_space_cube_origin
        wl_space_led_intensities = wl_space_led_intensities.sum(axis=1) + wl_space_cube_origin

        # Divide by lengths of original cone activation vectors for each stimulation color
        self._led_intensities_for_iso_cone_contrasts = wl_space_led_intensities / np.linalg.norm(self._cone_activations, axis=0)

    def _calc_led_intensities_for_iso_cone_contrasts_gr(self):

        # Illustrate cone isolating stimulation axes in cubix subspace
        cone_isolating_origin = self._cube_corner_coords_gr[0]

        # Mark target contrasts in each cone isolating axis with a dot
        cone_isolating_directions = self._cube_corner_coords_gr[1:4] - cone_isolating_origin
        cone_iso_vectors = cone_isolating_origin + self._cone_iso_contrasts * cone_isolating_directions

        # !!
        # This coordinate transformation only works for 3 LEDs,
        # bc the original S/M/L coordinate system also only has 3 dimensions

        # Transform cube coordinates to non-euclidean coordinates
        stimulation_space_norms = self._cone_activations / np.linalg.norm(self._cone_activations, axis=1)

        wl_space_cube_corners = np.array([np.linalg.solve(stimulation_space_norms, v) for v in self._cube_corner_coords_gr])
        wl_space_cube_origin = wl_space_cube_corners[0]

        # Calculate vectors in coordinate system defined by wavelength vectors in S/M/L coordinate system
        wavelength_vectors = np.array([np.linalg.solve(stimulation_space_norms, v) for v in cone_iso_vectors])

        wl_space_led_intensities = wavelength_vectors - wl_space_cube_origin
        wl_space_led_intensities = wl_space_led_intensities.sum(axis=1) + wl_space_cube_origin

        # Divide by lengths of original cone activation vectors for each stimulation color
        self._led_intensities_for_iso_cone_contrasts = wl_space_led_intensities / np.linalg.norm(self._cone_activations, axis=0)

    def plot_led_and_cone_spectra(self):

        plt.figure()
        plt.title('LED and cone absorption spectra')
        plt.xlabel('wavelength [nm]')
        plt.ylabel('relative spectra [%]')
        led_lines = plt.plot(self._wavelengths, self._led_spectra.T)
        for i, line in enumerate(led_lines):
            line.set_color(self._wavelength_cmap(self._normlize_wavelength(self._wavelengths_to_use[i])))
        plt.twinx()
        cone_lines = plt.plot(self._wavelengths, self._cone_spectra.T, linestyle='--')
        for i, line in enumerate(cone_lines):
            line.set_color(self._cone_plot_colors[i])

    def plot_relative_cone_activations(self):

        fig, ax = plt.subplots()
        plt.title('relative cone activation')
        legend_lines = []
        for i, cone_str in enumerate(self._cones_to_use):
            act_lines = plt.plot(self._wavelengths, self._activation_spectra[i].T,
                                 color=self._cone_plot_colors[i])
            legend_lines.append(act_lines[0])

        ax.legend(legend_lines, [f'{s.upper()}-cones' for s in self._cones_to_use])
        plt.xlabel('wavelength [nm]')
        plt.ylabel('relative cone activation[%]')

        fig, ax = plt.subplots()
        plt.title('Relative cone activations')
        x = np.arange(len(self._cones_to_use))
        width = 0.25
        multiplier = 0
        bottom = np.zeros(len(self._cones_to_use))

        for i, wavelength in enumerate(self._wavelengths_to_use):
            wl_color = self._wavelength_cmap(self._normlize_wavelength(self._wavelengths_to_use[i]))
            offset = width * multiplier
            rects = ax.bar(x, self._cone_activations[i, :], width, color=wl_color, label=f'{wavelength}nm')
            multiplier += 1
            bottom += self._cone_activations[i, :]

        ax.set_xticks(x + width, [s.upper() for s in self._cones_to_use])
        ax.legend()
        plt.xlabel('cones')
        plt.ylabel('relative cone activation [%]')

    def plot_color_space(self):

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot([0], [0], [0], marker='x', linestyle='none', color='k')

        for i, wl in enumerate(self._wavelengths_to_use):
            activation = self._cone_activations[i, :]
            ax.plot(*activation, marker='o', linestyle='none', color=self._wavelength_cmap(self._normlize_wavelength(wl)))

            vec = np.zeros((3, 2))
            vec[:, 0] = activation

            ax.plot(*vec, linestyle='--', linewidth=2, color=self._wavelength_cmap(self._normlize_wavelength(wl)),
                    label=('LED ' + str(wl) + ' nm'))

        verts = np.zeros((3, len(self._wavelengths_to_use) + 1))
        verts[:, 1:] = self._cone_activations.T

        hull = ConvexHull(verts.T)

        # each simplex is a the triangle of the convex hull
        for s in hull.simplices:
            # s = np.append(s, s[0])  # append the first coordinate at the end to create a closed polygon
            # ax.plot(verts[0, s], verts[1, s], verts[2, s], "k-", alpha=0.2)
            ax.plot_trisurf(*np.array([verts[:, i] for i in s]).T, color='black', alpha=0.1, linewidth=1, edgecolor='k',
                            linestyle='--')

        # plot S/M/L axis
        axis_length = np.max(self._cone_activations)
        ax.plot([0, axis_length], [0, 0], [0, 0], color=self._cone_plot_colors[0], linewidth=2)
        ax.plot([0, 0], [0, axis_length], [0, 0], color=self._cone_plot_colors[1], linewidth=2)
        ax.plot([0, 0], [0, 0], [0, axis_length], color=self._cone_plot_colors[2], linewidth=2)
        ax.text(axis_length / 2, 0, 0, 'S', zdir='x', color=self._cone_plot_colors[0], fontsize=10)
        ax.text(0, axis_length / 2, 0, 'M', zdir='y', color=self._cone_plot_colors[1], fontsize=10)
        ax.text(0, 0, axis_length / 2, 'L', zdir='z', color=self._cone_plot_colors[2], fontsize=10)

        plt.axis('off')
        plt.legend(loc='best')
        ax.grid(False)
        ax.set_aspect('equal')

        # plot cube
        ax.plot(*self._cube_corner_coords.T, marker='x', linestyle='none', color='r')
        pc = copy(self._pc)
        ax.add_collection3d(pc)
        cone_isolating_origin = self._cube_corner_coords[0]
        for i, v in enumerate(self._cube_corner_coords[1:4]):
            color = self._cone_plot_colors[i]
            ax.plot(*np.array([cone_isolating_origin, v]).T, color=color, linewidth=2, alpha=1.0)
        cone_isolating_directions = self._cube_corner_coords[1:4] - cone_isolating_origin
        cone_iso_vectors = cone_isolating_origin + self._cone_iso_contrasts * cone_isolating_directions
        for i, vec in enumerate(cone_iso_vectors):
            color = self._cone_plot_colors[i]
            ax.plot(*vec, color=color, marker='o')