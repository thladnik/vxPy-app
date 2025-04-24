"""
MappApp ./utils/sphere.py
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
from typing import Tuple

import numpy as np

from vxpy.utils import geometry


########
# !!! BE EXTREMELY CAREFUL WHEN EDITING THESE MODELS !!!
# Any changes will affect all visuals associated with the models!
#
########

class SimpleUVSphere:

    def __init__(self,
                 azimuth_levels: int = 60,
                 elevation_levels: int = 30):

        self.azimuth_levels = azimuth_levels
        self.elevation_levels = elevation_levels

        # Spherical coordinates in azimuth and elevation
        self.azimuth_space = np.linspace(-np.pi, np.pi, self.azimuth_levels, endpoint=True)
        self.elevation_space = np.linspace(-np.pi / 2, np.pi / 2, self.elevation_levels, endpoint=True)
        self.mesh_azimuths, self.mesh_elevations = np.meshgrid(self.azimuth_space, self.elevation_space)

        self.azimuths = np.ascontiguousarray(self.mesh_azimuths.flatten(), dtype=np.float32)
        self.elevations = np.ascontiguousarray(self.mesh_elevations.flatten(), dtype=np.float32)

        # 3D coordinates
        pos = geometry.sph2cart(self.azimuths, self.elevations, 1.)
        self.positions = np.ascontiguousarray(pos.T, dtype=np.float32)

        # Face indices
        faces = list()
        for i in np.arange(elevation_levels):
            for j in np.arange(azimuth_levels):
                faces.append([i * azimuth_levels + j, i * azimuth_levels + j + 1, (i + 1) * azimuth_levels + j + 1])
                faces.append([i * azimuth_levels + j, (i + 1) * azimuth_levels + j, (i + 1) * azimuth_levels + j + 1])
        self.faces = np.array(faces, dtype=np.uint32)
        self.indices = np.ascontiguousarray(self.faces.flatten())

    @staticmethod
    def get_uv_coordinates(positions: np.ndarray) -> np.ndarray:
        """Return 2d UV coordinates for given 3d spherical coordinate set of shape (N x 3)"""
        u = 0.5 + np.arctan2(positions[:, 0], positions[:, 1]) / np.pi / 2.0
        v = 0.5 + np.arcsin(-positions[:, 2]) / np.pi

        return np.stack([u, v]).T

    def get_subset(self, lower_azimuth: float, upper_azimuth: float,
                   lower_elevation: float, upper_elevation: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get a subset of positions and corresponding faces based on upper and lower azimuth/elevation bounds"""
        mask_selection = (lower_azimuth <= self.azimuths) & (self.azimuths <= upper_azimuth) \
                         & (lower_elevation <= self.elevations) & (self.elevations <= upper_elevation)
        selection_indices = np.where(mask_selection)[0]
        selection_set = set(selection_indices)
        positions = self.positions[mask_selection]
        faces = np.array([face for face in self.faces if len(set(face) & selection_set) == 3])

        # Fix face indices
        while faces.max() >= positions.shape[0]:
            faces -= faces.min()

        return selection_indices, positions, faces


class UVSphere:

    def __init__(self,
                 azim_lvls: int = 60,
                 elev_lvls: int = 30,
                 azimuth_range: float = 2 * np.pi,
                 upper_elev: float = np.pi / 4,
                 radius: float = 1.0):

        # Set parameters
        self.azim_lvls = azim_lvls
        self.elev_lvls = elev_lvls
        self.azimuth_range = azimuth_range
        self.upper_elev = upper_elev
        self.radius = radius

        # Calculate coordinates in azimuth and elevation
        az = np.linspace(0, self.azimuth_range, self.azim_lvls, endpoint=True)
        el = np.linspace(-np.pi / 2, self.upper_elev, self.elev_lvls, endpoint=True)
        self.azims, self.elevs = np.meshgrid(az, el)

        # Set vertex attributes
        self.a_azimuth = np.ascontiguousarray(self.azims.flatten(), dtype=np.float32)
        self.azimuth_degree = np.ascontiguousarray(self.azims.flatten() / np.pi / 2.0 * 360.0, dtype=np.float32)
        self.azimuth_degree2 = np.ascontiguousarray((self.azims.flatten() - np.pi) / np.pi / 2.0 * 360.0,
                                                    dtype=np.float32)
        self.azimuth_degree_zero_front_pos_cw = -np.ascontiguousarray(
            (self.azims.flatten() - np.pi) / np.pi / 2.0 * 360.0, dtype=np.float32)
        self.a_elevation = np.ascontiguousarray(self.elevs.flatten(), dtype=np.float32)
        self.elevation_degree = np.ascontiguousarray(self.elevs.flatten() / np.pi / 2.0 * 360.0, dtype=np.float32)
        self.a_position = geometry.sph2cart(self.a_azimuth, self.a_elevation, self.radius)
        self.a_position = np.ascontiguousarray(self.a_position.T, dtype=np.float32)

        # Set indices
        idcs = list()
        for i in np.arange(self.elev_lvls):
            for j in np.arange(self.azim_lvls):
                idcs.append([i * azim_lvls + j, i * azim_lvls + j + 1, (i + 1) * azim_lvls + j + 1])
                idcs.append([i * azim_lvls + j, (i + 1) * azim_lvls + j, (i + 1) * azim_lvls + j + 1])

        self.indices = np.ascontiguousarray(np.array(idcs).flatten(), dtype=np.uint32)

    def get_uv_coordinates(self):
        vecs = self.a_position
        u = 0.5 + np.arctan2(vecs[:, 0], vecs[:, 1]) / np.pi / 2.0
        v = 0.5 + np.arcsin(-vecs[:, 2]) / np.pi

        return np.stack([u, v]).T

    # def get_uv_coordinates(self):
    #     vecs = self.a_position
    #     u = 0.5 + np.arctan2(vecs[:,0], vecs[:,2]) / np.pi / 2.0
    #     v = 0.5 + np.arcsin(vecs[:,1]) / np.pi
    #
    #     return np.stack([u, v]).T


class ForeBackSphere:

    def __init__(self,
                 azim_lvls: int = 60,
                 elev_lvls: int = 30,
                 azimuth_range: float = 2 * np.pi,
                 upper_elev: float = np.pi / 4,
                 radius: float = 1.0,
                 subdivisionTimes: int = 1,
                 ):

        # Set parameters
        self.r = (1 + np.sqrt(5)) / 2
        self.init_vertices = np.array([
            [-1.0, self.r, 0.0],
            [1.0, self.r, 0.0],
            [-1.0, -self.r, 0.0],
            [1.0, -self.r, 0.0],
            [0.0, -1.0, self.r],
            [0.0, 1.0, self.r],
            [0.0, -1.0, -self.r],
            [0.0, 1.0, -self.r],
            [self.r, 0.0, -1.0],
            [self.r, 0.0, 1.0],
            [-self.r, 0.0, -1.0],
            [-self.r, 0.0, 1.0]
        ])
        self.init_faces = np.array([
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1]
        ])
        self.sdtimes = subdivisionTimes
        self.azim_lvls = azim_lvls
        self.elev_lvls = elev_lvls
        self.azimuth_range = azimuth_range
        self.upper_elev = upper_elev
        self.radius = radius

        # Calculate coordinates in azimuth and elevation
        az = np.linspace(0, self.azimuth_range, self.azim_lvls, endpoint=True)
        el = np.linspace(-np.pi / 2, self.upper_elev, self.elev_lvls, endpoint=True)
        self.azims, self.elevs = np.meshgrid(az, el)

        # Set vertex attributes
        self.a_azimuth = np.ascontiguousarray(self.azims.flatten(), dtype=np.float32)
        self.azimuth_degree = np.ascontiguousarray(self.azims.flatten() / np.pi / 2.0 * 360.0, dtype=np.float32)
        self.azimuth_degree2 = np.ascontiguousarray((self.azims.flatten() - np.pi) / np.pi / 2.0 * 360.0,
                                                    dtype=np.float32)
        self.azimuth_degree_zero_front_pos_cw = -np.ascontiguousarray(
            (self.azims.flatten() - np.pi) / np.pi / 2.0 * 360.0, dtype=np.float32)
        self.a_elevation = np.ascontiguousarray(self.elevs.flatten(), dtype=np.float32)
        self.elevation_degree = np.ascontiguousarray(self.elevs.flatten() / np.pi / 2.0 * 360.0, dtype=np.float32)
        self.a_position = geometry.sph2cart(self.a_azimuth, self.a_elevation, self.radius)
        self.a_position = np.ascontiguousarray(self.a_position.T, dtype=np.float32)

        # Set indices
        idcs = list()
        for i in np.arange(self.elev_lvls / 2):
            for j in np.arange(self.azim_lvls / 2):
                idcs.append([i * azim_lvls + j, i * azim_lvls + j + 1, (i + 1) * azim_lvls + j])
                idcs.append([i * azim_lvls + j + 1, (i + 1) * azim_lvls + j, (i + 1) * azim_lvls + j + 1])
        self.indices = np.ascontiguousarray(np.array(idcs).flatten(), dtype=np.uint32)

        [usV, usF] = geometry.subdivide_triangle(self.init_vertices, self.init_faces,
                                                 self.sdtimes)  # Compute the radius of all the vertices
        sphereR = geometry.vecNorm(usV[0, :])  # Compute the radius of all the vertices
        tileCen = np.mean(usV[usF, :], axis=1)  # Compute the center of each triangle tiles
        tileCen = geometry.vecNormalize(tileCen)

        # Create index buffer
        # Iout = np.arange(usF.size, dtype=np.uint32)
        # self.indices = Iout

        # Create vertex buffer
        # The orientation of each triangle tile is defined as the direction perpendicular to the first edge of the triangle;
        # Here each orientation vector is represented by a complex number for the convenience of later computation
        tile_orientation = geometry.vecNormalize(
            np.cross(tileCen, usV[usF[:, 1], :] - usV[usF[:, 0], :])) + 1.j * geometry.vecNormalize(
            usV[usF[:, 1], :] - usV[usF[:, 0], :])
        tileDist = geometry.sphAngle(tileCen, sphereR)  # Spherical distance for each tile pair
        usF = np.int64(usF.flatten())

        # Triangles must not share edges/vertices while doing texture mapping, this line duplicate the shared vertices for each triangle
        # self.a_position = geometry.vecNormalize(usV[usF, :])
        # self.a_texcoord = geometry.cen2tri(np.random.rand(int(Iout.size / 3)), np.random.rand(int(Iout.size / 3)),
        # .1).reshape([Iout.size, 2])

        self.tile_orientation = tile_orientation
        self.tile_center = tileCen
        self.intertile_distance = tileDist

    def get_uv_coordinates(self):
        vecs = self.a_position
        u = 0.5 + np.arctan2(vecs[:, 0], vecs[:, 1]) / np.pi / 2.0
        v = 0.5 + np.arcsin(-vecs[:, 2]) / np.pi

        return np.stack([u, v]).T


class IcoSphere_UVsphere:

    def __init__(self,
                 azim_lvls: int = 32,
                 elev_lvls: int = 30,
                 azimuth_range: float = 2 * np.pi,
                 upper_elev: float = np.pi / 4,
                 radius: float = 1.0,
                 subdivisionTimes: int = 1):
        ### Create sphere
        self.r = (1 + np.sqrt(5)) / 2
        self.init_vertices = np.array([
            [-1.0, self.r, 0.0],
            [1.0, self.r, 0.0],
            [-1.0, -self.r, 0.0],
            [1.0, -self.r, 0.0],
            [0.0, -1.0, self.r],
            [0.0, 1.0, self.r],
            [0.0, -1.0, -self.r],
            [0.0, 1.0, -self.r],
            [self.r, 0.0, -1.0],
            [self.r, 0.0, 1.0],
            [-self.r, 0.0, -1.0],
            [-self.r, 0.0, 1.0]
        ])
        self.init_faces = np.array([
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1]
        ])
        self.sdtimes = subdivisionTimes
        self.azim_lvls = azim_lvls
        self.elev_lvls = elev_lvls
        self.azimuth_range = azimuth_range
        self.upper_elev = upper_elev
        self.radius = radius
        self.coordinates = [self.vertex(*v) for v in self.init_vertices]

        [usV, usF] = geometry.subdivide_triangle(self.init_vertices, self.init_faces,
                                                 self.sdtimes)  # Compute the radius of all the vertices
        sphereR = geometry.vecNorm(usV[0, :])  # Compute the radius of all the vertices
        tileCen = np.mean(usV[usF, :], axis=1)  # Compute the center of each triangle tiles
        tileCen = geometry.vecNormalize(tileCen)

        # Create index buffer
        Iout = np.arange(usF.size, dtype=np.uint32)
        self.indices = Iout

        # Create vertex buffer
        # The orientation of each triangle tile is defined as the direction perpendicular to the first edge of the triangle;
        # Here each orientation vector is represented by a complex number for the convenience of later computation
        tile_orientation = geometry.vecNormalize(np.cross(tileCen, usV[usF[:, 1], :] - usV[usF[:, 0], :])) \
                           + 1.j * geometry.vecNormalize(usV[usF[:, 1], :] - usV[usF[:, 0], :])
        tileDist = geometry.sphAngle(tileCen, sphereR)  # Spherical distance for each tile pair
        usF = np.int64(usF.flatten())
        # Triangles must not share edges/vertices while doing texture mapping, this line duplicate the shared vertices for each triangle
        # self.a_position = geometry.vecNormalize(usV[usF, :])
        self.a_position = np.ascontiguousarray(geometry.vecNormalize(usV[usF, :]), dtype=np.float32)
        self.a_texcoord = geometry.cen2tri(np.random.rand(int(Iout.size / 3)), np.random.rand(int(Iout.size / 3)),
                                           .1).reshape([Iout.size, 2])

        self.tile_orientation = tile_orientation
        self.tile_center = tileCen
        self.intertile_distance = tileDist

        # Calculate coordinates in azimuth and elevation
        spherical_coordinates = self.get_vertices()
        az, el = np.array(
            geometry.cart2sph1(spherical_coordinates[0, :], spherical_coordinates[1, :], spherical_coordinates[2, :]))
        # az = np.linspace(0, self.azimuth_range, self.azim_lvls, endpoint=True)
        # el = np.linspace(-np.pi / 2, self.upper_elev, self.elev_lvls, endpoint=True)
        self.azims, self.elevs = np.meshgrid(az, el)

        # Set vertex attributes
        self.a_azimuth = np.ascontiguousarray(self.azims.flatten(), dtype=np.float32)
        self.azimuth_degree = np.ascontiguousarray(self.azims.flatten() / np.pi / 2.0 * 360.0, dtype=np.float32)
        self.azimuth_degree2 = np.ascontiguousarray((self.azims.flatten() - np.pi) / np.pi / 2.0 * 360.0,
                                                    dtype=np.float32)
        self.azimuth_degree_zero_front_pos_cw = -np.ascontiguousarray(
            (self.azims.flatten() - np.pi) / np.pi / 2.0 * 360.0, dtype=np.float32)
        self.a_elevation = np.ascontiguousarray(self.elevs.flatten(), dtype=np.float32)
        self.elevation_degree = np.ascontiguousarray(self.elevs.flatten() / np.pi / 2.0 * 360.0, dtype=np.float32)
        self.az = np.ascontiguousarray(az.flatten(), dtype=np.float32)
        self.el = np.ascontiguousarray(el.flatten(), dtype=np.float32)

    def vertex(self, x, y, z):
        vlen = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return [i / vlen for i in (x, y, z)]

    def get_vertices(self):
        return np.ascontiguousarray(np.array(self.coordinates, dtype=np.float32))

    def get_spherical_coordinates(self):
        spherical_coordinates = self.get_vertices()
        az, el = np.array(
            geometry.cart2sph1(spherical_coordinates[0, :], spherical_coordinates[1, :], spherical_coordinates[2, :]))
        return np.ascontiguousarray(az), np.ascontiguousarray(el)


class IcoSphere_UVsphere_new:

    def __init__(self,
                 radius: float = 1.0,
                 subdivisionTimes: int = 1):
        ### Create sphere
        self.r = (1 + np.sqrt(5)) / 2
        self.init_vertices = np.array([
            [-1.0, self.r, 0.0],
            [1.0, self.r, 0.0],
            [-1.0, -self.r, 0.0],
            [1.0, -self.r, 0.0],
            [0.0, -1.0, self.r],
            [0.0, 1.0, self.r],
            [0.0, -1.0, -self.r],
            [0.0, 1.0, -self.r],
            [self.r, 0.0, -1.0],
            [self.r, 0.0, 1.0],
            [-self.r, 0.0, -1.0],
            [-self.r, 0.0, 1.0]
        ])
        self.init_faces = np.array([
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1]
        ])
        self.sdtimes = subdivisionTimes
        self.radius = radius
        self.coordinates = [self.vertex(*v) for v in self.init_vertices]

        [usV, usF] = geometry.subdivide_triangle(self.init_vertices, self.init_faces,
                                                 self.sdtimes)  # Compute the radius of all the vertices
        sphereR = geometry.vecNorm(usV[0, :])  # Compute the radius of all the vertices
        tileCen = np.mean(usV[usF, :], axis=1)  # Compute the center of each triangle tiles
        tileCen = geometry.vecNormalize(tileCen)

        # Create index buffer
        Iout = np.arange(usF.size, dtype=np.uint32)
        self.indices = np.ascontiguousarray(Iout)

        # Create vertex buffer
        # The orientation of each triangle tile is defined as the direction perpendicular to the first edge of the triangle;
        # Here each orientation vector is represented by a complex number for the convenience of later computation
        tile_orientation = geometry.vecNormalize(np.cross(tileCen, usV[usF[:, 1], :] - usV[usF[:, 0], :])) \
                           + 1.j * geometry.vecNormalize(usV[usF[:, 1], :] - usV[usF[:, 0], :])
        tileDist = geometry.sphAngle(tileCen, sphereR)  # Spherical distance for each tile pair
        usF = np.int64(usF.flatten())
        # Triangles must not share edges/vertices while doing texture mapping, this line duplicate the shared vertices for each triangle
        self.a_position = geometry.vecNormalize(usV[usF, :])
        self.a_position = np.ascontiguousarray(geometry.vecNormalize(usV[usF, :]), dtype=np.float32)
        self.a_texcoord = geometry.cen2tri(np.random.rand(int(Iout.size / 3)), np.random.rand(int(Iout.size / 3)),
                                           .1).reshape([Iout.size, 2])

        self.tile_orientation = tile_orientation
        self.tile_center = tileCen
        self.tile_center_inshader = np.ascontiguousarray(np.repeat(tileCen, 3, 0), dtype=np.float32)

        # self.triangle = usV[usF, :]
        self.intertile_distance = tileDist

        # Calculate coordinates in azimuth and elevation
        azimuth = []
        elevation = []
        for i in range(len(self.a_position)):
            spherical_cordinates = self.a_position[i]
            az, el = np.array(
                geometry.cart2sph1(spherical_cordinates[0], spherical_cordinates[1], spherical_cordinates[2]))
            azimuth.append([az])
            elevation.append([el])

        # Set vertex attributes
        self.a_azimuth = np.ascontiguousarray(azimuth, dtype=np.float32)
        self.a_elevation = np.ascontiguousarray(elevation, dtype=np.float32)

        # Calculate coordinates in triangle center
        triangle_center_x = []
        triangle_center_y = []
        triangle_num = []
        for i in range(len(self.tile_center)):
            tile_center_cordinates = self.tile_center[i]
            x, y = np.array(
                geometry.cart2sph1(tile_center_cordinates[0], tile_center_cordinates[1], tile_center_cordinates[2]))
            triangle_center_x.append([x])
            triangle_center_y.append([y])
            triangle_num.append(i)
        self.triangle_num = np.ascontiguousarray(triangle_num, dtype=np.float32)
        self.triangle_center_x = np.ascontiguousarray(triangle_center_x, dtype=np.float32)
        self.triangle_center_y = np.ascontiguousarray(triangle_center_y, dtype=np.float32)
        self.triangle_center_x = (np.repeat(self.triangle_center_x, 3, 0))
        self.triangle_center_y = (np.repeat(self.triangle_center_y, 3, 0))
        self.triangle_num = (np.repeat(self.triangle_num, 3, 0))
        self.triangle_size = geometry.vecNorm(self.a_position[0] - self.a_position[1])
        # radius = geometry.vecNorm(self.a_position)
        # print(f'{radius=}')





    def vertex(self, x, y, z):
        vlen = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return [i / vlen for i in (x, y, z)]

    def get_vertices(self):
        return np.ascontiguousarray(np.array(self.coordinates, dtype=np.float32))

    def get_spherical_coordinates(self):
        spherical_coordinates = self.get_vertices()
        az, el = np.array(
            geometry.cart2sph1(spherical_coordinates[0, :], spherical_coordinates[1, :], spherical_coordinates[2, :]))
        return np.ascontiguousarray(az), np.ascontiguousarray(el)


class Yunyi0430Sphere:

    def __init__(self,
                 azim_lvls: int = 32,
                 elev_lvls: int = 30,
                 azimuth_range: float = 2 * np.pi,
                 upper_elev: float = np.pi / 4,
                 radius: float = 1.0,
                 subdivisionTimes: int = 1):

        ### Create sphere
        self.r = (1 + np.sqrt(5)) / 2
        self.init_vertices = np.array([
            [-1.0, self.r, 0.0],
            [1.0, self.r, 0.0],
            [-1.0, -self.r, 0.0],
            [1.0, -self.r, 0.0],
            [0.0, -1.0, self.r],
            [0.0, 1.0, self.r],
            [0.0, -1.0, -self.r],
            [0.0, 1.0, -self.r],
            [self.r, 0.0, -1.0],
            [self.r, 0.0, 1.0],
            [-self.r, 0.0, -1.0],
            [-self.r, 0.0, 1.0]
        ])
        self.init_faces = np.array([
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1]
        ])
        self.sdtimes = subdivisionTimes
        # Set parameters
        self.azim_lvls = azim_lvls
        self.elev_lvls = elev_lvls
        self.azimuth_range = azimuth_range
        self.upper_elev = upper_elev
        self.radius = radius

        # Calculate coordinates in azimuth and elevation
        az = np.linspace(0, self.azimuth_range, self.azim_lvls, endpoint=True)
        el = np.linspace(-np.pi / 2, self.upper_elev, self.elev_lvls, endpoint=True)
        self.azims, self.elevs = np.meshgrid(az, el)

        # Set vertex attributes
        self.a_azimuth = np.ascontiguousarray(self.azims.flatten(), dtype=np.float32)
        self.azimuth_degree = np.ascontiguousarray(self.azims.flatten() / np.pi / 2.0 * 360.0, dtype=np.float32)
        self.azimuth_degree2 = np.ascontiguousarray((self.azims.flatten() - np.pi) / np.pi / 2.0 * 360.0,
                                                    dtype=np.float32)
        self.azimuth_degree_zero_front_pos_cw = -np.ascontiguousarray(
            (self.azims.flatten() - np.pi) / np.pi / 2.0 * 360.0, dtype=np.float32)
        self.a_elevation = np.ascontiguousarray(self.elevs.flatten(), dtype=np.float32)
        self.elevation_degree = np.ascontiguousarray(self.elevs.flatten() / np.pi / 2.0 * 360.0, dtype=np.float32)
        self.a_position = geometry.sph2cart(self.a_azimuth, self.a_elevation, self.radius)
        self.a_position = np.ascontiguousarray(self.a_position.T, dtype=np.float32)

        # Set indices
        idcs = list()
        for i in np.arange(self.elev_lvls):
            for j in np.arange(self.azim_lvls):
                idcs.append([i * azim_lvls + j, i * azim_lvls + j + 1, (i + 1) * azim_lvls + j + 1])
                idcs.append([i * azim_lvls + j, (i + 1) * azim_lvls + j, (i + 1) * azim_lvls + j + 1])

        self.indices = np.ascontiguousarray(np.array(idcs).flatten(), dtype=np.uint32)

        [usV, usF] = geometry.subdivide_triangle(self.init_vertices, self.init_faces,
                                                 self.sdtimes)  # Compute the radius of all the vertices
        sphereR = geometry.vecNorm(usV[0, :])  # Compute the radius of all the vertices
        tileCen = np.mean(usV[usF, :], axis=1)  # Compute the center of each triangle tiles
        tileCen = geometry.vecNormalize(tileCen)

        # Create index buffer
        Iout = np.arange(azim_lvls * elev_lvls, dtype=np.uint32)
        # self.indices = Iout

        # Create vertex buffer
        # The orientation of each triangle tile is defined as the direction perpendicular to the first edge of the triangle;
        # Here each orientation vector is represented by a complex number for the convenience of later computation
        tile_orientation = geometry.vecNormalize(np.cross(tileCen, usV[usF[:, 1], :] - usV[usF[:, 0], :])) \
                           + 1.j * geometry.vecNormalize(usV[usF[:, 1], :] - usV[usF[:, 0], :])
        tileDist = geometry.sphAngle(tileCen, sphereR)  # Spherical distance for each tile pair
        usF = np.int64(usF.flatten())
        # Triangles must not share edges/vertices while doing texture mapping, this line duplicate the shared vertices for each triangle
        # self.a_position = geometry.vecNormalize(usV[usF, :])
        # self.a_texcoord = geometry.cen2tri(np.random.rand(int(Iout.size / 3)), np.random.rand(int(Iout.size / 3)),.1).reshape([Iout.size, 2])

        self.tile_orientation = tile_orientation
        self.tile_center = tileCen
        self.intertile_distance = tileDist

    def get_uv_coordinates(self):
        vecs = self.a_position
        u = 0.5 + np.arctan2(vecs[:, 0], vecs[:, 1]) / np.pi / 2.0
        v = 0.5 + np.arcsin(-vecs[:, 2]) / np.pi

        return np.stack([u, v]).T


class binary_Sphere:

    def __init__(self,
                 azim_lvls: int = 60,
                 elev_lvls: int = 30,
                 azimuth_range: float = 2 * np.pi,
                 upper_elev: float = np.pi / 4,
                 radius: float = 1.0):

        # Set parameters
        self.azim_lvls = azim_lvls
        self.elev_lvls = elev_lvls
        self.azimuth_range = azimuth_range
        self.upper_elev = upper_elev
        self.radius = radius

        # Calculate coordinates in azimuth and elevation
        az = np.linspace(0, self.azimuth_range, self.azim_lvls, endpoint=True)
        el = np.linspace(-np.pi / 2, self.upper_elev, self.elev_lvls, endpoint=True)
        self.azims, self.elevs = np.meshgrid(az, el)

        # Set vertex attributes
        self.a_azimuth = np.ascontiguousarray(self.azims.flatten(), dtype=np.float32)
        self.azimuth_degree = np.ascontiguousarray(self.azims.flatten() / np.pi / 2.0 * 360.0, dtype=np.float32)
        self.azimuth_degree2 = np.ascontiguousarray((self.azims.flatten() - np.pi) / np.pi / 2.0 * 360.0,
                                                    dtype=np.float32)
        self.azimuth_degree_zero_front_pos_cw = -np.ascontiguousarray(
            (self.azims.flatten() - np.pi) / np.pi / 2.0 * 360.0, dtype=np.float32)
        self.a_elevation = np.ascontiguousarray(self.elevs.flatten(), dtype=np.float32)
        self.elevation_degree = np.ascontiguousarray(self.elevs.flatten() / np.pi / 2.0 * 360.0, dtype=np.float32)
        self.a_position = geometry.sph2cart(self.a_azimuth, self.a_elevation, self.radius)
        self.a_position = np.ascontiguousarray(self.a_position.T, dtype=np.float32)

        # Set indices
        idcs = list()
        for i in np.arange(self.elev_lvls / 2):
            for j in np.arange(self.azim_lvls / 2):
                idcs.append([i * azim_lvls + j, i * azim_lvls + j + 1, (i + 1) * azim_lvls + j + 1])
                idcs.append([i * azim_lvls + j, (i + 1) * azim_lvls + j, (i + 1) * azim_lvls + j + 1])

        self.indices = np.ascontiguousarray(np.array(idcs).flatten(), dtype=np.uint32)

    def get_uv_coordinates(self):
        vecs = self.a_position
        u = 0.5 + np.arctan2(vecs[:, 0], vecs[:, 1]) / np.pi / 2.0
        v = 0.5 + np.arcsin(-vecs[:, 2]) / np.pi

        return np.stack([u, v]).T


class binary_Sphere2:

    def __init__(self,
                 azim_lvls: int = 6,
                 elev_lvls: int = 6,
                 # azim_lvls: int = 5 * np.pi / 9,
                 # elev_lvls: int = np.pi / 3,
                 azimuth_range: float = 2 * np.pi,
                 elev_range: float = 3 * np.pi / 4,
                 radius: float = 1.0, ):
        # Set parameters
        self.azim_lvls = azim_lvls
        self.elev_lvls = elev_lvls
        self.azimuth_range = azimuth_range
        self.elev_range = elev_range

        # Calculate coordinates in azimuth and elevation
        az = np.linspace(5 * np.pi / 9, 11 * np.pi / 18, self.azim_lvls, endpoint=True)
        el = np.linspace(np.pi / 3, np.pi / 2, self.elev_lvls, endpoint=True)
        self.azims, self.elevs = np.meshgrid(az, el)

        # Set vertex attributes
        self.a_azimuth = np.ascontiguousarray(self.azims.flatten(), dtype=np.float32)
        self.a_elevation = np.ascontiguousarray(self.elevs.flatten(), dtype=np.float32)
        # corners = [[-1.0, .0, .0],[-1.0, 0.5, .0],[-1.0, -1.0, .0],[1.0, .0, .0],[.0, -1.0, .0],[1.0, 1.0, .0],[.0, -1.0, .0],[.0, 1.0, .0]]

        self.a_position = (self.a_azimuth, self.a_elevation, 1.0)

        # Set indices
        idcs = list()
        for i in np.arange(self.elev_lvls):
            for j in np.arange(self.azim_lvls):
                idcs.append([i * azim_lvls + j, i * azim_lvls + j + 1, (i + 1) * azim_lvls + j + 1])
                idcs.append([i * azim_lvls + j, (i + 1) * azim_lvls + j, (i + 1) * azim_lvls + j + 1])

        self.indices = np.ascontiguousarray(np.array(idcs).flatten(), dtype=np.uint32)


class IcosahedronSphere:
    gr = 1.61803398874989484820

    corners = [
        [-1, gr, 0],
        [1, gr, 0],
        [-1, -gr, 0],
        [1, -gr, 0],
        [0, -1, gr],
        [0, 1, gr],
        [0, -1, -gr],
        [0, 1, -gr],
        [gr, 0, -1],
        [gr, 0, 1],
        [-gr, 0, -1],
        [-gr, 0, 1],
    ]

    _faces = [

        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],

        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],

        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],

        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    def __init__(self, subdiv_lvl, **kwargs):

        # Calculate initial vertices
        self._vertices = [self._vertex(*v) for v in self.corners]
        self._vertex_lvls = [0] * len(self.corners)

        # Subdivide faces
        self._cache = None
        self.subdiv_lvl = subdiv_lvl
        self._subdivide()

    def get_indices(self):
        return np.ascontiguousarray(np.array(self._faces, dtype=np.uint32)).flatten()

    def get_vertices(self):
        return np.ascontiguousarray(np.array(self._vertices, dtype=np.float32))

    def get_vertex_levels(self):
        return np.ascontiguousarray(np.array(self._vertex_lvls, dtype=np.int32))

    def get_spherical_coordinates(self):
        _vertices = self.get_vertices()
        az, el = np.array(geometry.cart2sph1(_vertices[0, :], _vertices[1, :], _vertices[2, :]))
        return np.ascontiguousarray(az), np.ascontiguousarray(el)

    def _vertex(self, x, y, z):
        vlen = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return [i / vlen for i in (x, y, z)]

    def _midpoint(self, p1, p2, v_lvl):
        key = '%i/%i' % (min(p1, p2), max(p1, p2))

        if key in self._cache:
            return self._cache[key]

        v1 = self._vertices[p1]
        v2 = self._vertices[p2]
        middle = [sum(i) / 2 for i in zip(v1, v2)]

        self._vertices.append(self._vertex(*middle))
        self._vertex_lvls.append(v_lvl)
        index = len(self._vertices) - 1

        self._cache[key] = index

        return index

    def _subdivide(self):
        self._cache = {}
        for i in range(self.subdiv_lvl):
            new_faces = []
            for face in self._faces:
                v = [self._midpoint(face[0], face[1], i + 1),
                     self._midpoint(face[1], face[2], i + 1),
                     self._midpoint(face[2], face[0], i + 1)]

                new_faces.append([face[0], v[0], v[2]])
                new_faces.append([face[1], v[1], v[0]])
                new_faces.append([face[2], v[2], v[1]])
                new_faces.append([v[0], v[1], v[2]])

            self._faces = new_faces


class CMNIcoSphere:

    def __init__(self, subdivisionTimes: int = 1):
        ### Create sphere
        self.r = (1 + np.sqrt(5)) / 2
        self.init_vertices = np.array([
            [-1.0, self.r, 0.0],
            [1.0, self.r, 0.0],
            [-1.0, -self.r, 0.0],
            [1.0, -self.r, 0.0],
            [0.0, -1.0, self.r],
            [0.0, 1.0, self.r],
            [0.0, -1.0, -self.r],
            [0.0, 1.0, -self.r],
            [self.r, 0.0, -1.0],
            [self.r, 0.0, 1.0],
            [-self.r, 0.0, -1.0],
            [-self.r, 0.0, 1.0]
        ])
        self.init_faces = np.array([
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1]
        ])
        self.sdtimes = subdivisionTimes
        [usV, usF] = geometry.subdivide_triangle(self.init_vertices, self.init_faces,
                                                 self.sdtimes)  # Compute the radius of all the vertices
        sphereR = geometry.vecNorm(usV[0, :])  # Compute the radius of all the vertices
        tileCen = np.mean(usV[usF, :], axis=1)  # Compute the center of each triangle tiles
        tileCen = geometry.vecNormalize(tileCen)

        # Create index buffer
        Iout = np.arange(usF.size, dtype=np.uint32)
        self.indices = Iout

        # Create vertex buffer
        # The orientation of each triangle tile is defined as the direction perpendicular to the first edge of the triangle;
        # Here each orientation vector is represented by a complex number for the convenience of later computation
        tile_orientation = geometry.vecNormalize(np.cross(tileCen, usV[usF[:, 1], :] - usV[usF[:, 0], :])) \
                           + 1.j * geometry.vecNormalize(usV[usF[:, 1], :] - usV[usF[:, 0], :])
        tileDist = geometry.sphAngle(tileCen, sphereR)  # Spherical distance for each tile pair
        usF = np.int64(usF.flatten())
        # Triangles must not share edges/vertices while doing texture mapping, this line duplicate the shared vertices for each triangle
        self.a_position = geometry.vecNormalize(usV[usF, :])
        self.a_texcoord = geometry.cen2tri(np.random.rand(int(Iout.size / 3)), np.random.rand(int(Iout.size / 3)),
                                           .1).reshape([Iout.size, 2])

        self.tile_orientation = tile_orientation
        self.tile_center = tileCen
        self.intertile_distance = tileDist


if __name__ == '__main__':

    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    from vxpy.utils import sphere
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    s = sphere.SimpleUVSphere()
    idcs, pos, faces = s.get_subset(-np.pi, np.pi, -np.pi / 4, np.pi / 4)

    ax = plt.subplot(projection='3d')

    ax.scatter(*pos.T)

    poly3d = []
    for face in faces:
        poly3d.append(pos[[*face, face[0]]])
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors='b', linewidths=1, alpha=0.5))
    ax.add_collection3d(Line3DCollection(poly3d, colors='k', linewidths=1, linestyles=':'))
    # f = [*face, face[0]]
    # ax.plot(*pos[f, :].T, color='blue', linestyle='--')
    plt.show()
