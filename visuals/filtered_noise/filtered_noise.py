import numpy as np
from vispy import gloo
from scipy.ndimage import gaussian_filter
import vxpy.core.visual as vxvisual
from vxpy.utils import plane
from scipy import interpolate

width = 108  #in px
height = 727  #in px
diagonal = 207  #in mm

screen_width = 1920
screen_height = 1080
screen_diagonal = 6096 #in mm

distance_to_stimulus_mm = 100  # in mm
visual_acuity_cycles_per_degree = 0.24


def calculate_pixel_density(width_px, height_px, diagonal_mm):
    # Diagonal in pixels
    diagonal_px = np.sqrt(width_px ** 2 + height_px ** 2)
    # Pixel density in pixels per mm
    pixel_density = diagonal_px / diagonal_mm
    return pixel_density


def generate_white_noise(width, height, seed):
    np.random.seed(seed)
    return np.random.rand(*width, *height)


# Function to apply a low-pass filter
def apply_lowpass_filter(image, sigma):
    sigma = sigma[0]
    return gaussian_filter(image, sigma)


class FilteredNoise(vxvisual.PlanarVisual):
    time = vxvisual.FloatParameter('time', internal=True)
    sigma = vxvisual.FloatParameter('sigma', default=1.0, limits=(0, 100), step_size=0.1, static=False)
    width = vxvisual.IntParameter('width', static=True, internal=True)
    height = vxvisual.IntParameter('height', static=True, internal=True)
    seed = vxvisual.IntParameter('seed', static=True)
    start_sigma = vxvisual.FloatParameter('start_sigma', default=0.0, limits=(0, 100), static=False)
    end_sigma = vxvisual.FloatParameter('end_sigma', default=10.0, limits=(0, 100), static=False)
    duration = vxvisual.FloatParameter('duration', default=100.0, limits=(0, 900), static=False)
    cutoff_frequency = vxvisual.FloatParameter('cutoff_frequency', static=False, internal=True)

    def __init__(self, *args, **kwargs):
        vxvisual.PlanarVisual.__init__(self, *args, **kwargs)

        white_noise = generate_white_noise(self.width.data, self.height.data, self.seed.data)
        lowpass_filtered = apply_lowpass_filter(white_noise, self.sigma.data)

        # Set up model of a 2d plane
        self.plane_2d = plane.XYPlane()

        # Get vertex positions and corresponding face indices
        faces = self.plane_2d.indices
        vertices = self.plane_2d.a_position

        # Create vertex and index buffers
        self.index_buffer = gloo.IndexBuffer(faces)
        self.position_buffer = gloo.VertexBuffer(vertices)

        # Create a shader program
        vert = self.load_vertex_shader('./filtered_noise.vert')
        frag = self.load_shader('./filtered_noise.frag')
        self.noise = gloo.Program(vert, frag)
        self.texture = gloo.Texture2D(lowpass_filtered, interpolation='linear')
        self.noise['u_texture'] = self.texture

        # Set positions with vertex buffer
        self.noise['a_position'] = self.position_buffer

        self.duration.data = 0.0
        self.start_sigma.data = 0.0
        self.end_sigma.data = 0.0

        self.time.connect(self.noise)
        self.sigma.connect(self.noise)
        self.width.connect(self.noise)
        self.height.connect(self.noise)

    def initialize(self, *args, **kwargs):
        # Reset time to 0.0 on each visual initialization
        self.time.data = 0.0

    def render(self, dt):
        self.time.data += dt
        #print(dt)

        x = [0.0, float(self.duration.data)]
        y = [float(self.start_sigma.data), float(self.end_sigma.data)]
        f = interpolate.interp1d(x, y)
        self.sigma.data = float(f(min(self.time.data, self.duration.data)))

        self.width.data = width
        self.height.data = height
        #print(self.sigma.data)

        white_noise = generate_white_noise(self.width.data, self.height.data, self.seed.data)
        lowpass_filtered = apply_lowpass_filter(white_noise, self.sigma.data)

        # Calculate frequency cutoff
        pixel_density = calculate_pixel_density(screen_width, screen_height, screen_diagonal)
        sigma_mm = self.sigma.data / pixel_density
        self.cutoff_frequency.data = 1 / (2 * np.pi * sigma_mm)
        print("cutoff frequency:", self.cutoff_frequency.data)

        self.noise['u_min_value'] = np.min(lowpass_filtered)
        self.noise['u_max_value'] = np.max(lowpass_filtered)
        self.texture.set_data(lowpass_filtered)
        self.noise.draw('triangles', self.index_buffer)
