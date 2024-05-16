import numpy as np
from vispy import gloo, app
from scipy.ndimage import gaussian_filter
import vxpy.core.visual as vxvisual
from vxpy.utils import plane
import matplotlib.pyplot as plt

#width = 98
#height = 734

width = 1000
height = 1000

diagonal = 20.6  #in cm


def calculate_power_spectrum(texture):
    fft_texture = np.fft.fftshift(np.fft.fft2(texture))
    power_spectrum = np.abs(fft_texture) ** 2
    return power_spectrum


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
    step_width = vxvisual.FloatParameter('step_width', default=0.01, limits=(0, 1), step_size=0.01, static=False)

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

        self.time.connect(self.noise)
        self.sigma.connect(self.noise)
        self.width.connect(self.noise)
        self.height.connect(self.noise)

    def initialize(self, *args, **kwargs):
        # Reset time to 0.0 on each visual initialization
        self.time.data = 0.0

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt
        print(dt)
        self.sigma.data += self.step_width.data
        self.width.data = width
        self.height.data = height
        #print(self.sigma.data)

        white_noise = generate_white_noise(self.width.data, self.height.data, self.seed.data)
        lowpass_filtered = apply_lowpass_filter(white_noise, self.sigma.data)

        # Calculate frequency cutoff
        cutoff_frequency = 1 / (2 * np.pi * self.sigma.data)
        #print("cutoff frequency:", cutoff_frequency)

        # Create power spectrum
        power_spectrum = calculate_power_spectrum(lowpass_filtered)
        plt.imshow(np.log(power_spectrum), cmap='viridis')
        plt.colorbar()
        plt.title(self.sigma.data)
        plt.show()

        self.noise['u_min_value'] = np.min(lowpass_filtered)
        self.noise['u_max_value'] = np.max(lowpass_filtered)
        self.texture.set_data(lowpass_filtered)
        self.noise.draw('triangles', self.index_buffer)
