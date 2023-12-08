import numpy as np
from vispy import app, gloo
from vispy.util.transforms import ortho
from scipy.ndimage import gaussian_filter
import vxpy.core.visual as vxvisual

W, H = 1000, 1000
pixels_height = 1000
pixels_width = 2000

def generate_white_noise(width, height, seed):
    np.random.seed(seed)
    return np.random.rand(*width, *height)

# Function to apply a low-pass filter
def apply_lowpass_filter(image, sigma):
    sigma = sigma[0]
    return gaussian_filter(image, sigma)


# A simple texture quad
data = np.zeros(4, dtype=[('a_position', np.float32, 2), ('a_texcoord', np.float32, 2)])
data['a_position'] = np.array([[0, 0], [W, 0], [0, H], [W, H]])
data['a_texcoord'] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


class FilteredNoise(vxvisual.PlanarVisual):
    time = vxvisual.FloatParameter('time', internal=True)
    sigma = vxvisual.FloatParameter('sigma', default=1.0, limits=(0, 100), step_size=0.1, static=False)
    width = vxvisual.IntParameter('width', static=True, internal=True)
    height = vxvisual.IntParameter('height', static=True, internal=True)
    seed = vxvisual.IntParameter('seed', static=True)
    step_width = vxvisual.FloatParameter('step_width', default=0.01, limits=(0, 1), step_size=0.01, static=False)

    def __init__(self, *args, **kwargs):
        vxvisual.PlanarVisual.__init__(self, *args, **kwargs)

        self.width.data = pixels_width
        self.height.data = pixels_height
        self.seed.data = 1
        self.sigma.data = 1.0
        white_noise = generate_white_noise(self.width.data, self.height.data, self.seed.data)
        lowpass_filtered = apply_lowpass_filter(white_noise, self.sigma.data)

        vert = self.load_vertex_shader('./filtered_noise.vert')
        frag = self.load_shader('./filtered_noise.frag')
        self.program = gloo.Program(vert, frag)
        self.texture = gloo.Texture2D(lowpass_filtered, interpolation='linear')

        self.program['u_texture'] = self.texture
        self.program.bind(gloo.VertexBuffer(data))

        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.projection = ortho(0, W, 0, H, -1, 1)
        self.program['u_projection'] = self.projection
        self.program['u_min_value'] = 0.0
        self.program['u_max_value'] = 1.0

        gloo.set_clear_color('white')

        self.sigma.connect(self.program)

    def initialize(self, *args, **kwargs):
        # Reset time to 0.0 on each visual initialization
        self.time.data = 0.0

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt
        #print(dt)
        self.sigma.data += self.step_width.data
        self.width.data = pixels_width
        self.height.data = pixels_height
        print(self.sigma.data)
        white_noise = generate_white_noise(self.width.data, self.height.data, self.seed.data)
        lowpass_filtered = apply_lowpass_filter(white_noise, self.sigma.data)
        self.program['u_min_value'] = np.min(lowpass_filtered)
        self.program['u_max_value'] = np.max(lowpass_filtered)
        gloo.clear(color=True, depth=True)
        self.texture.set_data(lowpass_filtered)
        self.program.draw('triangle_strip')

# window size
