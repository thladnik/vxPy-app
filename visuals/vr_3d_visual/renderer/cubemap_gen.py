import os

import time
from math import pi, sin, cos

import h5py
import numpy as np
from direct.actor.Actor import Actor
from direct.interval.MetaInterval import Sequence
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from panda3d.core import LPoint3f, ModelLoadRequest, Point3, TexGenAttrib, TextureStage, VBase3
from panda3d.core import NodePath
from direct.task import Task


def get_abspath(path):
    return os.path.join(os.getcwd(), *'visuals/vr_3d_visual/renderer'.split('/'), *path.split('/'))


class MyApp(ShowBase):

    def __init__(self, motion_data_presets: dict = None):
        # ShowBase.__init__(self, windowType='offscreen')
        ShowBase.__init__(self)
        global fb_size

        self.speed = 1

        self.motion_data_presets = motion_data_presets
        self.world_rot = False
        self.third_person = False
        self.start_time = time.perf_counter()
        self.default_z = 8.0

        # self.disableMouse()
        self.enableMouse()

        # Load the environment model
        self.scene = self.loader.loadModel(get_abspath('OceanFloor/OceanFloor.egg'))
        self.scene.setZ(-2)
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(0.1, 0.1, 0.1)
        self.scene.setPos(0, 0, 0)

        # Add tasks
        self.taskMgr.add(self.spin_camera_task, "SpinCameraTask")
        self.taskMgr.add(self.buffer_read_task, "BufferReadTask")

        # Add cube
        # self.cube = self.loader.loadModel('cube.egg')
        # self.cube.setScale(0.5, 0.5, 0.5)
        # self.cube = self.loader.loadModel('blueminnow/blueminnow.egg')
        self.main_actor = self.loader.loadModel(get_abspath('Goldfish/Goldfish.egg'))
        # self.cube.setScale(3., 3., 3.)
        self.main_actor.setScale(0.5, 0.5, 0.5)
        self.main_actor.setPos(0.0, 0.0, self.default_z)
        self.main_actor.reparentTo(self.render)

        # Make camera follow actor
        if not self.world_rot:
            if self.third_person:
                self.disableMouse()
                self.camera.reparentTo(self.main_actor)
                self.camera.setPos(0, -10, 2)
                self.camera.lookAt(self.main_actor)
            else:
                self.disableMouse()
                self.camera.setPos(40, -40, 40)


        self.rig = NodePath('rig')
        self.cbbuffer = self.win.makeCubeMap('env', fb_size, self.rig, to_ram=True)
        self.rig.reparentTo(self.main_actor)

        # Add texture to model (reflections)
        # self.cube.setTexGen(TextureStage.getDefault(), TexGenAttrib.MWorldCubeMap)
        # self.cube.setTexture(self.cbbuffer.getTexture())

        vegetation_spread = 40
        np.random.seed(1)
        self.finger_corals = [self.loader.loadModel(get_abspath('FingerCoral/FingerCoral.egg')) for _ in range(15)]
        for c in self.finger_corals:
            c.setPos(vegetation_spread / 2 * np.random.rand(), vegetation_spread * (np.random.rand() - 0.5),
                     -np.random.rand() / 2)
            c.setScale(*[2 * np.random.rand()] * 3)
            c.setH(360 * np.random.rand())
            c.reparentTo(self.render)

        self.fire_corals = [self.loader.loadModel(get_abspath('FireCoral/FireCoral.egg')) for _ in range(15)]
        for c in self.fire_corals:
            c.setPos(-vegetation_spread / 2 * np.random.rand(), vegetation_spread * (np.random.rand() - 0.5),
                     -np.random.rand() / 2)
            c.setScale(*[2 * np.random.rand()] * 3)
            c.setH(360 * np.random.rand())
            c.reparentTo(self.render)

        self.seaweeds = [self.loader.loadModel(get_abspath('seaweed/seaweed.egg')) for _ in range(30)]
        for c in self.seaweeds:
            c.setPos(-vegetation_spread * (np.random.rand() - 0.5), vegetation_spread * (np.random.rand() - 0.5),
                     -np.random.rand() / 2)
            c.setScale(*[15 * np.random.rand()] * 3)
            c.setH(360 * np.random.rand())
            c.reparentTo(self.render)

        # Add fishies
        self.smallfishes = [self.loader.loadModel(get_abspath('blueminnow/blueminnow.egg')) for _ in range(10)]
        self.smallfishes = self.smallfishes + [self.loader.loadModel(get_abspath('yellowminnow/yellowminnow.egg')) for _ in range(10)]
        self.fish_swims = []
        for fish in self.smallfishes:
            fish.setScale(*(1 + np.random.rand(3)))
            fish.reparentTo(self.render)
            self.fish_swims.append(self.create_circular_swim_sequence(fish, 10))
            self.fish_swims[-1].loop()

        self.bigfishes = [*[self.loader.loadModel(get_abspath('lilfish/lilfish.egg')) for _ in range(10)]]
        for fish in self.bigfishes:
            fish.setScale(*(0.5 + np.random.rand(3)/2))
            fish.reparentTo(self.render)
            self.fish_swims.append(self.create_circular_swim_sequence(fish, 10))
            self.fish_swims[-1].loop()

        self.intert = time.perf_counter()

        self.key_control_map = {'up': False,
                                'down': False,
                                'forward': False,
                                'backward': False,
                                'left': False,
                                'right': False,
                                'cw': False,
                                'ccw': False}

        self.accept("page_up", self.update_key_control_map, ['up', True])
        self.accept("page_up-up", self.update_key_control_map, ['up', False])
        self.accept("page_down", self.update_key_control_map, ['down', True])
        self.accept("page_down-up", self.update_key_control_map, ['down', False])
        self.accept("w", self.update_key_control_map, ['forward', True])
        self.accept("w-up", self.update_key_control_map, ['forward', False])
        self.accept("s", self.update_key_control_map, ['backward', True])
        self.accept("s-up", self.update_key_control_map, ['backward', False])
        self.accept("a", self.update_key_control_map, ['left', True])
        self.accept("a-up", self.update_key_control_map, ['left', False])
        self.accept("d", self.update_key_control_map, ['right', True])
        self.accept("d-up", self.update_key_control_map, ['right', False])
        self.accept("q", self.update_key_control_map, ['ccw', True])
        self.accept("q-up", self.update_key_control_map, ['ccw', False])
        self.accept("e", self.update_key_control_map, ['cw', True])
        self.accept("e-up", self.update_key_control_map, ['cw', False])

        self.taskMgr.add(self.move_actor_fish)

    def update_key_control_map(self, name, state):
        self.key_control_map[name] = state

    @staticmethod
    def create_circular_swim_sequence(actor, position_count) -> Sequence:

        def duration(v1, v2):
            return np.linalg.norm(v1 - v2) / (5 + 5 * np.random.rand())

        def rotation(v1, v2):
            dir_vec = (v2 - v1) / np.linalg.norm(v2 - v1)
            rot_dir = np.arctan2(dir_vec[1], dir_vec[0])
            return rot_dir / np.pi  * 180 - 90

        first_pos = [*(40 * (np.random.rand(2) - 0.5)), 10 + 5 * np.random.rand()]
        positions = [np.array(first_pos)]

        for i in range(position_count):
            pos = positions[-1] + np.array([20, 20, 2]) * (np.random.rand(3) - 0.5)
            positions.append(pos)

        positions = np.array(positions)

        actor.setPos(*positions[0])
        all_intervals = []
        for v1, v2 in zip(positions[:-1], positions[1:]):
            trans_interval = actor.posInterval(duration(v1, v2),
                                         Point3(*v2),
                                         startPos=Point3(*v1))

            rot_interval = actor.hprInterval(1, VBase3(rotation(v1, v2), 0, 0))
            all_intervals.append(rot_interval)
            all_intervals.append(trans_interval)

        # Loop around to first
        rot_interval = actor.hprInterval(1, VBase3(rotation(positions[-1], positions[0]), 0, 0))
        trans_interval = actor.posInterval(duration(positions[-1], positions[0]),
                                     Point3(*positions[0]),
                                     startPos=Point3(*positions[-1]))
        all_intervals.append(rot_interval)
        all_intervals.append(trans_interval)

        seq = Sequence(*all_intervals)

        return seq


    def _move_by_motion_presets(self):
        t = (time.perf_counter() - self.start_time) * self.speed

        idx = np.argmin(np.abs(self.motion_data_presets['time']-t))

        h = self.motion_data_presets['orientation'][idx]
        x, y = self.motion_data_presets['position'][idx]

        self.main_actor.setH(h)
        self.rig.setH(h)
        self.main_actor.setPos(x, y, self.default_z)


    def move_actor_fish(self, task):
        global x_pos, y_pos, heading
        dt = globalClock.getDt() * self.speed

        if self.motion_data_presets is not None:
            self._move_by_motion_presets()
            return task.cont

        rot_scale = 90 * dt
        rot_directions = {'cw': -1, 'ccw': 1}

        for direction, state in self.key_control_map.items():

            if not state:
                continue

            rot = self.main_actor.getH()
            pos = self.main_actor.getPos()
            x_comp_fb = -np.sin(rot / 360 * 2 * np.pi)
            y_comp_fb = np.cos(rot / 360 * 2 * np.pi)

            x_comp_lr = np.sin((rot + 90) / 360 * 2 * np.pi)
            y_comp_lr = -np.cos((rot + 90) / 360 * 2 * np.pi)

            trans_scale = 5 * dt
            trans_directions = {'forward': (x_comp_fb, y_comp_fb, 0),
                                'backward': (-x_comp_fb, -y_comp_fb, 0),
                                'right': (x_comp_lr, y_comp_lr, 0),
                                'left': (-x_comp_lr, -y_comp_lr, 0),
                                'up': (0, 0, 1),
                                'down': (0, 0, -1)}

            if direction in trans_directions:
                print(f'Translate {direction}', pos)
                translation = LPoint3f(*[trans_scale * v for v in trans_directions[direction]])
                self.main_actor.setPos(pos + translation)

            if direction in rot_directions:
                print(f'Rotate {direction} by {rot_scale} from {self.rig.getH()}deg')
                rotation = rot_scale * rot_directions[direction]
                self.main_actor.setH(self.main_actor.getH() + rotation)
                self.rig.setH(self.rig.getH() + rotation)
                print(f'>{self.rig.getH()}deg')

        x_pos.value = self.main_actor.getPos().get_x()
        y_pos.value = self.main_actor.getPos().get_y()
        heading.value = self.main_actor.getH()

        return task.cont

    def spin_camera_task(self, task):

        if self.world_rot:
            angleDegrees = 6.0
            angleRadians = angleDegrees * (pi / 180.0)
            self.camera.setPos(100 * sin(angleRadians), -100 * cos(angleRadians), 100)
            self.camera.setHpr(angleDegrees, -40, 0)
        else:
            if not self.third_person:
                self.camera.lookAt(self.main_actor)
        # self.camera.setPos(0, 0, 40)
        # self.camera.setHpr(0, -90, 0)
        return Task.cont

    def read_buffer(self):
        global data, fb_size

        tex = self.cbbuffer.getTexture(0)

        if not tex.hasRamImage():
            return None

        buffer_contents = np.frombuffer(tex.getRamImage(), dtype=np.uint8)
        # (cube sides, res_x, res_y, rgba)
        buffer_contents.shape = (6, fb_size, fb_size, 4)

        data[:] = buffer_contents

    def buffer_read_task(self, task):
        # print(f'Inter: {(time.perf_counter()-self.intert)*1000:.2f}ms')

        t = time.perf_counter()

        self.read_buffer()

        t = time.perf_counter() - t
        # print(f'{t*1000:.2f}ms')

        # cv2.waitKey(0)

        self.intert = time.perf_counter()
        return Task.cont


def run(raw_data, _x, _y, _heading, _fb_size):
    global data, fb_size, x_pos, y_pos, heading
    fb_size = _fb_size
    x_pos = _x
    y_pos = _y
    heading = _heading
    data = np.frombuffer(raw_data.get_obj(), dtype=np.uint8).reshape((6, fb_size, fb_size, 4))

    # with h5py.File('./2023-08-31_fish1_rec1_p20/Display.hdf5', 'r') as f:
    #     motion_data = {'time': f['ABCMeta_0/time'][:].squeeze(),
    #                    'orientation': f['ABCMeta_0/orientation'][:].squeeze(),
    #                    'position': f['ABCMeta_0/position'][:].squeeze()}

    motion_data = None

    p3d_app = MyApp(motion_data_presets=motion_data)
    p3d_app.run()
