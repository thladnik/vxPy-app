"""Zebrafish larva tail tracking plugin for VxPy

Uses DeepLabCut and therefore requires optional dependency `dlclive`
"""
from __future__ import annotations

from typing import Dict, Hashable, List, Tuple, Union

import cv2
import numpy as np
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.spatial import distance

from scipy import optimize,signal
from scipy.interpolate import interp1d

import vxpy.core.attribute as vxattribute
import vxpy.core.devices.camera as vxcamera
import vxpy.core.dependency as vxdependency
import vxpy.core.io as vxio
import vxpy.core.ipc as vxipc
import vxpy.core.logger as vxlogger
import vxpy.core.routine as vxroutine
import vxpy.core.ui as vxui
from vxpy.utils.widgets import Checkbox, DoubleSliderWidget, IntSliderWidget, UniformWidth,SearchableListWidget
from dlclive import DLCLive, Processor #tt
from time import perf_counter

log = vxlogger.getLogger(__name__)



class ZFTailTrackingUI(vxui.CameraAddonWidget):
    display_name = 'ZF tail tracking'

    _vspacer = QtWidgets.QSpacerItem(1, 20,
                                     QtWidgets.QSizePolicy.Policy.Maximum,
                                     QtWidgets.QSizePolicy.Policy.MinimumExpanding)

    def __init__(self, *args, **kwargs):
        vxui.CameraAddonWidget.__init__(self, *args, **kwargs)

        self.central_widget.setLayout(QtWidgets.QHBoxLayout())

        # Set up control panel
        self.ctrl_panel = QtWidgets.QWidget(self)
        self.ctrl_panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
                                      QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.ctrl_panel.setLayout(QtWidgets.QVBoxLayout())
        self.central_widget.layout().addWidget(self.ctrl_panel)

        self.uniform_label_width = UniformWidth()




        # User interface for setting inference parameters
        self.pose_estimation_user_interface = QtWidgets.QGroupBox('Tracking Model Parameters')
        self.pose_estimation_user_interface.setLayout(QtWidgets.QVBoxLayout())
        self.ctrl_panel.layout().addWidget(self.pose_estimation_user_interface)


        # Confidence threshold for displaying labeled points of pose
        self.displaying_confidence_threshold = DoubleSliderWidget(self, 'P-cutoff',
                                                  limits=(0, 1),
                                                  default=ZFTailTracking.instance().pcutoff,
                                                  step_size=0.001)
        self.displaying_confidence_threshold.connect_callback(self.update_confidence_threshold)
        self.pose_estimation_user_interface.layout().addWidget(self.displaying_confidence_threshold)
        self.uniform_label_width.add_widget(self.displaying_confidence_threshold.label)

        # Downsampling in model inference
        self.downsampling_factor = DoubleSliderWidget(self, 'Downsampling',
                                                limits=(0, 1.0),
                                                default=ZFTailTracking.instance().downsample,
                                                step_size=0.001)
        self.downsampling_factor.connect_callback(self.update_downsampling_factor)
        self.pose_estimation_user_interface.layout().addWidget(self.downsampling_factor)
        self.uniform_label_width.add_widget(self.downsampling_factor.label)


        '''Window in which inference will happen'''
        # Add button for new ROI creation
        self.ctrl_panel.layout().addItem(self._vspacer)
        self.add_roi_btn = QtWidgets.QPushButton('Add Inference Box')
        self.add_roi_btn.clicked.connect(self.add_roi)
        self.ctrl_panel.layout().addWidget(self.add_roi_btn)


        # Set up image plot
        self.frame_plot = FramePlot(parent=self)
        self.frame_plot.setSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                                      QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.central_widget.layout().addWidget(self.frame_plot)

        self.connect_to_timer(self.update_frame)


    def add_roi(self):
        self.frame_plot.add_roi()

    @staticmethod
    def update_confidence_threshold(pcutoff_value):
        ZFTailTracking.instance().pcutoff = pcutoff_value

    @staticmethod
    def update_downsampling_factor(downsampling_rate):
        ZFTailTracking.instance().downsample = downsampling_rate


    def update_frame(self):


        idx, time, frame = vxattribute.read_attribute(ZFTailTracking.frame_name)
        frame = frame[0]

        if frame is None:
            return

        # Update image
        self.frame_plot.image_item.setImage(frame)





class FramePlot(pg.GraphicsLayoutWidget):
    # Set up basics
    line_coordinates = None
    current_id = 0
    inferece_box = []

    def __init__(self, **kwargs):
        pg.GraphicsLayoutWidget.__init__(self, **kwargs)

        # Set up plot image item
        self.image_plot = self.addPlot(0, 0, 1, 10)
        self.image_plot.hideAxis('left')
        self.image_plot.hideAxis('bottom')
        self.image_plot.invertY(True)
        self.image_plot.setAspectLocked(True)
        self.image_item = pg.ImageItem()
        self.image_plot.addItem(self.image_item)


        # Make subplots update with whole camera frame
        #self.image_item.sigImageChanged.connect(self.update_subplots)

        # Bind mouse click event for drawing of lines
        self.image_plot.scene().sigMouseClicked.connect(self.mouse_clicked)

    def mouse_clicked(self, ev):
        pos = self.image_plot.vb.mapSceneToView(ev.scenePos())

        # First click: start new line
        if self.line_coordinates is not None and len(self.line_coordinates) == 0:
            self.line_coordinates = [[pos.x(), pos.y()]]

        # Second click: end line and create rectangular ROI + subplot
        elif self.line_coordinates is not None and len(self.line_coordinates) == 1:

            # Set second point of line
            self.line_coordinates.append([pos.x(), pos.y()])

            # Create inference box
            self.inferece_box = InferenceBox(np.array(self.line_coordinates))
            ZFTailTracking.instance().inferece_box = self.inferece_box

            # draw inference box rectangle
            self.image_plot.vb.addItem(self.inferece_box.rect)

    def resizeEvent(self, ev):
        pg.GraphicsLayoutWidget.resizeEvent(self, ev)

        # Update widget height
        if hasattr(self, 'ci'):
            self.ci.layout.setRowMaximumHeight(1, self.height() // 6)

    def add_roi(self):
        self.line_coordinates = []


class InferenceBox:


    def __init__(self, line_coordinates):

        roi_pen = pg.mkPen(color='red', width=4)
        handle_pen = pg.mkPen(color='blue')
        hover_pen = pg.mkPen(color='green')


        self.max_dimensions = vxattribute.read_attribute('frame_shape')[2][0][:2]

        self.rect_horizontal_size = np.linalg.norm(line_coordinates[0] - line_coordinates[1])
        # Create rect
        self.rect = pg.RectROI(pos=(line_coordinates[0,0],line_coordinates[0,1] - 0.5 *self.rect_horizontal_size),
                                    size=[self.rect_horizontal_size,  self.rect_horizontal_size],movable=True,
                               centered=True,pen=roi_pen, handlePen=handle_pen, handleHoverPen=hover_pen)

        self.update_coordinates_routine()
        self.rect.sigRegionChangeFinished.connect(self.update_coordinates_routine)



    def update_coordinates_routine(self):
        upper_left = self.rect.pos()
        rect_size =  self.rect.size()



        # dlc live can read cropping parameters as [x1,x2,y1,y2]
        coordinates = [
                       upper_left.x(),
                       upper_left.x() + rect_size.x(),
                        upper_left.y(),
                        upper_left.y() + rect_size.y()
        ]

        for idx in range(len(coordinates)):
            max_image_dim = self.max_dimensions[idx // 2]

            if coordinates[idx] < 0:
                coordinates[idx] = 0

            elif coordinates[idx] > max_image_dim - 1:
                coordinates[idx] = int(max_image_dim - 1)
            else:
                coordinates[idx] = int(coordinates[idx])



        # make sure coordinates are positive ints within the frame dimensions


        # store coordinates for use in routine
        vxattribute.write_attribute(ZFTailTracking.inferece_box_coordinates_name,
                                    coordinates
                                    )



    def update_size(self, *args, **kwargs):
        # Just call the line segment update, it sets everything including rect size
        pass





class ZFTailTracking(vxroutine.CameraRoutine):
    """
    """

    # Set required device
    camera_device_id = 'fish_embedded'

    frame_name = 'tail_tracking_frame'

    # the coordinates of the inference box
    inferece_box = None
    inferece_box_coordinates_name = "inference_box_coordinates"

    # Internal
    reference_frame: Union[None, np.ndarray] = None
    reference_points = []


    # path to the directory in which the .pb model is stored
    model_path = '../../Desktop/tail_tracking/exported-models/resnet50_1000us'

    # store all labels of the tail as tail pose in one numpy array
    tail_pose = np.zeros((9,3))

    #tt the cutoff value (float between 0 and 1) of confidence where to plot the labels
    pcutoff = 0.95

    # which point to visualize default is point 9
    visualize_this_point_idx = 8

    # the rate at which to downsample for inference
    downsample = 1.0

    # how to color the points
    colormap = [(r,g,b) for r in [0,150,255] for g in [0,150,255] for b in [0,150,255]]
    colormap[visualize_this_point_idx] = [255,0,0]

    def __init__(self, *args, **kwargs):
        vxroutine.CameraRoutine.__init__(self, *args, **kwargs)

        #tt store dlc model
        self.dlc_object = DLCLive(self.model_path)


    def require(self):
        # Add camera device to deps
        vxdependency.require_camera_device(self.camera_device_id)

    def setup(self):

        # Get camera specs
        camera = vxcamera.get_camera_by_id(self.camera_device_id)
        if camera is None:
            log.error(f'Camera {self.camera_device_id} unavailable for eye position tracking')
            return

        # Add frames
        vxattribute.ArrayAttribute(self.frame_name, (camera.width, camera.height, 3), vxattribute.ArrayType.uint16)

        # the array output of the dlc model
        vxattribute.ArrayAttribute('tail_pose_data', (9, 3), vxattribute.ArrayType.float64)

        # frame shape
        vxattribute.ArrayAttribute('frame_shape',(3,),vxattribute.ArrayType.uint16)

        # add inference box coordinates (upper left, lower right)
        vxattribute.ArrayAttribute(self.inferece_box_coordinates_name,(4,),vxattribute.ArrayType.uint16)

        # estimated velocity of fish
        vxattribute.ArrayAttribute('t_frame_retrival', (1,), vxattribute.ArrayType.float64)

        # for visualizing single point on the tail
        vxattribute.ArrayAttribute('point_of_interest_yval', (1,), vxattribute.ArrayType.float64)

        # to display how many frames are sampled per second (important to prevent aliasing in motion model)
        vxattribute.ArrayAttribute('fps', (1,), vxattribute.ArrayType.float64)


    def save_this_variable(self,variable_name):
        vxattribute.write_to_file(self, variable_name)

    def add_to_plotter(self):
        """Add variables to plotter """

        vxui.register_with_plotter('point_of_interest_yval', name=f'position of point idx: {self.visualize_this_point_idx}', axis='y position',units='pixels')
        vxui.register_with_plotter('fps', name=f'frames per second', axis='fps',units = 'counts')


    def initialize(self):

        # read a first frame shape
        frame_shape = vxattribute.get_attribute(self.frame_name).shape

        # record time of frame
        self.time_of_last_frame = perf_counter()

        # get the frame shape of the transpose as the model was trained on transpose images
        frame_shape_T = frame_shape[1], frame_shape[0], frame_shape[2]

        # store the frame shape
        vxattribute.write_attribute('frame_shape',frame_shape)

        _ = self.dlc_object.init_inference(np.zeros(frame_shape_T, dtype=np.uint8))

        if not ZFTailTracking.instance().dlc_object.is_initialized:
            log.error('Failed to initialize DLC model')

        self.add_to_plotter()
        self.save_this_variable('tail_pose_data')

        # for checking speed
        self.fps_calculator_frameNr = 0
        self.fps_calculator_time = None



    def apply_image_correction(self, frame: np.ndarray) -> np.ndarray:
        return np.clip(self.contrast * frame + self.brightness, 0, 255).astype(np.uint8)

    def apply_range(self, frame: np.ndarray) -> np.ndarray:
        return np.clip(frame, self.brightness_min, self.brightness_max).astype(np.uint8)

    def fps_calculator(self,t,every_x_frames = 100):

        self.fps_calculator_frameNr += 1

        if self.fps_calculator_frameNr % every_x_frames == 0:
            if self.fps_calculator_time is not None:
                vxattribute.write_attribute('fps',every_x_frames /(t -  self.fps_calculator_time))

            self.fps_calculator_frameNr = 0
            self.fps_calculator_time = t



    def process_frame(self,frame_arr):
        # this is because we turned camera
        frame_out = frame_arr.T

        # Reduce to mono
        if frame_arr.ndim > 2:
            frame_out = frame_arr[:, :, 0]
        return (frame_out)

    def update_inference_box(self,new_inf_box_coord):

        if new_inf_box_coord != [0, 0, 0, 0] and self.dlc_object.cropping != new_inf_box_coord:
            self.dlc_object.cropping = new_inf_box_coord
            log.info(f"Updating model cropping parameters to {self.dlc_object.cropping}")


    def model_inference(self,frame_arr):

        tail_pose = self.dlc_object.get_pose(
            frame_arr.T # when non horizontal fish is passed you get lower quality labels
        )

        return tail_pose



    def add_pose_to_frame(self,frame_arr,tail_pose):
        frame_out = np.repeat(frame_arr[:, :, None], 3, axis=-1)
        for idx, point in enumerate(tail_pose):
            y, x, confidence = point
            color = self.colormap[idx]
            if confidence > self.pcutoff:
                frame_out = cv2.circle(frame_out, (int(x), int(y)), 10, color, thickness=-1)
        return frame_out





    def main(self, **frames):


        # Read frame
        frame = frames.get(self.camera_device_id)
        t = perf_counter()

        self.fps_calculator(t, every_x_frames = 100)


        # Check if frame was returned
        if frame is None:
            return


        # process frame
        frame = self.process_frame(frame)

        # update the inference box
        self.update_inference_box(list(vxattribute.read_attribute(self.inferece_box_coordinates_name)[2][0]))

        # update resize parameter
        self.dlc_object.resize = np.maximum(self.downsample,0.01)

        # get the model estimates of the tail pose
        tail_pose = self.model_inference(frame)

        # plot the pose on the frame to see
        frame = self.add_pose_to_frame(frame,tail_pose)

        # write the frame again so that the UI can take show it
        vxattribute.write_attribute(self.frame_name, frame)
        vxattribute.write_attribute('point_of_interest_yval',tail_pose[self.visualize_this_point_idx,1])
        vxattribute.write_attribute('t_frame_retrival',t)
        vxattribute.write_attribute('tail_pose_data', tail_pose)


