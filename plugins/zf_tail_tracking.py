"""Eye tracking for zebrafish larvae - routine and user interface
"""
from __future__ import annotations

from typing import Dict, Hashable, List, Tuple, Union

import cv2
import numpy as np
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.spatial import distance

from scipy import optimize


import vxpy.core.attribute as vxattribute
import vxpy.core.devices.camera as vxcamera
import vxpy.core.dependency as vxdependency
import vxpy.core.io as vxio
import vxpy.core.ipc as vxipc
import vxpy.core.logger as vxlogger
import vxpy.core.routine as vxroutine
import vxpy.core.ui as vxui
from vxpy.utils.widgets import Checkbox, DoubleSliderWidget, IntSliderWidget, UniformWidth
from dlclive import DLCLive, Processor #tt


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

        # Image processing
        self.img_proc = QtWidgets.QGroupBox('Image processing')
        self.ctrl_panel.layout().addWidget(self.img_proc)
        self.img_proc.setLayout(QtWidgets.QVBoxLayout())
        self.use_img_corr = Checkbox(self, 'Use image correction',
                                     default=ZFTailTracking.instance().use_image_correction)
        self.use_img_corr.connect_callback(self.update_use_img_corr)
        self.uniform_label_width.add_widget(self.use_img_corr.label)
        self.img_proc.layout().addWidget(self.use_img_corr)
        self.img_contrast = DoubleSliderWidget(self.ctrl_panel, 'Contrast', default=ZFTailTracking.instance().contrast,
                                               limits=(0, 3), step_size=0.01)
        self.img_contrast.connect_callback(self.update_contrast)
        self.uniform_label_width.add_widget(self.img_contrast.label)
        self.img_proc.layout().addWidget(self.img_contrast)
        self.img_brightness = IntSliderWidget(self.ctrl_panel, 'Brightness',
                                              default=ZFTailTracking.instance().brightness, limits=(-200, 200))
        self.img_brightness.connect_callback(self.update_brightness)
        self.uniform_label_width.add_widget(self.img_brightness.label)
        self.img_proc.layout().addWidget(self.img_brightness)
        self.use_motion_corr = Checkbox(self, 'Use motion correction',
                                        default=ZFTailTracking.instance().use_motion_correction)
        self.use_motion_corr.connect_callback(self.update_use_motion_corr)
        self.uniform_label_width.add_widget(self.use_motion_corr.label)
        self.img_proc.layout().addWidget(self.use_motion_corr)

        # Eye position detection
        self.eye_detect = QtWidgets.QGroupBox('Tail Detection Parameters')
        self.eye_detect.setLayout(QtWidgets.QVBoxLayout())
        self.ctrl_panel.layout().addWidget(self.eye_detect)

        # Flip direction option
        self.flip_direction = Checkbox(self, 'Flip directions', ZFTailTracking.instance().flip_direction)
        self.flip_direction.connect_callback(self.update_flip_direction)
        self.eye_detect.layout().addWidget(self.flip_direction)
        self.uniform_label_width.add_widget(self.flip_direction.label)

        # Image threshold
        self.particle_threshold = DoubleSliderWidget(self, 'P-cutoff',
                                                  limits=(0.5, 1),
                                                  default=ZFTailTracking.instance().pcutoff,
                                                  step_size=0.001)
        self.particle_threshold.connect_callback(self.update_particle_threshold)
        self.eye_detect.layout().addWidget(self.particle_threshold)
        self.uniform_label_width.add_widget(self.particle_threshold.label)

        # Particle size
        self.particle_minsize = IntSliderWidget(self, 'Min. particle size',
                                                limits=(1, 1000),
                                                default=ZFTailTracking.instance().min_particle_size,
                                                step_size=1)
        self.particle_minsize.connect_callback(self.update_particle_minsize)
        self.eye_detect.layout().addWidget(self.particle_minsize)
        self.uniform_label_width.add_widget(self.particle_minsize.label)

        # Saccade detection
        # self.saccade_detect = QtWidgets.QGroupBox('Saccade detection')
        #self.saccade_detect.setLayout(QtWidgets.QHBoxLayout())
        #self.ctrl_panel.layout().addWidget(self.saccade_detect)
       # self.sacc_threshold = IntSliderWidget(self, 'Sacc. threshold [deg/s]',
                                              #limits=(1, 10000),
                                           #   default=ZFTailTracking.instance().saccade_threshold,
                                           #   step_size=1)
        #self.sacc_threshold.connect_callback(self.update_sacc_threshold)
        #self.saccade_detect.layout().addWidget(self.sacc_threshold)
        #self.uniform_label_width.add_widget(self.sacc_threshold.label)

        self.hist_plot = HistogramPlot(parent=self)
        self.ctrl_panel.layout().addWidget(self.hist_plot)

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
    def update_use_img_corr(value):
        ZFTailTracking.instance().use_image_correction = bool(value)

    @staticmethod
    def update_contrast(value):
        ZFTailTracking.instance().contrast = value

    @staticmethod
    def update_brightness(value):
        ZFTailTracking.instance().brightness = value

    @staticmethod
    def update_use_motion_corr(value):
        ZFTailTracking.instance().use_motion_correction = value

    @staticmethod
    def update_flip_direction(state):
        ZFTailTracking.instance().flip_direction = bool(state)

    @staticmethod
    def update_particle_threshold(pcutoff_value):
        ZFTailTracking.instance().pcutoff = pcutoff_value

    @staticmethod
    def update_particle_minsize(minsize):
        ZFTailTracking.instance().min_particle_size = minsize

    #@staticmethod
    #def update_sacc_threshold(sacc_thresh):
    #    ZFTailTracking.instance().saccade_threshold = sacc_thresh

    def update_frame(self):


        idx, time, frame = vxattribute.read_attribute(ZFTailTracking.frame_name)
        frame = frame[0]

        if frame is None:
            return

        # Update image
        self.frame_plot.image_item.setImage(frame)

        # Update pixel histogram
        self.hist_plot.update_histogram(self.frame_plot.image_item)


class HistogramPlot(QtWidgets.QGroupBox):

    def __init__(self, **kwargs):
        QtWidgets.QGroupBox.__init__(self, 'Histogram', **kwargs)
        self.setLayout(QtWidgets.QHBoxLayout())

        self.histogram = pg.HistogramLUTWidget(orientation='horizontal')
        self.histogram.item.setHistogramRange(0, 255)
        self.histogram.item.setLevels(0, 255)
        self.layout().addWidget(self.histogram)

        self.histogram.item.sigLevelsChanged.connect(self.update_levels)

    def update_histogram(self, image_item: pg.ImageItem):

        bins, counts = image_item.getHistogram()
        logcounts = counts.astype(np.float64)
        logcounts[counts == 0] = 0.1
        logcounts = np.log10(logcounts)
        logcounts[np.isclose(logcounts, -1)] = 0
        self.histogram.item.plot.setData(bins, logcounts)

    def update_levels(self, item: pg.HistogramLUTItem):
        lower, upper = item.getLevels()

        # Correct out of bounds values
        if lower < 0:
            lower = 0
            item.setLevels(lower, upper)
        if upper > 255:
            upper = 255
            item.setLevels(lower, upper)

        ZFTailTracking.instance().brightness_min = int(lower)
        ZFTailTracking.instance().brightness_max = int(upper)


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

        # Set up scatter item for tracking motion correction features
        self.scatter_item = pg.ScatterPlotItem(pen=pg.mkPen(color='blue'), brush=None)
        self.image_plot.addItem(self.scatter_item)

        # Make subplots update with whole camera frame
        #self.image_item.sigImageChanged.connect(self.update_subplots)

        # Bind mouse click event for drawing of lines
        self.image_plot.scene().sigMouseClicked.connect(self.mouse_clicked)

    def mouse_clicked(self, ev):
        pos = self.image_plot.vb.mapSceneToView(ev.scenePos())
        print(pos)
        # First click: start new line
        if self.line_coordinates is not None and len(self.line_coordinates) == 0:
            self.line_coordinates = [[pos.x(), pos.y()]]

        # Second click: end line and create rectangular ROI + subplot
        elif self.line_coordinates is not None and len(self.line_coordinates) == 1:
            # Set second point of line
            self.line_coordinates.append([pos.x(), pos.y()])
            #print("Line corrdinates", self.line_coordinates)

            # Create inference box
            self.inferece_box = InferenceBox(np.array(self.line_coordinates))
            ZFTailTracking.instance().inferece_box = self.inferece_box

            # draw inference box rectangle
            self.image_plot.vb.addItem(self.inferece_box.rect)

    def resizeEvent(self, ev):
        pg.GraphicsLayoutWidget.resizeEvent(self, ev)

       # print("RESIZE EVENT")

        # Update widget height
        if hasattr(self, 'ci'):
            self.ci.layout.setRowMaximumHeight(1, self.height() // 6)

    def add_roi(self):
      #  print("reached add roi")
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
                print(f'Inference Box value {coordinates[idx]} not allowed. Setting to 0.')
                coordinates[idx] = 0

            elif coordinates[idx] > max_image_dim - 1:
                print(f'Inference Box value {coordinates[idx]} over allowed max of {max_image_dim - 1}. Setting to max.')
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







class FreqAmpEstimator():
    """instances of this class calculate the frequency * amplitude product of a discrete periodic signal.
    Frequency is calculated by finding maxima/minima and calculating the time passed between them."""

    # use an average of the last n data points
    average_last_n_freq_amps = 1

    # how many pixels in a meter
    n_pixels_in_meter = 600 * (1 / 0.0042)

    def __init__(self, tolerance, min_freq=1):
        # for storing data
        self.y_hist = []
        self.t_hist = []

        # store extrema y and t values
        self.when_last_extrema = []
        self.y_last_extrema = []

        # correct for noise
        self.tolerance = tolerance
        self.freq_estimate_hist_unaveraged = []
        self.amp_hist_unaveraged = []

        # at what freq do we consider tail to have stopped moving
        self.min_freq = min_freq

        #
        self.current_amp_freq_prod = 0

    def triplet_has_extremum(self):

        # make sure we have a triplet of data
        assert len(self.y_hist) == 3 and len(self.t_hist) == 3

        found_extremum = False

        # look for maximum
        if self.y_hist[1] - self.y_hist[0] > self.tolerance and self.y_hist[1] - self.y_hist[2] > self.tolerance:
            found_extremum = True

        if self.y_hist[0] - self.y_hist[1] > self.tolerance and self.y_hist[2] - self.y_hist[1] > self.tolerance:
            found_extremum = True

        return (found_extremum)

    def store_new_data(self, y, t):

        # add points to history
        self.y_hist.append(y)
        self.t_hist.append(t)

        # drop values if we do not need them
        if len(self.y_hist) > 3:
            self.y_hist.pop(0)
            self.t_hist.pop(0)

    def estimate_freq(self):

        # calculate frequency if we have more than 1 stored extremum and store it
        estimated_freq = 1 / (2 * (self.when_last_extrema[-1] - self.when_last_extrema[-2]))
        self.freq_estimate_hist_unaveraged.append(estimated_freq)

        print('current freq estimate', estimated_freq)

        # delete unnecessary freq histories
        if len(self.freq_estimate_hist_unaveraged) > self.average_last_n_freq_amps:
            self.freq_estimate_hist_unaveraged.pop(0)
            self.when_last_extrema.pop(0)

    def get_amplitude(self):

        # calulate amplitude
        diff_pix = np.abs(self.y_last_extrema[-2] - self.y_last_extrema[-1])
        diff_meters = diff_pix / self.n_pixels_in_meter
        self.amp_hist_unaveraged.append(diff_meters)

        print('Current amplitude in meters', diff_meters)

        # delete unnecessary history
        if len(self.amp_hist_unaveraged) > self.average_last_n_freq_amps:
            self.amp_hist_unaveraged.pop(0)
            self.y_last_extrema.pop(0)

    def check_if_tail_stagnated(self):

        # say it stagnated if there is no part of the triplet that moved
        if np.abs(self.y_hist[0] - self.y_hist[1]) < self.tolerance and np.abs(
                self.y_hist[1] - self.y_hist[2]) < self.tolerance:
            # update freq* amp to 0
            self.current_amp_freq_prod = 0

            # delete history of extrema
            self.when_last_extrema = []
            self.y_last_extrema = []
            self.freq_estimate_hist_unaveraged = []
            self.amp_hist_unaveraged = []
            return (True)

        # If too much time has passed since last extremum consider the tail as non moving
        if self.when_last_extrema:
            if 2 * (self.t_hist[-1] - self.when_last_extrema[-1]) > 1 / self.min_freq:
                # update freq* amp to 0
                self.current_amp_freq_prod = 0

                # delete history of extrema
                self.when_last_extrema = []
                self.y_last_extrema = []
                self.freq_estimate_hist_unaveraged = []
                self.amp_hist_unaveraged = []
                return (True)

    def main(self, y, t):
        """Function returns a product of frequency estimated and amplitude """

        # first store new data
        self.store_new_data(y, t)

        # only continue if we have sufficient date
        if len(self.y_hist) < 3:
            # print('not enough data')
            return (None)

        # return frequency 0 if the tail does not move anymore
        if self.check_if_tail_stagnated():
            # print('too much time passed from prev extremum')
            return (self.current_amp_freq_prod)

        # check if there was an extremum and dont estimate the frequency otherwise
        if not self.triplet_has_extremum():
            # the current estimate of amp*frequeny has not changed since the tail is still moving
            return (self.current_amp_freq_prod)

        # store time and value of extremum (middle data point)
        self.when_last_extrema.append(self.t_hist[-2])
        self.y_last_extrema.append(self.y_hist[-2])

        # print('updated when last extrmum', self.when_last_extrema)

        # check if we only have current extremum
        if len(self.when_last_extrema) < 2:
            # print('only one extremum returning current freq estimate')

            # ???
            return (self.current_amp_freq_prod)

        # if we have more than 1 extremum update the freq amp product and return it
        self.estimate_freq()
        self.get_amplitude()
        self.current_amp_freq_prod = np.mean(self.freq_estimate_hist_unaveraged) * np.mean(self.amp_hist_unaveraged)

        # print('estimated frequency')
        return (self.current_amp_freq_prod)


class VelocityEstimator():
    current_velocity_estimate = None
    unit = 'm/s'

    def __init__(self, freq_amp_estimator):
        self.freq_amp_estimator = freq_amp_estimator

    def turn_fA_to_vel(self, freq_amp):

        print('Getting freq amp value of ', freq_amp)
        if freq_amp != 0:

            # relationship is best based on  Leeuwen et al 2015
            p = 996.232
            mu = 0.883e-3
            l = 4.2e-3
            a = 41.29 * (p * l / mu) ** (-0.741)
            b = 0.525

            def f(v):
                return (a * v ** 0.259 + b * v - freq_amp)

            def fprime(v):
                return (0.259 * a * v ** (-0.741) + b)

            sol = optimize.root_scalar(f, bracket=[0, 0.5], method='brentq')

            print('found root at ', sol.root)

            return (sol.root)

        else:
            return (0)

    def main(self, y, t):
        freq_amp_estimate = self.freq_amp_estimator.main(y, t)

        if freq_amp_estimate is not None:
            self.current_velocity_estimate = self.turn_fA_to_vel(freq_amp_estimate)
            return (self.current_velocity_estimate)
        else:
            return (None)


class ZFTailTracking(vxroutine.CameraRoutine):
    """Routine that detects an arbitrary number of zebrafish eye pairs in a
    monochrome input frame

    Args:
        roi_maxnum (int): maximum number of eye pairs to be detected
        thresh (int): initial binary threshold to use for segmentation
        min_size (int): initial minimal particle size. Anything below this size will be discarded
        saccade_threshold (int): initial saccade velocity threshold for binary saccade trigger
    """
    # Image corrections
    use_image_correction = False
    contrast = 1.0
    brightness = 0
    brightness_min = 0
    brightness_max = 255
    use_motion_correction = False
    # Eye detection
    roi_maxnum = 5
    flip_direction = False
    binary_threshold = 60
    min_particle_size = 20

    # Set required device
    camera_device_id = 'fish_embedded'

    frame_name = 'tail_tracking_frame'

    # the coordinates of the inference box
    inferece_box = None
    inferece_box_coordinates_name = "inference_box_coordinates"

    # Internal
    reference_frame: Union[None, np.ndarray] = None
    reference_points = []



    #tt path to the directory in which the .pb model is stored
    model_path = '../tail_tracking/exported-models/resnet50_1000us'

    #tt store all labels of the tail as tail pose in one numpy array
    tail_pose = np.zeros((9,3))

    #tt the cutoff value (float between 0 and 1) of confidence where to plot the labels
    pcutoff = 0.95

    # which point to visualize default is point 9
    visualize_this_point_idx = 8

    def __init__(self, *args, **kwargs):
        vxroutine.CameraRoutine.__init__(self, *args, **kwargs)

        #tt store dlc model
        self.dlc_object = DLCLive(self.model_path)

        # get a velocity estimator
        self.freq_amp_estimator = FreqAmpEstimator(tolerance=10)

        # get a velocity estimator
        self.velocity_estimator = VelocityEstimator(self.freq_amp_estimator)

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
        vxattribute.ArrayAttribute('tail_pose_data', (9, 3), vxattribute.ArrayType.float64)

        # frame shape
        vxattribute.ArrayAttribute('frame_shape',(3,),vxattribute.ArrayType.uint16)

        # add inference box coordinates (upper left, lower right)
        vxattribute.ArrayAttribute(self.inferece_box_coordinates_name,(4,),vxattribute.ArrayType.uint16)

        # estimated velocity of fish
        vxattribute.ArrayAttribute('speed', (1,), vxattribute.ArrayType.float64)
        vxattribute.ArrayAttribute('swim_angle', (1,), vxattribute.ArrayType.float64)

        # for visualizing single point on the tail
        vxattribute.ArrayAttribute('point_of_interest', (1,), vxattribute.ArrayType.float64)




    def add_to_plotter(self):
        """Add this variable to plotter """

        vxui.register_with_plotter('speed', name='speed', axis='speed',units='mm/s')
        vxui.register_with_plotter('swim_angle', name='angle', axis='angle',units='deg')
        vxui.register_with_plotter('point_of_interest', name=f'position of point idx: {self.visualize_this_point_idx}', axis='y position',units='pixels')


    def initialize(self):

        # read a first frame shape
        frame_shape = vxattribute.get_attribute(self.frame_name).shape

        # get the frame shape of the transpose as the model was trained on transpose images
        frame_shape_T = frame_shape[1], frame_shape[0], frame_shape[2]


        # store the frame shape
        vxattribute.write_attribute('frame_shape',frame_shape)

        _ = self.dlc_object.init_inference(np.zeros(frame_shape_T, dtype=np.uint8))

        if not ZFTailTracking.instance().dlc_object.is_initialized:
            log.error('Failed to initialize DLC model')

        self.add_to_plotter()

    def apply_image_correction(self, frame: np.ndarray) -> np.ndarray:
        return np.clip(self.contrast * frame + self.brightness, 0, 255).astype(np.uint8)

    def apply_range(self, frame: np.ndarray) -> np.ndarray:
        return np.clip(frame, self.brightness_min, self.brightness_max).astype(np.uint8)

    def main(self, **frames):

        # Read frame
        frame = frames.get(self.camera_device_id)

        # Check if frame was returned
        if frame is None:
            return

        # this is because we turned camera
        frame = frame.T

        # Reduce to mono
        if frame.ndim > 2:
            frame = frame[:, :, 0]

        # retrieve new cropping parameters
        inf_box_coord = list(vxattribute.read_attribute(self.inferece_box_coordinates_name)[2][0])

        if inf_box_coord != [0,0,0,0] and self.dlc_object.cropping != inf_box_coord:

            self.dlc_object.cropping = inf_box_coord
            print("Updating model cropping parameters to ", self.dlc_object.cropping)

        #print('frame.T shape ', frame.T.shape)
        tail_pose = self.dlc_object.get_pose(
            frame.T # when non horizontal fish is passed you get lower quality labels
        )
        #print(tail_pose)

        vxattribute.write_attribute('tail_pose_data', tail_pose)

        frame = np.repeat(frame[:,:,None], 3, axis=-1)
        for y, x, confidence in tail_pose:
            if confidence > self.pcutoff:
                frame = cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0),thickness=-1)

        vxattribute.write_attribute(self.frame_name, frame)



        # calculate speed
        velocity_estimate = self.velocity_estimator.main(tail_pose[self.visualize_this_point_idx,0],vxipc.get_time())

        vxattribute.write_attribute('speed',velocity_estimate)
        vxattribute.write_attribute('swim_angle',np.random.randint(0,20))
        vxattribute.write_attribute('point_of_interest',tail_pose[self.visualize_this_point_idx,0])

