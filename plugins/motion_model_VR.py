
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
from abc import ABC, abstractmethod
from time import perf_counter

import vxpy.core.container as vxcontainer

log = vxlogger.getLogger(__name__)



class MotionModelUI(vxui.CameraAddonWidget):
    display_name = 'Motion Model VR'

    _vspacer = QtWidgets.QSpacerItem(1, 20,
                                     QtWidgets.QSizePolicy.Policy.Maximum,
                                     QtWidgets.QSizePolicy.Policy.MinimumExpanding)


    selected_MM_parameters = {}

    # this is to translate the ui selected points into indices of the pose array.
    ui_idx_to_pose_idx = {'-1': [i for i in range(9)],
                          '0': [0],
                          '1': [5],
                          '2': [3],
                          '3': [6],
                          '4': [2],
                          '5': [7],
                          '6': [4],
                          '7': [8],
                          '8': [1]
                          }
    # same here: translate positions on the tail to indices. We want segments so we take two adjacent indices
    ui_seg_to_pose_seg = {'0': [0, 5],
                          '1': [5, 3],
                          '2': [3, 6],
                          '3': [6, 2],
                          '4': [2, 7],
                          '5': [7, 4],
                          '6': [4, 8],
                          '7': [8, 1],
                          '-1': [0, 1]}


    def __init__(self, *args, **kwargs):
        vxui.CameraAddonWidget.__init__(self, *args, **kwargs)

        self.central_widget.setLayout(QtWidgets.QHBoxLayout())

        '''Control pannel'''
        self.ctrl_panel = QtWidgets.QWidget(self)
        self.ctrl_panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
                                      QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.ctrl_panel.setLayout(QtWidgets.QVBoxLayout())
        self.central_widget.layout().addWidget(self.ctrl_panel)

        self.uniform_label_width = UniformWidth()


        '''Frame plot to visualize selection of points for motion model '''
        self.frame_plot = FramePlot(parent=self)


        self.frame_plot.setSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                                      QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.central_widget.layout().addWidget(self.frame_plot)

        self.connect_to_timer(self.update_frame)


        '''Translational Speed Model'''
        # create region for translational model
        self.translational_model_user_interface = QtWidgets.QGroupBox('Translational Speed Model Parameters')
        self.translational_model_user_interface.setLayout(QtWidgets.QVBoxLayout())
        self.ctrl_panel.layout().addWidget(self.translational_model_user_interface)

        # estimation algorithm for translational speed
        self.translational_speed_estimator_interface = SearchableListWidget(self)
        self.translational_speed_estimator_interface.add_item('Vertical Speed')
        self.translational_speed_estimator_interface.add_item('Frequency Amplitude Product (Discrete Fourier Transform)')
        self.translational_speed_estimator_interface.itemDoubleClicked.connect(self.update_translational_speed_estimator)
        self.translational_model_user_interface.layout().addWidget(self.translational_speed_estimator_interface)

        # Points to use
        self.translational_speed_pts_used = IntSliderWidget(self, 'Points Used for Translational Speed Estimation',
                                              default=-1, limits=(-1, 8))
        self.translational_speed_pts_used.connect_callback(self.update_used_pts_translational_speed)
        self.translational_model_user_interface.layout().addWidget(self.translational_speed_pts_used)
        self.uniform_label_width.add_widget(self.translational_speed_pts_used.label)




        '''Angular Speed Model'''
        self.angular_motion_user_interface = QtWidgets.QGroupBox('Angular Speed Model Parameters')
        self.angular_motion_user_interface.setLayout(QtWidgets.QVBoxLayout())
        self.ctrl_panel.layout().addWidget(self.angular_motion_user_interface)

        # estimation algorithm for angular speed
        self.angular_speed_estimator_interface = SearchableListWidget(self)
        self.angular_speed_estimator_interface.add_item('Tail Segment Angular Speed')
        self.angular_speed_estimator_interface.itemDoubleClicked.connect(
            self.update_angular_speed_estimator)
        self.angular_motion_user_interface.layout().addWidget(self.angular_speed_estimator_interface)

        # Points to use
        self.angular_speed_segments_used = IntSliderWidget(self, 'Segment to use for Angular Speed Estimation',
                                                            default=7, limits=(-1, 7))
        self.angular_speed_segments_used.connect_callback(self.update_used_segments_angular_speed)
        self.angular_motion_user_interface.layout().addWidget(self.angular_speed_segments_used)
        self.uniform_label_width.add_widget(self.angular_speed_segments_used.label)


    def re_initialize_motion_model(self):
        MotionModelApplication.instance().initialize_motion_model = True

    def update_translational_speed_estimator(self,list_item):
       MotionModelApplication.instance().MM_parameters['translation_speed_model_type'] = list_item.text()
       self.re_initialize_motion_model()

    def update_angular_speed_estimator(self,list_item):
        MotionModelApplication.instance().MM_parameters['angular_speed_model_type'] = list_item.text()
        self.re_initialize_motion_model()

    def update_used_pts_translational_speed(self,pts):
        MotionModelApplication.instance().MM_parameters['translation_speed_model_pts_used'] = pts
        self.re_initialize_motion_model()

    def update_used_segments_angular_speed(self,seg_nr):
        MotionModelApplication.instance().MM_parameters['angular_speed_model_segment_used'] = seg_nr
        self.re_initialize_motion_model()

    def update_frame(self):

        frame_name = 'fish_embedded_frame'
        idx, time, frame = vxattribute.read_attribute(frame_name)
        frame = frame[0]

        if frame is None:
            return


        # read the pose
        tail_pose = vxattribute.read_attribute('tail_pose_data')[2][0]

        # add the tracked points selected for the translational motion model
        pts_in_array_indices = self.get_selected_points_indices_from_ui('translation_speed_model_pts_used')

        if pts_in_array_indices is not None:
            # modify the frame
            frame = self.add_points_to_frame(frame,pts_in_array_indices,[255,0,0],tail_pose)

        # add the tracked points selected for the angular speed motion model
        ang_pts_as = self.get_selected_points_indices_from_ui('angular_speed_model_segment_used')

        if ang_pts_as is not None:
            # modify the frame
            frame = self.add_points_to_frame(frame, ang_pts_as, [0, 255, 0], tail_pose)

        # Update image
        self.frame_plot.image_item.setImage(frame)


    def get_selected_points_indices_from_ui(self,key):
        # points selected from the ui
        used_pts = MotionModelApplication.instance().MM_parameters.get(key,None)

        # only add points if the motion model is initialized (no need to initialze it) and meaningful points added
        if used_pts is not None:
            if key == 'translation_speed_model_pts_used':
                # get the pose array indices from ui points. for translational model
                return self.ui_idx_to_pose_idx[str(used_pts)]

            elif key == 'angular_speed_model_segment_used':
                # for angular speed model
                return self.ui_seg_to_pose_seg[str(used_pts)]
        else:
            return None


    def add_points_to_frame(self,frame_arr,points_array_indices,color ,tail_pose ):


        if frame_arr.ndim < 3:
            frame_out = np.repeat(frame_arr[:, :, None], 3, axis=-1)
        else:
            frame_out = frame_arr

        if tail_pose is not None:
            for idx in points_array_indices:
                y, x = tail_pose[idx,:2]
                color = color
                frame_out = cv2.circle(frame_out, (int(x), int(y)), 10, color, thickness=-1)
        return frame_out




class FramePlot(pg.GraphicsLayoutWidget):


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





'''

                                                    Estimators 

'''


class Estimator(ABC):

    def __init__(self, sampling_freq, use_these_pts_idx: list = [p for p in range(9)], desired_min_f: int = 5,
                 sampling_freq_change_tol: float = 0.2, delta_pix_tail_movements: int = 30):

        # signal processing stuff
        self.sampling_freq = sampling_freq
        self.n_pts = int(np.ceil(sampling_freq / desired_min_f)) # nr of y data points to track in the history
        self.sampling_freq_change_tol = sampling_freq_change_tol

        # robustness
        self.delta_pix_tail_movements = delta_pix_tail_movements

        # data storage
        self.init_y_at = 300
        self.min_f = desired_min_f
        self.use_these_pts_idx = use_these_pts_idx
        self.t_hist = [None for i in range(100)]
        self.y_hist = np.ones((len(use_these_pts_idx), self.n_pts)) * self.init_y_at

        # points to use
        self.use_these_pts_idx = use_these_pts_idx




    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'name' not in cls.__dict__:
            raise TypeError(f"Estimator class {cls.__name__} must have 'name' to be selectable via UI.")



    def update_data(self, y, t):

        # store y,t for next data points
        self.y_hist = np.column_stack((self.y_hist[:, 1:], y))
        self.t_hist.append(t)
        self.t_hist.pop(0)

    def update_sf_related_attributes(self) -> None:

        # dont do anything if we do not have enough time samples
        if None in self.t_hist:
            return ()

        # 1) calculate the current sf from the sampled times. Take a median to make it robust against
        # time outliers such as proceses which take a long time eg model initialization
        sf_new = int(np.ceil(1/np.median(np.diff(self.t_hist))))

        assert sf_new >= 1

        # 2) check if estimated sampling frequency is significantly different
        if np.abs(sf_new - self.sampling_freq) / self.sampling_freq > self.sampling_freq_change_tol:
            log.info(
                f'Updating sampling frequency attribute of {self.name} from {self.sampling_freq} to {sf_new} because the difference is greater than {100 * self.sampling_freq_change_tol} %')

            # a) update the sampling frequency
            self.sampling_freq = sf_new

            # b) nr of points to sample to achieve desired frequency resolution
            sampled_points_new = np.maximum(int(np.ceil(sf_new / self.min_f)),3)
            pts_diff = sampled_points_new - self.n_pts
            if pts_diff == 0:
                # no need to update further
                return ()
            else:
                # reinitialize array because you can get weired results if data points are sampled at different
                # sampling frequencies
                self.y_hist = np.ones((len(self.use_these_pts_idx), sampled_points_new)) * self.init_y_at

                # update number of sampled points. prevent too small number of points for preventing weired behavior
                self.n_pts = sampled_points_new


    @abstractmethod
    def estimate(self) -> np.float64:
        pass


class VerticalSpeedEstimator(Estimator):
    """This Class calculates the vertical velocity of tail points and therefore estimates the product of frequency and
     peak-to-peak Amplitude of tail points by dividing the max vertical velocity by py/"""

    # This calue changes for different cameras
    pixels_per_meter = 170000

    name = 'Vertical Speed'

    def __init__(self, sampling_freq, use_these_pts_idx: list = [p for p in range(9)], desired_min_f: int = 5,
                 sampling_freq_change_tol: float = 0.2, delta_pix_tail_movements: int = 30):

        super().__init__(sampling_freq, use_these_pts_idx, desired_min_f,
                 sampling_freq_change_tol, delta_pix_tail_movements)

    def calculate_vertical_speed(self):
        '''
        Method takes the max speed (ms) of any tail point in the stored time period
        '''
        if None in self.t_hist:
            return (0)

        delta_t = np.diff(self.t_hist[-2:])
        max_speeds = np.max(np.diff(self.y_hist, axis=1)) / (self.pixels_per_meter * delta_t)

        # divide by pi to estimate the peak to peak amplitude frequency product
        return (np.max(max_speeds) / np.pi)

    def tail_is_moving(self) -> np.ndarray:
        '''Judges the the tail points as moving if there was any movement in the y direction of any point in the tracked time window '''
        return (np.diff(self.y_hist, axis=1) > self.delta_pix_tail_movements).any()



    def estimate(self, pose_array, t):

        # select the y_array from the pose_array, only y values
        y_array = pose_array[:,1]

        # take the y values of the points of interest from the array and add them to data store
        self.update_data(y_array[self.use_these_pts_idx], t)

        # check if there was change in sampling frequency
        self.update_sf_related_attributes()

        vertical_speed = self.calculate_vertical_speed() if self.tail_is_moving() else 0


        #print('called estimate on vertical speed instance')
        #print(type(vertical_speed), vertical_speed)

        return np.float64(vertical_speed)


class DFTEstimator(Estimator):
    '''
    The number of sampled points equals the number of points in the frequency space. The max frequency is nyquist/2 so sampling frequency / 2.
    So sampling frequency / number of points is the frequency resolution. if the minimum frequency resolution is min_f then nPts = sf/min_f
    '''

    name = 'Frequency Amplitude Product (Discrete Fourier Transform)'

    pixels_per_meter = 170000

    def __init__(self, sampling_freq, use_these_pts_idx: list = [p for p in range(9)], desired_min_f: int = 5,
                 sampling_freq_change_tol: float = 0.2, delta_pix_tail_movements: int = 30):

        super().__init__(sampling_freq, use_these_pts_idx, desired_min_f,
                 sampling_freq_change_tol, delta_pix_tail_movements)


    def tail_points_moving(self) -> np.ndarray:
        '''Judges the the tail points as moving if there was any movement in the y direction of any point in the tracked time window '''
        return (np.diff(self.y_hist, axis=1) > self.delta_pix_tail_movements).any(axis=1)

    def calculate_frequency(self, tail_points_active: np.ndarray) -> np.float64:

        y_arr_centered = self.y_hist[tail_points_active, :] - np.mean(self.y_hist[tail_points_active, :],
                                                                      axis=1).reshape(-1, 1)

        fft_result = np.fft.fft(y_arr_centered, axis=1)

        # get mag of complex number
        fft_abs = np.abs(fft_result)[:, :int(self.n_pts / 2)]

        # compate the magnitudes to the median accross frequencies for each dimension. this is the gain
        median_mag = np.median(fft_abs, axis=1)

        # normalize while adding a small term to avoid zero division
        fft_abs_gain = fft_abs / (median_mag[:, np.newaxis] + 1e-6)

        # turn to meaningful non-normalized frequencies
        # note the frequency resolition will be roughtly sampling freq / n points
        fft_freq = np.fft.fftfreq(self.n_pts, 1 / self.sampling_freq)[:int(self.n_pts / 2)]
        max_freqs = fft_freq[np.argmax(fft_abs_gain, axis=1)]

        # print(f'Time {self.t_hist[-1]}')
        # for i in range(self.n_pose_points):
        #   print(f'\tgain , freq in dim {i}', [max_mags[i],max_freqs[i]] )
        #  print(f'\t\t all freq gain {i}', [fft_abs_gain[i]])

        average_freq = np.mean(max_freqs)

        return (average_freq)

    def get_amplitude(self):

        return (np.max(np.max(self.y_hist, axis=1) - np.min(self.y_hist, axis=1)) / self.pixels_per_meter)

    def estimate(self, pose_array: np.ndarray, t):
        '''Main function to estimate fA'''

        # take only y data from pose array
        y_array = pose_array[:,1]

        # take the y values of the points of interest from the array and add them to data store
        self.update_data(y_array[self.use_these_pts_idx], t)

        # check if there was change in sampling frequency
        self.update_sf_related_attributes()

        # print('DEBUG: self.sampling_freq estimate of DFT',self.sampling_freq)

        # bool array stating which tail points are active
        tail_point_moving = self.tail_points_moving()

        # get averaged frequency estimate of points
        freq_est = self.calculate_frequency(tail_point_moving) if tail_point_moving.any() else 0
        self.current_freq_average = freq_est

        # multiply with amplitude
        amp_est = self.get_amplitude()
        self.current_fA = amp_est * freq_est

        return np.float64(self.current_fA)


class AngularSpeedTailSegmentEstimator(Estimator):
    '''This class estimates the fictive angular velocity (unit is rad/s) of the fish by assuming the change in angle of fish heading
    is proportional to the change in angle of the tail. Zero angles are in line with the
    x axis and positive is CCW.
    '''

    name = 'Tail Segment Angular Speed'

    def __init__(self, use_these_pts_idx=[0, 8], exp_decay_factor=1,alpha_lower_bound = 0.05,movement_thresh = 1,scaling_factor = 1):

        # list of indices to use
        if len(use_these_pts_idx) == 2:
            self.use_these_pts_idx = use_these_pts_idx
        else:
            raise ValueError(f'Passed list of length unequal 2')

        # angle the tail made  with the horizontal axis in last frame
        self.tail_segment_angle_hist = [None for i in range(10)]

        # store angular velocities here
        self.angular_fish_speed = 0

        # previous time point
        self.t_hist = [None, None]

        # for exponential time decay
        self.alpha = exp_decay_factor
        self.alpha_t = 1
        self.alpha_lower_bound = alpha_lower_bound

        # min absolute differenence between poses at which to consider tail moving
        self.movement_thresh = movement_thresh

        self.scaling_factor = scaling_factor
    @staticmethod
    def get_angle_between_points(a, b):
        ''' takes two points and returns the angle between their connecting line and the horizontal line of the image.
        delta_t is the difference in time between the succesive frames'''

        delta_x = b[0] - a[0]
        delta_y = b[1] - a[1]

        return (np.arctan2(delta_y, delta_x))

    def get_angular_tail_speed(self):

        angular_tail_segment_speed = np.diff(self.tail_segment_angle_hist[-2:]) / np.diff(self.t_hist)
        return angular_tail_segment_speed

    def get_angular_fish_speed(self, angular_tail_segment_velocity):
        ''' Transforms the angular vel of the tail to angular vel of the fish (pi - tail angle). Since the y axis if
        flipped in numpy the negation part is canceled. Adding pi is not necessry because the way the fish lays in the
        frame is facing negative x values. To fictively make the fish swim in positive x directions we would have to add
        pi to the angle. We are filming from below so flip the sign.
        '''
        return (-angular_tail_segment_velocity)

    def update_data(self, current_angle, t):
        self.tail_segment_angle_hist.append(current_angle)
        self.tail_segment_angle_hist.pop(0)
        self.t_hist.append(t)
        self.t_hist.pop(0)

    def tail_is_moving(self):
        return (True if (np.diff(self.tail_segment_angle_hist) > self.movement_thresh).any() else False)

    def estimate(self, pose_array, t):

        # select points of segment
        start_point = pose_array[self.use_these_pts_idx[0], :2]
        stop_point = pose_array[self.use_these_pts_idx[1], :2]

        # calculate the angle between points
        current_tail_angle = self.get_angle_between_points(start_point, stop_point)

        # Store results
        self.update_data(current_tail_angle, t)

        if not None in self.t_hist + self.tail_segment_angle_hist:

            # check if tail is moving
            tail_moves = self.tail_is_moving()

            # calculate angular speed of tail segment
            angular_speed = self.get_angular_tail_speed()

            # apply exponential decay
            angular_speed *= (self.alpha_lower_bound + (1 - self.alpha_lower_bound) * self.alpha_t)
            # print('\nang speed factor', self.alpha_lower_bound + (1 - self.alpha_lower_bound) * self.alpha_t)
            # print('\tang speed ',angular_speed)

           # update decay factor for next iteration onbly if the tail is moving
            if tail_moves:
                self.alpha_t *= self.alpha
            else:
                self.alpha_t = 1



        else:
            tail_moves = False
            angular_speed = 0

            # reset exponential decay
            self.alpha_t = 1

        # translate to fish angular speed in its coordinate system
        self.angular_fish_speed = self.get_angular_fish_speed(angular_speed)
        return self.scaling_factor * np.float64(self.angular_fish_speed)




class BendingAngleAngularSpeedEstimator():
    '''This class estimates the angular speed of fish from the maximum bending angle between adjacent tail segments in a turn.
    For this it keeps a moving max of the bending angles. Inspired by Thandiakal and Laudar 2020'''
    name = 'Max Bending Angle'



    def __init__(self,
                 typical_turn_duration=0.2,
                 sampling_freq=120,
                 use_these_pts_idx: list = [7, 4, 8],
                 scaling_factor=1
                 ):

        self.t_hist = [None, None]
        self.bending_angle_hist = [None, None]

        self.n_pts = 2

        assert 0 not in use_these_pts_idx  # the
        self.use_these_pts_idx = use_these_pts_idx

        self.scaling_factor = scaling_factor

    def calculate_bending_angles(self, arr):
        '''Calculate the bendin angle between points stacked in rows of the array arr'''
        assert arr.shape[0] >= 3

        # get angle between adjacent tail segments
        bending_angles = []

        for i in range(1, len(self.use_these_pts_idx) - 1):
            idx_center_point = self.use_these_pts_idx[i]
            idx_prev_point = self.use_these_pts_idx[i - 1]
            idx_next_point = self.use_these_pts_idx[i + 1]
            v1 = arr[idx_center_point] - arr[idx_prev_point]
            v2 = arr[idx_next_point] - arr[idx_center_point]
            # print(f'Got v1 {v1} and v2 {v2}')

            angle1 = np.arctan2(v1[1], v1[0])
            angle2 = np.arctan2(v2[1], v2[0])
            # print(f'angle1 {angle1} and angle2 {angle2}')

            bending_angle = angle2 - angle1
            bending_angle = (bending_angle + np.pi) % (2 * np.pi) - np.pi

            bending_angles.append(bending_angle)
        # print('got  bending angles:', bending_angles)
        return np.array(bending_angles)

    def update_data(self, most_extreme_bending_angle, t):

        self.bending_angle_hist.append(most_extreme_bending_angle)
        self.bending_angle_hist.pop(0)

        self.t_hist.append(t)
        self.t_hist.pop(0)

    def get_angular_velocity(self):
        if not None in self.t_hist:
            delta_ang = np.diff(self.bending_angle_hist)
            delta_t = np.diff(self.t_hist)
            # print(f'\nDelta t {delta_t} and delta ang {delta_ang} and vel {delta_ang/delta_t}')
            # print('calc ang vel with bending anggle hist and diff:',self.bending_angle_hist,np.diff(self.bending_angle_hist))
            return np.float64(delta_ang / delta_t)

        else:
            return (np.float64(0))

    def estimate(self, pose: np.ndarray, t):

        # print('Calling estimate of Bending Angle estimator')

        if isinstance(t, np.ndarray):
            t_float = np.float64(t)
        else:
            t_float = t

        # self.update_sf_related_attributes()

        current_bending_angles = self.calculate_bending_angles(pose[:, :2])
        most_extreme_ba = current_bending_angles[np.argmax(np.abs(current_bending_angles))]
        assert most_extreme_ba.size == 1
        # print('most ext ba', most_extreme_ba)

        self.update_data(most_extreme_ba, t_float)

        # takte the angular valocity
        ang_vel = self.get_angular_velocity()
        # print('ang vel',ang_vel)

        #
        if (np.sign(ang_vel) * np.sign(most_extreme_ba)) == 1 and (np.abs(ang_vel) > 1e-3):
            return (self.scaling_factor * ang_vel)
        else:
            return (0)


'''

                                    Translator functions from estimation to speed

'''



class fAToSpeedFunction():
    current_velocity_estimate = None
    unit = 'm/s'

    # some attributes for debugging
    current_fA = 0


    def __init__(self, use_interpolated_velocity=True,linear = False):
        self.use_interpolated_velocity = use_interpolated_velocity
        self.linear = linear

        if use_interpolated_velocity:
            self.fA_stored = np.linspace(0, 0.3, 1000)
            self.vel_stored = np.array(list(map(self.turn_fA_to_vel, self.fA_stored)))
            self.velocity_interpolator = interp1d(self.fA_stored, self.vel_stored, fill_value=0, bounds_error=False)




    def turn_fA_to_vel(self, freq_amp):
        # relationship is best based on  Leeuwen et al 2015
        p = 996.232
        mu = 0.883e-3
        l = 4.2e-3  # length of fish!
        a = 41.29 * (p * l / mu) ** (-0.741) if not self.linear else 0
        b = 0.525
        def f(v):
            return (a * v ** 0.259 + b * v - freq_amp)

        sol = optimize.root_scalar(f, bracket=[0, 0.7], method='brentq')

        return (sol.root)

    def interpolate_velocity(self, freq_amp):
        interpol = self.velocity_interpolator(freq_amp)
        return interpol

    def translate(self,freq_amp_estimate):

        # find the velocity
        if self.use_interpolated_velocity:
            current_velocity_estimate = self.interpolate_velocity(freq_amp_estimate)
        else:
            current_velocity_estimate = self.turn_fA_to_vel(freq_amp_estimate)

        return np.float64(current_velocity_estimate)






'''
                                                    MotionModel itself
'''

















'''
                                utils: Filters and Other useful tools
'''

class Filter(ABC):


    def __init__(self, order, sampling_freq = 100, cutoff_freq = 10, sampling_freq_change_tol: float = 0.2):

        # store filter characteristics
        self.sampling_freq = sampling_freq
        self.order = order
        self.cutoff_freq = cutoff_freq
        self.sampling_freq_change_tol = sampling_freq_change_tol

        # array to store some t values to adjust sampling frequency
        self.t_hist = [None for i in range(100)]

        # make that we still have to initialize the array
        self.initialized = False



    def update_data(self,t: float):
        self.t_hist.append(t)
        self.t_hist.pop(0)


    def update_sf_related_attributes(self) -> None:

        # dont do anything if we do not have enough time samples
        if None in self.t_hist:
            return ()

        # 1) calculate the current sf from the sampled times. Take a median to make it robust against
        # time outliers such as proceses which take a long time eg model initialization
        sf_new = int(np.ceil(1/np.median(np.diff(self.t_hist))))

        #print(f'DEBUG sf_new {sf_new} unrounded {np.median(np.diff(self.t_hist))},t_hist in update_sf of Estimator: {self.t_hist}',)

        assert sf_new >= 1

        # 2) check if estimated sampling frequency is significantly different
        if np.abs(sf_new - self.sampling_freq) / self.sampling_freq > self.sampling_freq_change_tol:
            log.info(
                f'Updating sampling frequency attribute of {self.name} from {self.sampling_freq} to {sf_new} because difference is greater than {100 *self.sampling_freq_change_tol} %')

            # update the sampling frequency
            self.sampling_freq = sf_new

            # reinitialize filter now with a different sampling frequency
            self.__init__(order=self.order,
                          sampling_freq=sf_new,
                          cutoff_freq= self.cutoff_freq,
                          sampling_freq_change_tol = self.sampling_freq_change_tol

                          )

    @abstractmethod
    def initialize(self,y):
        pass

    @abstractmethod
    def filter(self,y,t: float) -> float:
        pass




class ButterFilter(Filter):
    '''instnaces perform butterworth low pass filtering for signals of array type of certain shape,
     float or int for certain '''

    name = 'Butterworth Low-Pass Filter'

    # changes in sf


    def __init__(self, order, sampling_freq, cutoff_freq, sampling_freq_change_tol: float = 0.2):


        super().__init__(order=order,
                         sampling_freq=sampling_freq,
                         cutoff_freq=cutoff_freq,
                         sampling_freq_change_tol=sampling_freq_change_tol)




    def initialize(self,y):

        # make sure we are passing permissible cutoff freq. Must be lower than nyquist
        cutoff_freq_checked = np.minimum(self.cutoff_freq, np.ceil(self.sampling_freq / 2) - 1)
        # print('DEBUG initialize butter cutoff_freq_checked',cutoff_freq_checked)
        self.b, self.a = signal.butter(N=self.order,
                                       btype='low',
                                       Wn=cutoff_freq_checked,
                                       fs=self.sampling_freq
                                       )


        if isinstance(y,np.ndarray):


            self.is_array = True
            self.numel = y.size

            # this is the initial states of the butterworth filters for each dimension
            self.zi = [signal.lfilter_zi(self.b, self.a) * 0 for _ in range(self.numel)]

        elif isinstance(y, (float,int)):
            self.is_array = False
            self.numel = 1
            self.zi = [signal.lfilter_zi(self.b, self.a) * 0]

        else:
            raise TypeError(f'ButterFilter expected int, float, or array but got {type(y)}')

        self.initialized = True

    def filter(self, y,t = None):

        # store incoming t values
        self.update_data(t)

        # update the sampling frequency if necessary
        self.update_sf_related_attributes()


        if not self.initialized:
            self.initialize(y)


        y_input = np.array([y]) if not self.is_array else y


        # for 1d arrays
        if self.numel > 1:

            # preallocate an array for ys
            filtered_y = np.zeros_like(y_input)

            # loop over the entries of the array, treating them as independent
            for y_idx in range(self.numel):
                filtered_y[y_idx], self.zi[y_idx] = signal.lfilter(self.b, self.a, [y_input[y_idx]], zi=self.zi[y_idx])

        # for int float type y
        else:
            filtered_y, self.zi[0] = signal.lfilter(self.b, self.a, y_input, zi=self.zi[0])

        return filtered_y[0] if not self.is_array else filtered_y






class MovingAverageFilter(Filter):
    name = 'Moving Average Filter'

    def __init__(self, order,sampling_freq,cutoff_freq,sampling_freq_change_tol):
        super().__init__(order = order)
        self.sampling_freq = sampling_freq
        self.cutoff_freq = cutoff_freq
        self.sampling_freq_change_tol = sampling_freq_change_tol

    def initialize(self,y):
        self.y_hist = [y] * self.order

    def update_data(self,y,t):

        # store t values
        super().update_data(t)

        # store y values
        self.y_hist.append(y)
        self.y_hist.pop(y)


    def filter(self, y,t):
        self.update_data(y,t)

        return (np.sum(self.y_hist) / self.order)




class MotionModel:
    '''This is a class is a wrapper which takes different estimators and calculates motion of the fish'''

    # dicts of permissible clases of motion models
    MM_classes = {class_.name: class_ for class_ in [VerticalSpeedEstimator,
                                                     DFTEstimator,
                                                     AngularSpeedTailSegmentEstimator #,BendingAngleAngularSpeedEstimator
                                                     ]}

    # Filter classes
    Filt_classes ={class_.name: class_ for class_ in [ButterFilter]}
    Filt_classes[None] = None # in case we do not want filtering

    # Different instances of the class that turns frequecy * Amplitude to speed. One for linear relationships another
    # for the true relationship. This is suboptimal and too complicated
    fA_speed_translator_instances = {'true (non-linear)':fAToSpeedFunction(linear=False),
                                      'linear approximation':fAToSpeedFunction(linear=True)}

    # dicts that relate the indices selected in the user interface for the translational model
    # to the actual indices of the points in the
    # pose array.
    ui_idx_to_pose_idx = {'-1': [i for i in range(9)],
     '0': [0],
     '1': [5],
     '2': [3],
     '3': [6],
     '4': [2],
     '5': [7],
     '6': [4],
     '7': [8],
     '8': [1]
    }

    # same here: translate positions on the tail to indices. We want segments so we take two adjacent indices
    ui_seg_to_pose_seg = {'0': [0, 5],
                          '1': [5, 3],
                          '2': [3, 6],
                          '3': [6, 2],
                          '4': [2, 7],
                          '5': [7, 4],
                          '6': [4, 8],
                          '7': [8, 1],
                          '-1': [0, 1]}



    default_parameter_dict = {

        # translational speed
        'translation_speed_model_type': 'Frequency Amplitude Product (Discrete Fourier Transform)',
        'translation_speed_model_pts_used': -1,
        'translation_speed_pre_filter_type': ButterFilter.name,
        'translation_speed_post_filter_type': ButterFilter.name,
        'fA_to_speed_function_type': 'true (non-linear)',

        # angular speed
        'angular_speed_model_type': 'Tail Segment Angular Speed',
        'angular_speed_model_segment_used': 7,
        'angular_speed_pre_filter_type': ButterFilter.name,
        'angular_speed_post_filter_type': ButterFilter.name



    }

    def __init__(self,parameters: dict = {}):
        '''Initializes motion model instance with paramertses from a dictionary'''

        self.parameters = parameters

        '''' initialize translation speed estimator '''
        # the motion model class the user selected or the default one
        trans_MM_name = parameters.get('translation_speed_model_type', self.default_parameter_dict['translation_speed_model_type'])
        trans_MM = self.MM_classes[trans_MM_name]

        # indices to use converted from ui indices to pose array indices
        trans_idx_list = self.ui_idx_to_pose_idx[str(parameters.get('translation_speed_model_pts_used', self.default_parameter_dict['translation_speed_model_pts_used']))]

        # set the model with extracted parameters
        self.translational_speed_model = trans_MM(
            sampling_freq=120,
            use_these_pts_idx= trans_idx_list,
            sampling_freq_change_tol= 0.3
        )



        ''' initialize angular speed estimator '''
        # the motion model class the user selected or the default one
        ang_MM_name = parameters.get('angular_speed_model_type',self.default_parameter_dict['angular_speed_model_type'])
        ang_MM = self.MM_classes[ang_MM_name]

        # indices to use converted from ui indices to pose array indices
        ui_selected_points = parameters.get('angular_speed_model_segment_used',self.default_parameter_dict['angular_speed_model_segment_used'])
        ang_seg_idx_list = self.ui_seg_to_pose_seg[str(ui_selected_points)]

        # iniitialize the angular speed model: angular speed of tail segment estimator
        self.angular_speed_model = ang_MM(
            use_these_pts_idx= ang_seg_idx_list,
            exp_decay_factor=0.6,
            alpha_lower_bound=0.001,
            movement_thresh = np.pi/8,
            scaling_factor = 1

        )



        ''' intialize the filters'''
        trans_pre_filt_name = parameters.get('translation_speed_pre_filter_type',self.default_parameter_dict['translation_speed_pre_filter_type'])
        trans_post_filt_name = parameters.get('translation_speed_post_filter_type',self.default_parameter_dict['translation_speed_post_filter_type'])
        trans_pre_filt = self.Filt_classes[trans_pre_filt_name]
        trans_post_filt = self.Filt_classes[trans_post_filt_name]
        if trans_pre_filt is not None:
            self.translational_speed_pre_filter = trans_pre_filt(
                order=1,
                sampling_freq= 120,
                cutoff_freq=10,
                sampling_freq_change_tol= 0.3

            )
        else:
            self.translational_speed_pre_filter = None


        if trans_post_filt is not None:
            self.translational_speed_post_filter =trans_post_filt(
                order=1,
                sampling_freq= 120,
                cutoff_freq=10,
                sampling_freq_change_tol= 0.3
            )

        else:
            self.translational_speed_post_filter = None


        ang_pre_filt_name = parameters.get('angular_speed_pre_filter_type',
                                             self.default_parameter_dict['angular_speed_pre_filter_type'])
        ang_post_filt_name = parameters.get('angular_speed_post_filter_type',
                                              self.default_parameter_dict['angular_speed_post_filter_type'])
        ang_pre_filt = self.Filt_classes[ang_pre_filt_name]
        if ang_pre_filt is not None:
            self.angular_speed_pre_filter = ang_pre_filt(
                order=3,
                sampling_freq= 120,
                cutoff_freq=5,
                sampling_freq_change_tol=0.3
            )
        else:
            self.angular_speed_pre_filter = None


        ang_post_filt = self.Filt_classes[ang_post_filt_name]
        if ang_post_filt is not None:
            self.angular_speed_post_filter = ang_post_filt(
                order=2,
                sampling_freq= 120,
                cutoff_freq=7,
                sampling_freq_change_tol=0.3
            )
        else:
            self.angular_speed_post_filter = None

        # if TEST_ANG_MM_MODE:
        #     self.angular_speed_post_filter = None


        '''initialize the frunction from frequency amplitude to speed'''
        translator_name = parameters.get('fA_to_speed_function_type',self.default_parameter_dict['fA_to_speed_function_type'])
        self.fA_speed_translator = self.fA_speed_translator_instances[translator_name]


        # show motion model parameters in log. Comment if it clutters the log
        log.info(
            ''.join([f'Done initializing Motion Model of following:\n\tTranslational Speed Model:',
            f'\n\t\tType: {self.translational_speed_model.name}',
             f'\n\t\tIndices (pose array idx differs from UI selected): {self.translational_speed_model.use_these_pts_idx}',
             f'\n\t\tFilter before fA to speed conversion: {str(self.translational_speed_pre_filter)}',
             f'\n\t\tFilter after fA to speed conversion: {str(self.translational_speed_post_filter)}',
             f'\n\tAngular Speed Model:',
            f'\n\t\tType: {self.angular_speed_model.name}',
             f'\n\t\tIndices of segment (pose array idx differs from UI selected): {self.angular_speed_model.use_these_pts_idx}',
             f'\n\t\tFilter before fA to speed conversion: {str(self.angular_speed_pre_filter)}',
             f'\n\t\tFilter after fA to speed conversion: {str(self.angular_speed_post_filter)}',
             f'\n\tFrequency-Amplitude product to speed translator linear approximation set to: {str(self.fA_speed_translator.linear)}'])
        )


    def get_model_input_from_UI_dict(self,key_name: str, UI_params: dict, identifier_model_input_dict: dict):

        # get the identifier vaule from the UI dict or default value
        identifier_str = str(UI_params.get(key_name,self.default_parameter_dict[key_name]))

        # use that identifier value as a key in the dictionay
        model_input = identifier_model_input_dict[identifier_str]

        return model_input

    @staticmethod
    def apply_filter(filter_obj: List[Union[None, ButterFilter]] ,value,t: np.float64):
        '''Checks if the filter object exits i.e. if it should be filtered. If not it returns unlatered value'''

        return filter_obj.filter(value,t) if filter_obj is not None else value


    def main(self, tail_pose, t):
        '''This function translates tail movements to virtual movements of the fish.
        All returned angles/angular speeds are in radians (per sec). Distances/translational_speed are in meters (per sec)'''
        #print('\tcalling main on MotionModel with params', self.parameters)


        '''translational speed'''
        # get frequency peak-to-peak amplitude product
        current_fA = self.translational_speed_model.estimate(tail_pose,t)

        # lowpass before translation if specified
        current_fA = self.apply_filter(self.translational_speed_pre_filter,current_fA,t)

        # translate this to translational speed
        translational_speed = self.fA_speed_translator.translate(current_fA)

        # lowpass before translation if specified the
        # print('DEBUG trans speed into post filter',type(translational_speed))
        translational_speed = self.apply_filter(self.translational_speed_post_filter, translational_speed, t)

        '''angular speed'''
        angular_speed = self.angular_speed_model.estimate(tail_pose,t)

        # lopwass if specified
        angular_speed = self.apply_filter(self.angular_speed_post_filter,angular_speed,t)

        return translational_speed,angular_speed









class MotionModelApplication(vxroutine.CameraRoutine):

    # Set required device
    camera_device_id = 'fish_embedded'

    frame_name = 'tail_tracking_frame'

    MM_parameters: dict = {}
    initialize_motion_model = True

    output_attribute_names = ['translational_speed', 'angular_speed']

    def __init__(self, *args, **kwargs):
        vxroutine.CameraRoutine.__init__(self, *args, **kwargs)



    def setup(self):
        # Get camera specs
        camera = vxcamera.get_camera_by_id(self.camera_device_id)
        if camera is None:
            log.error(f'Camera {self.camera_device_id} unavailable for eye position tracking')
            return

        # write attributes
        vxattribute.ArrayAttribute('translational_speed', (1,), vxattribute.ArrayType.float64)
        vxattribute.ArrayAttribute('angular_speed', (1,), vxattribute.ArrayType.float64)


    def init_motion_model(self):

        # intialize
        self.motion_model = MotionModel(self.MM_parameters)

        # remember we initialized MM
        self.initialize_motion_model = False


    def initialize(self):

        vxattribute.write_to_file(self,'translational_speed')
        vxattribute.write_to_file(self,'angular_speed')

        vxui.register_with_plotter('translational_speed',
                                   name=f'translational_speed', axis='Trans. Speed',
                                   units='m/s')

        vxui.register_with_plotter('angular_speed',
                                   name=f'angular_speed', axis='Ang. Speed',
                                   units='rad/s')



    @staticmethod
    def write_data_as_attribute(trans_speed,ang_speed):
        vxattribute.write_attribute('translational_speed', trans_speed)
        vxattribute.write_attribute('angular_speed', ang_speed)

    @staticmethod
    def read_data_from_attributes(attr_str: str):
        ''' Gets data from attirbute and returns it. Returned as array. '''

        return vxattribute.read_attribute(attr_str)[2][0]

    @staticmethod
    def write_data(data):
        pass



    # #  implement this if you want
    # def add_vel_vec_to_frame(self,frame_arr,velocity_vector,start_point = np.array([300, 300]),size_factor = 6000):
    #     '''plots the velocity vector in the displayed frame. The velocity vector calculated is assuming
    #      the fish is orient4ed towards positive x values. In the frame it is points towards smaller x. Also numpy
    #      has a flipped y axis so we need to flip the vector both vertically and horizontally. This corresponds to
    #      rotating it by pi. '''
    #
    #     # scale and rotate velocity vector
    #     diff_vec = size_factor * np.array([[0, -1], [-1, 0]]) @ velocity_vector
    #     end_point = start_point + diff_vec
    #     end_point = [int(coord) for coord in end_point]
    #
    #     return cv2.arrowedLine(frame_arr, start_point, end_point, (0, 255, 0), 9)


    def main(self, *args, **kwargs):

        if self.initialize_motion_model:
            self.init_motion_model()
            return ()


        pose_array = self.read_data_from_attributes('tail_pose_data')


        t = np.float64(self.read_data_from_attributes('t_frame_retrival'))

        translational_speed, angular_speed = self.motion_model.main(pose_array,t)

        self.write_data_as_attribute(translational_speed,angular_speed)



