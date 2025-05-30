from vxpy import main

if __name__ == '__main__':

    # Define configuration
    _config = {
        'CALIBRATION_PATH': 'calibrations/spherical_4_screen_cylinder.yaml',
        'CAMERA_USE': True,
        'CAMERA_DEVICES': {
            'fish_embedded': {
                'api': 'vxpy.devices.camera.virtual_camera.VirtualCamera',
                'data_source': 'HDF5',
                'data_path': 'Single_Fish_Spontaneous_1_115fps',
                'preload_data': False,
                'width': 720,
                'height': 480,
                'frame_rate': 115
            },
            'fish_swimming': {
                'api': 'vxpy.devices.camera.virtual_camera.VirtualCamera',
                'data_source': 'HDF5',
                'data_path': 'Multi_Fish_Vertical_Swim_10fps',
                'preload_data': False,
                'width': 1920,
                'height': 1080,
                'frame_rate': 10
            },
            'multiple_fish_embedded': {
                'api': 'vxpy.devices.camera.virtual_camera.VirtualCamera',
                'data_source': 'HDF5',
                'data_path': 'Multi_Fish_Eyes_Cam_20fps',
                'preload_data': False,
                'width': 752,
                'height': 480,
                'frame_rate': 20
            }
        },
        'ROUTINES': {
            'vxpy.routines.camera_capture.Frames': {},
            'vxpy.plugins.zf_eyeposition_tracking.EyePositionDetectionRoutine': {
                'roi_maxnum': 1,
                'min_size': 45
            },
            'vxpy.routines.display_capture.Frames': {},
            'vxpy.routines.calculate_csd.CalculatePSD': {}
        },
        'DISPLAY_USE': True,
        'DISPLAY_FPS': 60,
        'DISPLAY_GL_VERSION': '420 core',
        'DISPLAY_TRANSFORM': 'Spherical4ChannelProjectionTransform',
        'GUI_USE': True,
        'GUI_REFRESH': 30,
        'GUI_FPS': 20,
        'GUI_SCREEN': 0,
        'GUI_ADDONS': {
            'vxpy.plugins.camera_widgets.CameraStreamAddon': {},
            'vxpy.plugins.zf_eyeposition_tracking.EyePositionDetectionAddon': {},
            'vxpy.plugins.display_widgets.VisualInteractor': {
                'detached': True,
                'preferred_size': [600, 400],
                'preferred_pos': [1200, 400]
            },
            'vxpy.plugins.display_widgets.VisualStreamAddon': {},
            'vxpy.plugins.io_core_widgets.DisplayPSD': {}
        },
        'IO_USE': True,
        'IO_DEVICES': {
            'Dev1_virtual': {
                'api': 'vxpy.devices.virtual_daq.VirtualDaqDevice',
                'pins': {
                    'on_off_1s_di': {
                        'signal': 'di',
                        'fun': 'on_off',
                        'args': {'freq': 0.5, 't_offset': 0.0}
                    },
                    'noisy_sinewave_input': {
                        'signal': 'ai',
                        'fun': 'whitenoise_sinewave',
                        'args': {'freq': 5, 't_offset': 1.0, 'nlvl': 1.0}
                    }
                }
            },
            'Dev2_virt': {
                'api': 'vxpy.devices.virtual_daq.VirtualDaqDevice',
                'pins': {
                    'slow_sinewave_input': {
                        'signal': 'ai',
                        'fun': 'sinewave',
                        'args': {'freq': 2, 't_offset': 1.0}
                    }
                }
            }
        },
        'IO_MAX_SR': 1000,
        'WORKER_USE': True,
        'REC_ENABLE': True,
        'REC_OUTPUT_FOLDER': './recordings/',
        'REC_ATTRIBUTES': {
            'ai*': {},
            'ao*': {},
            'di*': {},
            'do*': {},
            'eyepos_*': {},
            'fish_embedded_frame': {'videoformat': 'avi', 'codec': 'xvid'},
            'display_frame': {'videoformat': 'mpeg', 'codec': 'h265', 'bitrate': 5000},
            'ai_slow_sinewave_input': {}, 'test_sines_whitenoise': {}, 'sawtooth_analogin': {}
        }
    }

    # Run configuration
    main(_config)
