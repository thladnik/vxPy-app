import sys

from vxpy.calibration_manager import run_calibration

if __name__ == '__main__':
    run_calibration(sys.argv[1])
