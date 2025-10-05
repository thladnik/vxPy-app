Application layer of VxPy. Requires the core [VxPy](https://github.com/thladnik/vxPy) package to run.

## Installation

Make sure you have Python 3.8 or higher installed on your system
* On Windows 11: download and install the Python 3.8+ binaries if not already installed from https://www.python.org/downloads/
* On Ubuntu: run `user@machine: ~$ sudo apt-get install python3.x` in a terminal to install desired Python version (e.g. 3.10) 

Download the repository as a ZIP file and extract it to a folder of your choice. 

* If you're using Windows 11 run the `install.bat` file in the extracted folder.
* If you're using Ubuntu, open a terminal in the extracted folder and run `bash install.sh`.'

## Running VxPy

To run the example configuration, run the `run.bat` file on Windows (double click) or `bash run.sh` in a terminal on Ubuntu.

To run your own configuration, create your own configuration file (see `configurations/example.yaml` for reference) and run (in a terminal):
* Windows: `.\run.bat path/to/your_config.yaml` 
* Ubuntu: `bash run.sh path/to/your_config.yaml`

Alternatively, if you plan on adding your own stimuli or analysis routines, follow the installation steps in the core documentation at [VxPy](https://github.com/thladnik/vxPy) 

> [!NOTE]  
> The first time when running the example configuration, it may take a while for everything to start. Even after VxPy has started, the Camera process monitor may be stuck on "Starting" while VxPy is downloading the camera sample files.
