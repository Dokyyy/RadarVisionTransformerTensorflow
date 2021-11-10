import sys
import getpass
import torch

# sets variable to tell if we are running in debug mode
if __name__ == '__main__':
    DEBUG = sys.gettrace() is not None
else:
    DEBUG = False

USER = getpass.getuser()

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

debug_mode = sys.gettrace() is not None
if debug_mode:
    # run everything on the cpu
    tf.config.run_functions_eagerly(True)
else:
    # run normally on the gpu
    pass

### FILE PATHS ###
if USER == 'jskaggs93':
    PATH_TO_FOOTPRINTS = '/home/jskaggs93/Datasets/WeatherRadar/chips'
    PATH_TO_SAR = '/home/jskaggs93/Datasets/eo_sar_256'
else:
    PATH_TO_FOOTPRINTS = '/mnt/nvme1n1p1/MATLAB/CS/CS_650/chips'
    PATH_TO_SAR = '/home/spencer/PycharmProjects/CycleGAN/CycleGAN/datasets/eo_sar_256'

### Model Configuration ###
BATCH_SIZE = 60
EPOCHS = 100

pixel_loss = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
multigpu = True

STEP_X = 16
STEP_Y = 16
