from configuration import *
import os
import cv2
import glob
from footprint import Footprint
import multiprocessing
import numpy as np
import random
# from config import *
import torch
import matplotlib.pyplot as plt
import scipy.io as sio
import sys

if __name__ == '__main__':
    DEBUG_DATALOARDER = sys.gettrace() is not None
else:
    DEBUG_DATALOARDER = False


class ChipLoader:

    def __init__(self, data_dir='/mnt/nvme1n1p1/MATLAB/CS/CS_650/chips'):
        self.data_dir = data_dir
        self.num_patches = 16

    def __call__(self, filename):
        chip = sio.loadmat(self.data_dir + '/' + filename)

        if chip['chip']['footprints'][0,0].size < 1:
            return

        # chip = list(chip['chip']['footprints'][0, 0][0])
        new_chip = []
        for i, foot in enumerate(chip['chip']['footprints'][0, 0][0]):  # loop through footprints
            foot_vals = []
            for j, val in enumerate(foot[0]):  # loop through footprint values
                foot_vals += [val[0]]
            new_chip += [Footprint(foot_vals)]

        for i, foot in enumerate(chip['chip']['xy'][0, 0][0]):  # loop through xy
            foot_vals = []
            for j, val in enumerate(foot[0]):  # loop through footprint values
                foot_vals += [val[0]]
            new_chip[i].xs = foot_vals

        for i, foot in enumerate(chip['chip']['pointer'][0, 0][0]):  # loop through pointer
            foot_vals = []
            for j, val in enumerate(foot[0]):  # loop through footprint values
                foot_vals += [val[0]]
            new_chip[i].pointers = foot_vals

        img = chip['chip']['img'][0,0]

        patches = []
        patch_size = STEP_X
        img = np.array(img)
        for i in range(self.num_patches):
            for j in range(self.num_patches):
                patches += [img[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size].reshape((1, STEP_X * STEP_Y))]
                # temp_patch = img[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size].reshape((1,step_x * step_y))
                # test[i*j, :] = temp_patch
                # cv2.imshow('t', test)
                # cv2.waitKey(15)
        patches = np.array(patches)


        imgs = patches[0]
        for i in range(len(patches) - 1):
            imgs = cv2.vconcat([imgs,patches[i + 1]])
            # cv2.imshow('t', imgs)
            # cv2.waitKey(10)

        # cv2.waitKey()

        return new_chip, patches

    def get_dir(self):
        return self.data_dir


class Imgloader:
    def __init__(self, data_dir=PATH_TO_SAR + '/trainA/train', num_patches=16):
        self.data_dir = data_dir
        # num_patches right now has a to be a power of 2
        self.num_patches = num_patches

    def __call__(self, filename):
        img = cv2.imread(self.data_dir + '/' + filename)
        if np.mean(img) == 0:
            print(filename)

        return img

    def get_dir(self):
        return self.data_dir


class DataLoader:

    def __init__(self, BATCH_SIZE=BATCH_SIZE, data_dir=PATH_TO_SAR + '/trainA/train/', cpu_count=multiprocessing.cpu_count()):
        self.BATCH_SIZE = BATCH_SIZE
        self.results = []
        self.batch_num = 0
        self.epoch = 0
        self.data_dir = data_dir
        self.cpu_count = cpu_count
        self.read_loc = 0

    def load_data(self, BATCH_SIZE=BATCH_SIZE):
        proc = Imgloader(self.data_dir)
        files = os.listdir(proc.get_dir())
        files.sort()
        files = files[self.read_loc:self.read_loc + BATCH_SIZE]
        self.read_loc += BATCH_SIZE
        pool = multiprocessing.Pool(processes=self.cpu_count)
        self.results = pool.map(proc, files)
        return self.results

    def get_batch(self):
        batch = self.load_data(BATCH_SIZE)
        return np.array(batch, dtype=float)


if __name__ == "__main__":
    if DEBUG_DATALOARDER:
        loader = DataLoader(cpu_count=1)
    else:
        loader = DataLoader(cpu_count=24)

    batch = loader.get_batch()
    # cv2.imshow('test', test)
    # cv2.waitKey()
    x = 1

