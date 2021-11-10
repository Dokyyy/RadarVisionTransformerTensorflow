import os
import cv2
import glob
import multiprocessing
import numpy as np
import random
from configuration import *
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
batch_size=BATCH_SIZE


def plot(out, name):
    plt.figure()
    plt.title(name)
    # img = rearrange(out, 'c h w -> h w c').cpu().detach().numpy()
    # plt.imshow((img - np.min(img)) / (np.max(img) - np.min(img)))
    plt.imshow(out)
    plt.colorbar()
    plt.show()


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

    def __init__(self, BATCH_SIZE=batch_size, data_dir=PATH_TO_SAR + '/trainA/train'):
        self.BATCH_SIZE = BATCH_SIZE
        self.results = []
        self.batch_num = 0
        self.epoch = 0
        self.data_dir = data_dir
        self.cpu_count = multiprocessing.cpu_count()
        self.batch_norm = nn.BatchNorm2d(3)

    def load_data(self, num_processes=multiprocessing.cpu_count(), num_sections=16):
        proc = Imgloader(self.data_dir, num_patches=num_sections)
        files = os.listdir(proc.get_dir())
        pool = multiprocessing.Pool(processes=num_processes)
        self.results = pool.map(proc, files)

    def get_batch(self):
        if self.BATCH_SIZE * self.batch_num + self.BATCH_SIZE > len(self.results):
            self.batch_num = 0
            self.epoch += 1
            # random.shuffle(self.results)
        batch = self.results[self.batch_num * self.BATCH_SIZE: self.batch_num * self.BATCH_SIZE + self.BATCH_SIZE]
        self.batch_num += 1
        batch = np.array(batch)
        # out_img = np.zeros_like(batch).astype(np.float32)
        # for j in range(BATCH_SIZE):
        #     for i in range(3):
        #         c_min = np.min(batch[j, :, :, i])
        #         if (np.max(batch[j, :, :, i]) - c_min) == 0:
        #             plot(batch[j,:,:,:], 'broken')
        #         out_img[j, :, :, i] = (batch[j, :, :, i] - c_min) / (np.max(batch[j, :, :, i]) - c_min)


        return torch.as_tensor(batch.astype(np.float32)).to(device), self.batch_num, self.epoch

        # output_tensor = np.concatenate((np.ones((batch_size,batch.shape[2],1)), batch.transpose(0,2,1)), axis=2)
        # temp = torch.as_tensor(output_tensor).cuda()
        # temp = reformat_picture(temp[0,:,:-1],16,16)


        # return torch.as_tensor(batch.transpose(0,2,1)).cuda(), torch.as_tensor(output_tensor).cuda(), self.epoch


def reformat_picture(img_tensor, num_sections, patch_size):
    patches = []
    img = img_tensor.cpu().detach().numpy()
    # img = img.transpose()
    y, x = img.shape
    step_y = y // num_sections
    step_x = x // num_sections
    reconstructed_img = np.zeros((num_sections * patch_size, num_sections * patch_size))

    for i in range(num_sections ** 2):
        patches += [img[i, :].reshape((patch_size, patch_size))]
        reconstructed_img[int(i/num_sections)*patch_size:int(i/num_sections)*patch_size + patch_size, (i % num_sections)*patch_size:((i) % num_sections)*patch_size + patch_size] = patches[i]
        # cv2.imshow('t', reconstructed_img.astype('uint8'))
        # cv2.waitKey()

    # for i in range(int(np.sqrt(num_sections))):
    #     vertical = []
    #     for idx, patch in enumerate(patches[::2]):
    #         vertical += [cv2.vconcat([patch, patches[idx + 1]])]
    #     patches = vertical
    #
    # for i in range(int(np.sqrt(num_sections))):
    #     horizontal = []
    #     for idx, patch in enumerate(patches[::2]):
    #         horizontal += [cv2.hconcat([patch, patches[idx + 1]])]
    #     patches = horizontal



    # cv2.imshow('test', reconstructed_image[0])
    # cv2.waitKey()
    return reconstructed_img


if __name__ == "__main__":
    loader = DataLoader(BATCH_SIZE=BATCH_SIZE)
    loader.load_data(num_processes=24, num_sections=8)
    batch, output_tensor, epoch = loader.get_batch(BATCH_SIZE=BATCH_SIZE)
    test = reformat_picture(batch[0,:,:],8,32)
    # cv2.imshow('test', test)
    # cv2.waitKey()
    x = 1

