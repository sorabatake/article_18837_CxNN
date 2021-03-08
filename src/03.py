import os
import matplotlib as mpl
mpl.use('Agg')
import struct
import numpy as np
import glob
from tqdm import tqdm
import pickle
import cv2
import matplotlib.pyplot as plt
from slcinfo import SLC_L11
import gc
import random as rd

def get_intensity(_slc):
    return 20*np.log10(abs(_slc)) - 83.0 - 32.0

def adjust_value(_x):
    _x = np.asarray(normalization(_x)*255, dtype="uint8")
    return cv2.equalizeHist(_x)

def grayplot(_file_name, _v):
    plt.figure()
    plt.imsave(_file_name, _v, cmap = "gray")
    print("[Save]", _file_name)

def normalization(_x):
    return (_x-np.amin(_x))/(np.amax(_x)-np.amin(_x))

bridge_center = np.load("./bridge_center.npy")
noise_data_cog_x = 1064
noise_data_cog_y = 1650
noise_data_range = 600

train_bc = bridge_center[:-3]
test_bc = bridge_center[-3:]
save_slc_path =  "YOUR_PATH"
random_num = 5
image_size = 70

train_x_dataset = np.zeros(((random_num*train_bc.shape[0]*2), image_size, image_size, 2))
train_y_dataset = np.zeros(((random_num*train_bc.shape[0]*2), 2))

test_x_dataset = np.zeros(((test_bc.shape[0]*2), image_size, image_size, 2))
test_y_dataset = np.zeros(((test_bc.shape[0]*2), 2))

with open(save_slc_path, "rb") as f:
  slc_data = pickle.load(f)

  # train
  data_counter = 0
  for i in range(train_bc.shape[0]):
    for j in range(random_num):
      offset = rd.randint(-4, 4)
      tmp_x = train_bc[i, 0] + offset
      tmp_y = train_bc[i, 1] + offset
      slc_tmp = slc_data.extract_data_cog(int(tmp_x), int(tmp_y), int(image_size/2))
      train_x_dataset[data_counter, :, :, 0] = slc_tmp.real
      train_x_dataset[data_counter, :, :, 1] = slc_tmp.imag
      train_y_dataset[data_counter, 0] = 1
      slc_tmp = get_intensity(slc_tmp)
      grayplot("./data/train/" + str(data_counter)  + "_v.png", adjust_value(slc_tmp))
      data_counter += 1

  for i in range(int(train_x_dataset.shape[0]*0.5)):
    offset = rd.randint(-int(noise_data_range*0.5), int(noise_data_range*0.5))
    tmp_x = noise_data_cog_x + offset
    tmp_y = noise_data_cog_y + offset
    slc_tmp = slc_data.extract_data_cog(int(tmp_x), int(tmp_y), int(image_size/2))
    train_x_dataset[data_counter, :, :, 0] = slc_tmp.real
    train_x_dataset[data_counter, :, :, 1] = slc_tmp.imag
    train_y_dataset[data_counter, 1] = 1
    slc_tmp = get_intensity(slc_tmp)
    grayplot("./data/train/" + str(data_counter)  + "_v.png", adjust_value(slc_tmp))
    data_counter += 1

  np.save("./data/train/train_x.npy", train_x_dataset)
  np.save("./data/train/train_y.npy", train_y_dataset)

  # test
  data_counter = 0
  for i in range(test_bc.shape[0]):
    tmp_x = test_bc[i, 0]
    tmp_y = test_bc[i, 1]
    slc_tmp = slc_data.extract_data_cog(int(tmp_x), int(tmp_y), int(image_size/2))
    test_x_dataset[data_counter, :, :, 0] = slc_tmp.real
    test_x_dataset[data_counter, :, :, 1] = slc_tmp.imag
    test_y_dataset[data_counter, 0] = 1
    slc_tmp = get_intensity(slc_tmp)
    grayplot("./data/test/" + str(data_counter)  + "_v.png", adjust_value(slc_tmp))
    data_counter += 1

  for i in range(test_bc.shape[0]):
    offset = rd.randint(-int(noise_data_range*0.5), int(noise_data_range*0.5))
    tmp_x = noise_data_cog_x + offset
    tmp_y = noise_data_cog_y + offset
    slc_tmp = slc_data.extract_data_cog(int(tmp_x), int(tmp_y), int(image_size/2))
    test_x_dataset[data_counter, :, :, 0] = slc_tmp.real
    test_x_dataset[data_counter, :, :, 1] = slc_tmp.imag
    test_y_dataset[data_counter, 1] = 1
    slc_tmp = get_intensity(slc_tmp)
    grayplot("./data/test/" + str(data_counter)  + "_v.png", adjust_value(slc_tmp))
    data_counter += 1

  np.save("./data/test/test_x.npy", test_x_dataset)
  np.save("./data/test/test_y.npy", test_y_dataset)

