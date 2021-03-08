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

# Fields
SAVE_DIRECTORY="YOUR_PATH"

def get_intensity(_slc):
    return 20*np.log10(abs(_slc)) - 83.0 - 32.0

def normalization(_x):
    return (_x-np.amin(_x))/(np.amax(_x)-np.amin(_x))

def adjust_value(_x):
    _x = np.asarray(normalization(_x)*255, dtype="uint8")
    return cv2.equalizeHist(_x)

def slc2pauli(_hh, _hv, _vv, _vh, _cross=0):
    if _cross == 0:
        _cross = (_hv + _vh) / 2
    else:
        _cross = _hv
    single = ((_hh+_vv)/np.sqrt(2)) # single, odd-bounce.
    double = ((_hh-_vv)/np.sqrt(2)) # double, even-bounce
    vol = np.sqrt(2)*_cross # vol
    return np.asarray([single, double, vol])

def grayplot(_file_name, _v):
    plt.figure()
    #plt.imshow(_v, cmap = "gray")
    #plt.savefig(filename)
    plt.imsave(_file_name, _v, cmap = "gray")
    print("[Save]", _file_name)

def rgbplot(_file_name, _alpha_r, _gamma_g, _beta_b):
    plttot=(np.dstack([_alpha_r, _gamma_g, _beta_b]))
    plttot=np.asarray(plttot, dtype="uint8")
    plt.figure()
    plt.imshow(plttot)
    plt.imsave(_file_name, plttot)
    print("[Save]", _file_name)
    return

# Entry point
def main():
    product_list = os.listdir(SAVE_DIRECTORY)
    for product_name in product_list:
        slc_list = glob.glob(SAVE_DIRECTORY + product_name + "/IMG-*.pkl")
        for slc_name in slc_list:
            with open(slc_name, "rb") as f:
                slc = pickle.load(f)
            if "IMG-HV" in slc_name:
                slc_hv = slc.data
            if "IMG-HH" in slc_name:
                slc_hh = slc.data
            if "IMG-VV" in slc_name:
                slc_vv = slc.data
            if "IMG-VH" in slc_name:
                slc_vh = slc.data
            slc_tmp = get_intensity(slc.data)
            grayplot(slc_name + "_p.png", adjust_value(slc_tmp))
            slc = None
            gc.collect()
if __name__=="__main__":
       main()

