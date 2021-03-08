import os
import struct
import numpy as np
import glob
import pickle
import gc
from slcinfo import SLC_L11

# Fields
SAVE_DIRECTORY="YOUR_PATH"

# Entry point
def main():
    product_list = os.listdir(SAVE_DIRECTORY)
    for product_name in product_list:
        slc_list = glob.glob(SAVE_DIRECTORY + product_name + "/IMG-*_D")
        for slc_name in slc_list:
            print(slc_name)
            slc = SLC_L11(slc_name)
            slc.parse()
            slc.crop_data(1500, 15000, 5000) # if you use cropping
            with open(slc_name + ".pkl", "wb") as f:
                pickle.dump(slc, f)
            slc = None
            gc.collect()

if __name__=="__main__":
       main()
