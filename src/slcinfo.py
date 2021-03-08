import numpy as np
from tqdm import tqdm # !pip install tqdm
import struct
import copy as cp
class SLC_L11:
    def __init__(self, _file_name):
        self.file_name = _file_name
        self.fp = 0
        self.width = 0
        self.height = 0
        self.level = 0
        self.format = ""
        self.data = []

    def get_width(self):
        self.fp.seek(248)
        self.width = int(self.fp.read(8))
        print("width: ", self.width)

    def get_height(self):
        self.fp.seek(236)
        self.height = int(self.fp.read(8))
        print("height: ", self.height)

    def get_level(self):
        self.fp.seek(276) # if l1.1, return "544"
        self.level = int(self.fp.read(4))

    def get_format(self):
        self.fp.seek(400)
        self.format = self.fp.read(28).decode()

    def get_data(self):
        self.fp.seek(720)
        prefix_byte = 544
        data = np.zeros((self.height, self.width, 2))
        width_byte_length = prefix_byte + self.width*8 # prefix + width * byte (i32bit + q32bit)
        for i in tqdm(range(self.height)):
            data_line = struct.unpack(">%s"%(int((width_byte_length)/4))+"f", self.fp.read(int(width_byte_length)))
            data_line = np.asarray(data_line).reshape(1, -1)
            data_line = data_line[:, int(prefix_byte/4):] # float_byte:
            slc = data_line[:,::2] + 1j*data_line[:,1::2]
            data[i, :, 0] = slc.real
            data[i, :, 1] = slc.imag
        # multiplex
        self.data = data[:, :, 0] + 1j*data[:, :, 1]
        print("done")

    def crop_data(self, _offset_x, _offset_y, _size):
        self.data = self.data[_offset_y:(_offset_y+_size), _offset_x:(_offset_x+_size)]

    def extract_data_cog(self, _offset_x, _offset_y, _size):
        return cp.deepcopy(self.data[(_offset_y-_size):(_offset_y+_size), (_offset_x-_size):(_offset_x+_size)])

    def parse(self):
        self.fp = open(self.file_name, mode='rb')
        self.get_width()
        self.get_height()
        self.get_level()
        self.get_format()
        self.get_data()
        self.fp.close()
        self.fp = 0

