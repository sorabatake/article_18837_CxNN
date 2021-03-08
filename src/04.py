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
import gc
import random as rd
# keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import (Layer, InputLayer, Conv2D, Lambda)
from tensorflow.keras.models import Model
from tensorflow import keras

# custom class
class Cx2DAffine(Layer):
  def __init__(self, output_dim, activation, **kwargs):
    self.output_dim = output_dim
    self.activation = activation
    super(Cx2DAffine, self).__init__(**kwargs)

  def build(self, input_shape):
    # initialization
    self.weight_real = self.add_weight(name='weight_real',
                                  shape=(input_shape[2], self.output_dim),
                                  initializer='glorot_uniform')

    self.weight_imag = self.add_weight(name='weight_imag',
                                  shape=(input_shape[2], self.output_dim),
                                  initializer='glorot_uniform')

    self.bias_real   = self.add_weight(name='bias_real',
                                  shape=(1, self.output_dim),
                                  initializer='zeros')

    self.bias_imag   = self.add_weight(name='bias_imag',
                                  shape=(1, self.output_dim),
                                  initializer='zeros')

    super(Cx2DAffine, self).build(input_shape)

  def call(self, x):
    # input
    x_real = Lambda(lambda x: x[:, 0, :], output_shape=(x.shape[2], ))(x) # real
    x_imag = Lambda(lambda x: x[:, 1, :], output_shape=(x.shape[2], ))(x) # imag

    # computation according to mpx
    real = K.dot(x_real, self.weight_real) - K.dot(x_imag, self.weight_imag)
    imag = K.dot(x_real, self.weight_imag) + K.dot(x_imag, self.weight_real)

    real = real + self.bias_real
    imag = imag + self.bias_imag

    # activation
    if self.activation == 'normalize':
      length = K.sqrt( K.pow(real, 2) + K.pow(imag, 2) )
      real = real / length
      imag = imag / length

    # expand for concatenation
    real = K.expand_dims(real, 1)
    imag = K.expand_dims(imag, 1)

    # merge mpx
    cmpx = keras.layers.concatenate([real, imag], axis=1)
    return cmpx

  def compute_output_shape(self, input_shape):
    return(input_shape[0], 2, self.output_dim)

def loss_z(y, y_hat):
    y_real = Lambda(lambda x: x[:, 0, :])(y) # real
    y_imag = Lambda(lambda x: x[:, 1, :])(y) # imag
    y_hat_real = Lambda(lambda x: x[:, 0, 0])(y_hat) # real
    y_hat_imag = Lambda(lambda x: x[:, 1, 0])(y_hat) # imag
    y_hat_real = K.expand_dims(y_hat_real, 1)
    y_hat_imag = K.expand_dims(y_hat_imag, 1)
    errors_real = y_real - y_hat_real
    errors_imag = y_imag - y_hat_imag
    errors = K.abs(errors_real) +  K.abs(errors_imag)
    return  K.min(errors, axis=1)

def class_to_label(_class, _categories=2, _periodicity=2):
  target_class = (_class + 0.5 + (_categories * np.arange(_periodicity))) / (_categories * _periodicity) * 2 * 3.14159265358979323846
  target_class = np.exp(1j*target_class)
  return_value = []
  return_value = np.empty((2, _periodicity))

  for i in range(_periodicity):
    return_value[0, i] = target_class.real[i]
    return_value[1, i] = target_class.imag[i]

  return np.array(return_value, dtype='float32')

def label_to_class(_estimate, _categories=2, _periodicity=2):
  angle = np.mod(np.angle(_estimate) + 2*np.pi, 2*np.pi)
  p = int(np.floor (_categories * _periodicity * angle / (2*np.pi)))
  return np.mod(p, _categories)

def main():
  # preparing dataset
  train_x_dataset = np.load("./data/train/train_x.npy")
  train_y_dataset = np.load("./data/train/train_y.npy")
  test_x_dataset = np.load("./data/test/test_x.npy")
  test_y_dataset = np.load("./data/test/test_y.npy")

  train_x_dataset = train_x_dataset.reshape((train_x_dataset.shape[0], -1, 2))
  train_x_dataset = np.transpose(train_x_dataset, (0, 2, 1))
  train_y_dataset = np.argmax(train_y_dataset, axis=1)

  periodicity = 2
  label_data = np.empty((0, 2, periodicity))
  for i in range(train_y_dataset.shape[0]):
    label_data = np.append(label_data, [class_to_label(train_y_dataset[i])], axis=0)
  train_y_dataset = label_data

  test_x_dataset = test_x_dataset.reshape((test_x_dataset.shape[0], -1, 2))
  test_x_dataset = np.transpose(test_x_dataset, (0, 2, 1))
  test_y_dataset = np.argmax(test_y_dataset, axis=1)

  # build architecture
  input = Input(shape=(2, 70*70,))
  cxnn = Cx2DAffine(128, activation='normalize')(input)
  cxnn = Cx2DAffine(128, activation='normalize')(cxnn)
  cxnn = Cx2DAffine(128, activation='normalize')(cxnn)
  cxnn = Cx2DAffine(1, activation='normalize')(cxnn)
  cxnn_model = Model(input, cxnn)
  cxnn_model.compile(optimizer='adam', loss=loss_z)
  print(cxnn_model.summary())

  # train
  cxnn_model.fit(train_x_dataset, train_y_dataset, epochs=1000, batch_size=5, shuffle=True)

  # prediction
  result = cxnn_model.predict(test_x_dataset)
  for i, _tmp in enumerate(result):
    mlx = _tmp[0] + 1j*_tmp[1]
    print("prediction:", label_to_class(mlx))
    print("class:", test_y_dataset[i])
    print("---------")

if __name__=="__main__":
  main()

