# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 19:24:23 2021

@author: Dell
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses

from tensorflow.keras.models import Model

from model import Autoencoder,Encoder,Decoder
from dataset import generate_dataset
from channel import noise

train_x, train_y, test_x, test_y = generate_dataset(length = 6000)


train_ds = tf.data.Dataset.from_tensor_slices(
    (train_x)).batch(16)



model_encoder = Encoder(err_dim=7)
model_decoder = Decoder(code_dim = 4)

#loss_object = tf.keras.losses.MeanSquaredError()
lamda = 0.001

def loss_object(C):
    L = []
    D = tf.zeros((16,16))
    C=tf.transpose(C)
    n = tf.ones((16,))
    for i in range(16):
        ci = tf.reshape(C[:,i],(1,7))
        cj = tf.transpose(ci)
        norm = tf.reshape(tf.matmul(ci,cj),(1,))
        L.append(norm*n)
    L = tf.stack(L)
    D = L-2*tf.matmul(tf.transpose(C),C)+tf.transpose(L)
    diag = tf.ones((16,))*1000
    D = tf.linalg.set_diag(D,diag)
    return -1*lamda*tf.reduce_min(D)

mse = tf.keras.losses.MeanSquaredError()
optimizer_encoder = tf.keras.optimizers.Adam()
optimizer_decoder = tf.keras.optimizers.Adam()

train_loss_encoder = tf.keras.metrics.Mean(name='train_loss_encoder')
train_loss_decoder = tf.keras.metrics.Mean(name='train_loss_decoder')

reg_param = 0.01
#@tf.functions
def train_step(msg):
  with tf.GradientTape() as encoder_tape,tf.GradientTape() as decoder_tape:
    x = msg[:,0:4]
    x =  2*x-1
    encoded = model_encoder(x, training=True)
    Es = tf.math.reduce_mean(encoded**2,axis = 1)
    Es = tf.ones_like(Es)
    snr = tf.cast(msg[:,4:5],tf.float32)
    n = noise(snr,Es)
    r = tf.math.add(encoded,n)
    decoded = model_decoder(r)
    # predictions = predictions >= 0
    # predictions  = tf.cast(predictions ,dtype = tf.float32)
    # predictions  = 2*predictions-1
    
    decoder_loss = mse(x,decoded)
    encoder_loss = decoder_loss#loss_object(encoded) + reg_param*decoder_loss
    # print(reg_param*decoder_loss)
    # print(loss_object(encoded))
    # f
  gradients_of_encoder = encoder_tape.gradient(encoder_loss, model_encoder.trainable_variables)
  gradients_of_decoder = decoder_tape.gradient(decoder_loss, model_decoder.trainable_variables)

  optimizer_encoder.apply_gradients(zip(gradients_of_encoder, model_encoder.trainable_variables))
  optimizer_decoder.apply_gradients(zip(gradients_of_decoder, model_decoder.trainable_variables))

  train_loss_encoder(encoder_loss)
  train_loss_decoder(decoder_loss)


EPOCHS = 1000

checkpoint_train_encoder = tf.train.Checkpoint(model_encoder)
manager_train_encoder = tf.train.CheckpointManager(
    checkpoint_train_encoder, directory="ckpts-1/model_encoder", max_to_keep=1)
checkpoint_train_decoder = tf.train.Checkpoint(model_decoder)
manager_train_decoder = tf.train.CheckpointManager(
    checkpoint_train_decoder, directory="ckpts-1/model_decoder", max_to_keep=1)

checkpoint_train_encoder.restore('ckpts-1/model_encoder_wts-1245')
trainLossEncoder = []
trainLossDecoder = []


minLossTrainEncoder = 99
minLossTrainDecoder = 99
for epoch in range(EPOCHS):
  train_loss_encoder.reset_states()
  train_loss_decoder.reset_states()

  for vals in train_ds:
    train_step(vals)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss_encoder.result()}, '
    f'Loss: {train_loss_decoder.result()}'
  )

  if minLossTrainEncoder > train_loss_encoder.result().numpy():
      minLossTrainEncoder = train_loss_encoder.result().numpy()
      manager_train_encoder.save(1)
  if minLossTrainDecoder > train_loss_decoder.result().numpy():
      minLossTrainDecoder = train_loss_decoder.result().numpy()
      manager_train_decoder.save(1)

  trainLossEncoder.append(train_loss_encoder.result().numpy())
  trainLossDecoder.append(train_loss_decoder.result().numpy())

