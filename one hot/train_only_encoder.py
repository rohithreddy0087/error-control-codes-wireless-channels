# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 19:33:55 2021

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

train_x, train_y, test_x, test_y = generate_dataset(length = 1)


train_ds = tf.data.Dataset.from_tensor_slices(
    (train_x)).batch(16)

model_encoder = Encoder(err_dim=7)

lamda = 1
def normalize(x):
    return x/np.linalg.norm(x,ord = 3)
def loss_object(C):
    #C =  tf.math.add(C,1)
    #C =  tf.math.divide(C,2)
    #C = tf.math.round(C)
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
    #L = normalize(L)
    T = 2*tf.matmul(tf.transpose(C),C)
    #T = normalize(T)
    D = L-T+tf.transpose(L)
    #D = norm
    diag = tf.ones((16,))*10
    D = tf.linalg.set_diag(D,diag)
    return -1*lamda*tf.reduce_min(D)

optimizer_encoder = tf.keras.optimizers.Adam()

train_loss_encoder = tf.keras.metrics.Mean(name='train_loss_encoder')


#@tf.functions
def train_step(msg):
  with tf.GradientTape() as encoder_tape:
    x = msg[:,0:4]
    encoded = model_encoder(x, training=True)
    # predictions = encoded >= 0
    # predictions  = tf.cast(predictions ,dtype = tf.float32)
    # predictions  = 2*predictions-1
    encoder_loss = loss_object(encoded)
  gradients_of_encoder = encoder_tape.gradient(encoder_loss, model_encoder.trainable_variables)

  optimizer_encoder.apply_gradients(zip(gradients_of_encoder, model_encoder.trainable_variables))

  train_loss_encoder(encoder_loss)


EPOCHS = 100000
checkpoint_train_encoder = tf.train.Checkpoint(model_encoder)
#checkpoint_train_encoder.restore('ckpts-1/model_encoder_only2/ckpt-1')
manager_train_encoder = tf.train.CheckpointManager(
    checkpoint_train_encoder, directory="ckpts-1/model_encoder_only3", max_to_keep=1)


trainLossEncoder = []


minLossTrainEncoder = 99

for epoch in range(EPOCHS):
  train_loss_encoder.reset_states()

  for vals in train_ds:
    train_step(vals)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss_encoder.result()}, '
  )

  if minLossTrainEncoder > train_loss_encoder.result().numpy():
      minLossTrainEncoder = train_loss_encoder.result().numpy()
      manager_train_encoder.save(1)


  trainLossEncoder.append(train_loss_encoder.result().numpy())
