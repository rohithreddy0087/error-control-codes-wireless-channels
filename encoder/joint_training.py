# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 19:07:20 2021

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

from model import Autoencoder
from dataset import generate_dataset

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:

    predictions,en = model(images, training=True)
    
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)

  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)


train_x, train_y, test_x, test_y = generate_dataset(length = 6000)


train_ds = tf.data.Dataset.from_tensor_slices(
    (train_x,train_y)).batch(16)



model = Autoencoder()

loss_object = tf.keras.losses.MeanSquaredError()
#loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0,name='binary_crossentropy')

optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')

EPOCHS = 10


checkpoint_train = tf.train.Checkpoint(model)
manager_train = tf.train.CheckpointManager(
    checkpoint_train, directory="ckpts-1/train_model", max_to_keep=1)
model.encoder.load_weights('ckpts-1/model_encoder_only_joint/ckpt-1')
#checkpoint = tf.train.Checkpoint(model_encoder)
#checkpoint.restore('ckpts-1/model_encoder_only/ckpt-1')
print('Encoder model loaded')
trainLoss = []

minLossTrain = 99
for epoch in range(EPOCHS):
  train_loss.reset_states()
  
  for vals, labels in train_ds:
    train_step(vals, labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
  )
  
  if minLossTrain > train_loss.result().numpy():
      minLossTrain = train_loss.result().numpy()
      manager_train.save(1)

  trainLoss.append(train_loss.result().numpy())

