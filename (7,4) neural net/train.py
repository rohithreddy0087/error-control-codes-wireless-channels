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

from model import Autoencoder
from dataset import generate_dataset


train_x, train_y, test_x, test_y = generate_dataset(length = 1)


train_ds = tf.data.Dataset.from_tensor_slices(
    (train_x, train_y)).batch(16)#.shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(32)


model = Autoencoder()

loss_object = tf.keras.losses.MeanSquaredError()
#loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0,name='binary_crossentropy')

optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:

    predictions,_ = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):

  predictions,_ = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 10000


checkpoint_train = tf.train.Checkpoint(model)
manager_train = tf.train.CheckpointManager(
    checkpoint_train, directory="ckpts-1/train_model", max_to_keep=1)
checkpoint_test = tf.train.Checkpoint(model)
manager_test = tf.train.CheckpointManager(
    checkpoint_test, directory="ckpts-1/test_model", max_to_keep=1)


trainLoss = []
trainAccuracy = []
testLoss = []
testAccuracy = []

minLossTrain = 99
minLossTest = 99
for epoch in range(EPOCHS):
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  
  for vals, labels in train_ds:
    train_step(vals, labels)

  for vals, test_labels in test_ds:
    test_step(vals, test_labels)


  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )
  

  if minLossTrain > train_loss.result().numpy():
      minLossTrain = train_loss.result().numpy()
      manager_train.save(1)
  if minLossTest > test_loss.result().numpy():
      minLossTest = test_loss.result().numpy()
      manager_test.save(1)
  trainLoss.append(train_loss.result().numpy())
  testLoss.append(test_loss.result().numpy())
  trainAccuracy.append(train_accuracy.result().numpy() * 100)
  testAccuracy.append(test_accuracy.result().numpy() * 100)

print('Minimum training loss is '+str(minLossTrain))
print('Minimum test loss is '+str(minLossTest))
