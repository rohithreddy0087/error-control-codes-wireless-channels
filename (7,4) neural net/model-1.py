import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model

from channel import noise
## (7,4) error control code
## first model with error control code using 'tanh' as activation function
## at the ende of encoder
class Encoder(Model):
    def __init__(self,err_dim):
        super(Encoder, self).__init__()
        self.encode = tf.keras.Sequential([
                          layers.Dense(err_dim, activation='relu'),
                          layers.Dense(err_dim, activation='tanh'),
                        ])
    def call(self, x):
        encoded = self.encode(x)
        return encoded

class Decoder(Model):
    def __init__(self,code_dim):
        super(Decoder, self).__init__()
        self.decode = tf.keras.Sequential([
                          layers.Dense(code_dim,input_shape = (7,), activation='relu'),
                          layers.Dense(code_dim, activation='sigmoid')
                        ])

    def call(self, x):
        decoded = self.decode(x)
        return decoded

class Autoencoder(Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = Encoder(err_dim=7)
    self.decoder = Decoder(code_dim=4)
  def call(self,msg):
    x = msg[:,0:4]
    x = tf.cast(x,tf.int32)
    encoded = self.encoder(x)
    n = noise(msg[:,4:5])
    r = tf.math.add(encoded,n)
    decoded = self.decoder(r)
    return decoded,encoded
