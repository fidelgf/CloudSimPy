import tensorflow as tf
import numpy as np

class BrainBig(tf.keras.Model):
    name = 'BrainBig'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(3, input_shape=(None, state_size), activation=tf.tanh)
        self.dense_2 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_3 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_4 = tf.keras.layers.Dense(18, activation=tf.tanh)
        self.dense_5 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_6 = tf.keras.layers.Dense(1)

    def call(self, state):
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        state = self.dense_4(state)
        state = self.dense_5(state)
        state = self.dense_6(state)
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)


class Brain(tf.keras.Model):
    name = 'Brain'

    def __init__(self, state_size):
        super().__init__()
        #self.dense_1 = tf.keras.layers.Dense(3, input_shape=(None, state_size), activation=tf.tanh)
        #self.dense_2 = tf.keras.layers.Dense(9, activation=tf.tanh)
        #self.dense_3 = tf.keras.layers.Dense(18, activation=tf.tanh)
        #self.dense_4 = tf.keras.layers.Dense(9, activation=tf.tanh)
        #self.dense_5 = tf.keras.layers.Dense(1)
        self.denseltsm_1 = tf.keras.layers.LSTM(units=3, input_shape=(state_size,1), return_sequences=True)
        self.denseltsm_2 = tf.keras.layers.LSTM(units=3, return_sequences=True)
        self.denseltsm_3 = tf.keras.layers.LSTM(units=3, return_sequences=True)
        self.denseltsm_4 = tf.keras.layers.LSTM(units=3)
        self.denseltsm_5 = tf.keras.layers.Dense(units=1)

    def call(self, state):
        #state = self.dense_1(state)
        #state = self.dense_2(state)
        #state = self.dense_3(state)
        #state = self.dense_4(state)
        #state = self.dense_5(state)
        state = np.reshape(state, (state.shape[0],state.shape[1], 1))
        state = self.denseltsm_1(state)
        state = self.denseltsm_2(state)
        state = self.denseltsm_3(state)
        state = self.denseltsm_4(state)
        state = self.denseltsm_5(state)
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)


class BrainSmall(tf.keras.Model):
    name = 'BrainSmall'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(3, input_shape=(None, state_size), activation=tf.tanh)
        self.dense_2 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_3 = tf.keras.layers.Dense(6, activation=tf.tanh)
        self.dense_4 = tf.keras.layers.Dense(1)


    def call(self, state):
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        state = self.dense_4(state)
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)

class LSTMBrain(tf.keras.Model):
    name = 'LSTMBrain'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.LSTM(units=3, input_shape=(state_size,1), return_sequences=True)
        self.dense_2 = tf.keras.layers.LSTM(units=3)
        self.dense_3 = tf.keras.layers.Dense(1)

    def call(self, state):
        state = np.reshape(state, (state.shape[0], state.shape[1], 1))
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        #print("state: {}".format(state))
        #print("salida: {}".format(tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)))
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)
