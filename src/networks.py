import tensorflow as tf


class Actor(tf.keras.models.Model):
    def __init__(self, states=3, actions=1, units=(512, 512), activation="relu", optimizer="Adam", noise=0.1, **kwargs):
        super().__init__(**kwargs)
        self.state = states
        self.action = actions
        self.units = units
        self.activation = activation
        input = tf.keras.Input(shape=(states,))
        dense = [input]
        dense.extend([tf.keras.layers.Dense(unit, activation) for unit in self.units])
        dense.append(tf.keras.layers.Dense(actions, activation="tanh"))
        self.model = tf.keras.models.Sequential(dense)
        self.compile(optimizer=optimizer)
        self.noise = noise

    def call(self, inputs, training=None, mask=None, exploration=False):
        out = self.model(inputs)
        if exploration:
            out = out + tf.random.normal(stddev=self.noise, shape=(self.action,))
        return out


class Critic(tf.keras.models.Model):
    def __init__(self, states=3, actions=1, q_values=1, units=(512, 512), activation="relu", optimizer="Adam", **kwargs):
        super().__init__(**kwargs)
        self.state = states
        self.actions = actions
        self.q_values = q_values
        self.units = units
        self.activation = activation
        input_state = tf.keras.Input(shape=(states,))
        input_action = tf.keras.Input(shape=(actions,))
        concat = tf.keras.layers.Concatenate(-1)([input_state, input_action])
        dense = list()
        dense.extend([tf.keras.layers.Dense(unit, activation) for unit in self.units])
        dense.append(tf.keras.layers.Dense(q_values, activation="linear"))
        for i, l in enumerate(dense):
            if i == 0:
                x = l(concat)
            else:
                x = l(x)
        self.model = tf.keras.models.Model([input_state, input_action], x)
        self.compile(optimizer=optimizer)

    def call(self, inputs, training=None, mask=None, noised=False):
        return self.model(inputs)